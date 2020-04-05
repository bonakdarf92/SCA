from DarmstadtNetwork import DarmstadtNetwork
import networkx as nx 
import numpy as np 
import cvxpy as cp 
from tqdm import tqdm 
import gurobipy 
import stela as st 
import pygsp as ps 


class Solver:

    def __init__(self, solver_type, signal, Graph):
        self.type = solver_type
        #self.solution = None
        #self.cost = None 
        #self.settigns = None 
        #self.variable = None 
        self.signals = signal
        self.Graph = Graph
        #self.problem = None 
        self.lambd = 2*np.max(self.signals) - 1

    def cut_based(self):
        self.variable = cp.Variable(self.Graph.n_vertices, boolean=True)
        lookup = self.Graph.A.astype(np.double)
        indices = lookup.tocsr().nonzero()
        second = []
        for k in range(len(lookup.nonzero()[0])):
            second.append(cp.abs(self.variable[indices[0][k]] - self.variable[indices[1][k]]))
    
        object_cut = cp.Minimize(cp.sum_squares(self.signals - self.variable) + self.lambd * cp.sum(second))
        self.problem = cp.Problem(object_cut)
        self.problem.solve(solver=cp.GUROBI)

    def path_based2(self, stGraph, verbose=False, threshold=None, dictionary=None, save=False):
        self.path_based()
        preselection = self.variable
        if dictionary:
            print("Lade Dictionary")
            self.pathDic = np.load('./PathDic_20_7.npz', allow_pickle=True)['arr_0']
        else:
            print("Erstelle Simplen Graphen")
            paths = nx.all_simple_paths(stGraph, source='s', target='t', cutoff=20)
            self.Adjacency = self.Graph.A.astype(np.double)
            list_paths = [p for p in paths if len(p) > 7]
            self.pathDic = np.zeros((self.Graph.n_vertices, len(list_paths)))
            count = 0
            s = [t for t in stGraph.nodes._nodes]
            print("Bestimme Dictionary")
            for k in list_paths:
                k.pop(0)
                k.pop()
                locs = [s.index(t) for t in k]
                self.pathDic[locs, count] = 1
                count += 1
            for kick in preselection:
                pass
            tmp = np.max(self.Adjacency * self.pathDic, 0)
            self.pathDic = self.pathDic[:, tmp < 3]
        if save:
            np.savez('./PathDic_20_7.npz', self.pathDic)
            print("Dictionary gespeichert")
        print("Beginne Optimierung")
        self.variable = cp.Variable(len(self.pathDic[0]), nonneg=True)
        #constraint = [cp.sum(self.variable) == 1]#, cp.norm(self.Adjacency@self.pathDic@self.variable,'inf') <=2]
        # constraint = [cp.norm(self.Adjacency @ self.variable, 'inf') <= 2, cp.sum(self.variable)- 0.5*cp.sum(self.Adjacency @ self.variable) - 1 == 0 ]
        # object_farid = cp.Minimize(cp.sum_squares(self.signals - self.variable) + self.lambd * cp.norm(W @ self.variable,1) )
        object_farid = cp.Minimize(0.5 * cp.sum_squares(self.signals - self.pathDic@self.variable) + self.lambd * cp.norm(self.variable, 1) )#- 0.01*cp.sum(cp.log(self.variable)))# + self.lambd*cp.norm1(self.variable))
        self.problem = cp.Problem(object_farid)#, constraint)
        self.problem.solve(solver=cp.GUROBI, verbose=verbose)
        # self.solution = pathDic @ self.variable
        # self.solution = self.pathDic[:, np.argmax(self.variable)]
        # self.variable.value[self.variable.value <= 1e-2] = 0
        x = self.variable.value.copy()
        x[x <= 1e-4] = 0
        # self.recons = [self.pathDic[:, k] * self.variable[k] for k in self.variable.value.nonzero()[0][:]]
        self.recons = [self.pathDic[:, k] * x[k] for k in x.nonzero()[0][:]]
        sol = cp.sum(self.recons[:])
        if threshold:
            sol[sol <= threshold] = 0
        else:
            sol[sol <= np.var(self.signals)] = 0
        self.solution = sol

    def path_based(self, verbose=False, threshold=True):
        self.Adjacency = self.Graph.A.astype(np.double)
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        constraint = [cp.norm(self.Adjacency @ self.variable, 'inf') <= 2, self.variable <= 1]
        object_path = cp.Minimize(cp.sum_squares(self.signals - self.variable) + self.lambd * cp.norm(self.variable, 1))
        self.problem = cp.Problem(object_path, constraint)
        self.problem.solve(solver=cp.GUROBI,verbose=verbose)
        if threshold:
            self.variable.value[self.variable.value <= self.lambd] = 0
            self.variable.value[self.variable.value != 0] = 1
        if len(self.variable.value.nonzero()[0]) != 0:
            W_ = self.Adjacency.copy()
            rows, cols = [k for k in W_.nonzero()[0]], [k for k in W_.nonzero()[1]]
            W = W_[np.ix_(self.variable.value.nonzero()[0], self.variable.value.nonzero()[0])]
            self.subAdjacency = self.Adjacency[np.ix_(self.variable.value.nonzero()[0],self.variable.value.nonzero()[0])]
            self.sparse_var = cp.Variable(self.subAdjacency.shape[0], boolean=True)
            constraint2 = [2*cp.sum(self.sparse_var) - cp.sum(self.subAdjacency @ self.sparse_var) == 2]
            object_path_sparse = cp.Minimize(cp.norm(self.signals[np.ix_(self.variable.value.nonzero()[0])] - self.sparse_var) - np.ones(self.subAdjacency.shape[0],)@self.sparse_var)
            self.subProblem = cp.Problem(object_path_sparse, constraint2)
            self.subProblem.solve(solver=cp.GUROBI, verbose=verbose)
            # reconstruct = dict(zip(self.variable.value.nonzero()[0],self.subProblem.value))
            # print(reconstruct)
            print("Anzahl der aktivierte Knoten {0}  Anzahl der optimierten Knoten {1}".format(len(self.variable.value.nonzero()[0]), len(self.sparse_var.value.nonzero()[0])))

    def path_farid(self, verbose=False):
        self.Adjacency = self.Graph.A.astype(np.double)

        resid = (np.max(self.signals)*np.ones(self.Graph.n_vertices,) - self.signals)
        W = self.Adjacency.copy()
        rows, cols = [k for k in W.nonzero()[0]], [k for k in W.nonzero()[1]]
        lookup = self.Graph.A.astype(np.double)
        indices = lookup.tocsr().nonzero()
        second = []
        for k in range(len(lookup.nonzero()[0])):
            second.append(cp.abs(self.variable[indices[0][k]] - self.variable[indices[1][k]]))
        for k in range(W.count_nonzero()):
            W[rows[k],cols[k]] = (resid[rows[0]] + resid[cols[k]])/2

        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        y = ((1+np.max(self.signals)) / 2) * np.ones(self.Graph.n_vertices,) - self.signals
        #constraint = [cp.norm(self.Adjacency @ self.variable, 'inf') <= 2, 2*cp.sum(self.variable) - cp.sum(self.Adjacency @ self.variable) - 2 == 0]
        #constraint = [cp.norm(self.Adjacency @ self.variable, 'inf') <= 2, self.variable <= 1]#, self.variable <= 1]
        object_path = cp.Minimize(self.signals@self.signals + 2*self.variable@resid)
        self.problem = cp.Problem(object_path)
        self.problem.solve(solver=cp.GUROBI, verbose=verbose)
        


    ''' gLap Algorithmus lambda 0.3 binär''' 
    def glap_binary(self, verbose=False):
        self.variable = cp.Variable(self.Graph.n_vertices, boolean=True)
        obje = cp.Minimize(cp.sum_squares(self.signals-self.variable) + self.lambd * cp.quad_form(self.variable,self.Graph.L))
        self.problem = cp.Problem(obje)
        self.problem.solve(solver=cp.GUROBI, verbose=verbose)
        # return x, problem

    ''' path Algorithmus einfach reell'''
    def path_real(self, verbose=False):    
        self.Adjacency = self.Graph.A.astype(np.double)
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        constr = [cp.norm(self.Adjacency@self.variable, 'inf') <= 2, self.variable <= 1]
        obje_path = cp.Minimize(cp.sum_squares(self.signals - self.variable))
        self.problem = cp.Problem(obje_path, constr)
        self.prog.solve(solver=cp.GUROBI, verbose=verbose)
        # return x1, prog

    ''' path Algorithmus C2 lambda 2x_max - 1 '''
    def path_lmax(self, verbose=False):
        self.Adjacency = self.Graph.A.astype(np.double)#nx.convert_matrix.to_scipy_sparse_matrix(nx.to_directed(self.Graph))
        self.variable = cp.Variable(self.Graph.n_vertices, boolean=True)
        constr2 = [cp.norm(self.Adjacency@self.variable, 'inf') <= 2, self.variable <= 1]
        obj_far = cp.Minimize(cp.sum_squares(self.signals - self.variable) + (2*np.max(self.signals) - 1)*cp.sum(self.variable))
        self.problem = cp.Problem(obj_far, constr2)
        self.problem.solve(solver=cp.GUROBI,verbose=verbose)
        # self.variable.value[self.variable.value <= self.lambd] = 0
        # self.variable.value[self.variable.value != 0] = 1
        # return x2, prog2

    def multi_signal_decomp(self, K=3):
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)

    def network_anomaly(self):
        pass

    def optim_stela(self, mu, threshold=None):
        self.val, self.st_x, self.error = st.stela_lasso(self.pathDic, self.signals, mu*cp.norm_inf(self.signals@self.pathDic).value, 500)

        #if threshold:
        #    self.stela_sol[self.stela_sol <= threshold] = 0
        #else:
        #    self.stela_sol[self.stela_sol <= np.var(self.signals)] = 0
        #self.stela_sol =


