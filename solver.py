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
            object_path_sparse = cp.Minimize(cp.sum(self.sparse_var))
            self.subProblem = cp.Problem(object_path_sparse, constraint2)
            self.subProblem.solve(solver=cp.GUROBI, verbose=verbose)
            #reconstruct = dict(zip(self.variable.value.nonzero()[0],self.subProblem.value))
            #print(reconstruct)
            print(len(self.sparse_var.value.nonzero()[0]))
            
    
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
        #return x, problem

    ''' path Algorithmus einfach reell'''
    def path_real(self, verbose=False):    
        self.Adjacency = self.Graph.A.astype(np.double)
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        constr = [cp.norm(self.Adjacency@self.variable, 'inf') <= 2, self.variable <= 1]
        obje_path = cp.Minimize(cp.sum_squares(self.signals - self.variable))
        self.problem = cp.Problem(obje_path, constr)
        self.prog.solve(solver=cp.GUROBI, verbose=verbose)
        #return x1, prog 

    ''' path Algorithmus C2 lambda 2x_max - 1 '''
    def path_lmax(self, verbose=False):
        self.Adjacency = self.Graph.A.astype(np.double)#nx.convert_matrix.to_scipy_sparse_matrix(nx.to_directed(self.Graph))
        self.variable = cp.Variable(self.Graph.n_vertices, boolean=True)
        constr2 = [cp.norm(self.Adjacency@self.variable, 'inf') <= 2, self.variable <= 1]
        obj_far = cp.Minimize(cp.sum_squares(self.signals - self.variable) + (2*np.max(self.signals) - 1)*cp.sum(self.variable))
        self.problem = cp.Problem(obj_far, constr2)
        self.problem.solve(solver=cp.GUROBI,verbose=verbose)
        #self.variable.value[self.variable.value <= self.lambd] = 0
        #self.variable.value[self.variable.value != 0] = 1
        #return x2, prog2

    
    def multi_signal_decomp(self):
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
