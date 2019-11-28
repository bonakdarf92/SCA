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

    def path_based(self, verbose=False):
        self.Adjacency = self.Graph.A.astype(np.double)
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        constraint = [cp.norm(self.Adjacency @ self.variable, 'inf') <= 2, self.variable <= 1]
        object_path = cp.Minimize(cp.sum_squares(self.signals - self.variable))
        self.problem = cp.Problem(object_path, constraint)
        self.problem.solve(solver=cp.GUROBI,verbose=verbose)
        self.variable.value[self.variable.value <= self.lambd] = 0
        self.variable.value[self.variable.value != 0] = 1
        


    ''' gLap Algorithmus lambda 0.3 binär''' 
    def glap_binary(self, verbose=False):
        self.variable = cp.Variable(self.Graph.n_vertices, boolean=True)
        obje = cp.Minimize(cp.sum_squares(self.signals-self.variable) + self.lambd * cp.quad_form(self.variable,self.Graph.L))
        self.problem = cp.Problem(obje)
        self.problem.solve(solver=cp.GUROBI, verbose=verbose)
        #return x, problem

    ''' path Algorithmus einfach reell'''
    def path_real(self, verbose=False):    
        self.Adjacency = nx.convert_matrix.to_scipy_sparse_matrix(nx.to_directed(self.Graph))
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        constr = [cp.norm(self.Adjacency@self.variable, 'inf') <= 2, self.variable <= 1]
        obje_path = cp.Minimize(cp.sum_squares(self.signals - self.variable))
        self.problem = cp.Problem(obje_path, constr)
        self.prog.solve(solver=cp.GUROBI, verbose=verbose)
        #return x1, prog 

    ''' path Algorithmus C2 lambda 2x_max - 1 '''
    def path_lmax(self, verbose=False):
        self.Adjacency = nx.convert_matrix.to_scipy_sparse_matrix(nx.to_directed(self.Graph))
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
        constr2 = [cp.norm(self.Adjacency@self.variable, 'inf') <= 2, self.variable <= 1]
        obj_far = cp.Minimize(cp.sum_squares(self.signals - self.variable) + (2*np.max(self.signals) - 1)*cp.sum(self.variable))
        self.problem = cp.Problem(obj_far, constr2)
        self.problem.solve(solver=cp.GUROBI,verbose=verbose)
        #return x2, prog2

    
    def multi_signal_decomp(self):
        self.variable = cp.Variable(self.Graph.n_vertices, nonneg=True)
