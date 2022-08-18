# -*- coding: utf-8 -*-
"""
MISPIAU/BSPIAU: Toy Example
S. Punla-Green
"""

import numpy as np
from pyomo.environ import *
import networkx as nx
from functions import create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths_multiple_interdictions
from functions import create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx


def return_interdicted_arcs(M,G,B):
    interdicted=[]
    for (i,j) in set(G.edges):
        for b in range(1,B-1):
            if M.x[(i,j),b].value >= 0.9:
                interdicted.append((i,j))
    return interdicted

#%%


G=nx.DiGraph()
G.add_edge(0,1,length=2,interdicted_length=2)
G.add_edge(0,2,length=8,interdicted_length=1)
G.add_edge(1,3,length=5,interdicted_length=0)
G.add_edge(2,3,length=5,interdicted_length=0)

B=2
vdim=B*len(set(G.edges()))
R={}

for (i,j) in set(G.edges):
    r=G[i][j]['interdicted_length']
    k=0
    for (u,v) in set(G.edges):
        if i==u and j==v:
            R[(i,j,k,1)]=r
            R[(i,j,k,2)]=r
        k=k+1    
        
S=0
T=3
opt=SolverFactory('gurobi_direct')
M_MISPIAU=create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R,S,T,vdim,B)
opt.solve(M_MISPIAU)
print('Interdict:')
for (i,j) in set(G.edges):
    if M_MISPIAU.x[(i,j),1].value>=0.9:
        print(f'({i},{j})')
    if M_MISPIAU.x[(i,j),2].value>=0.9:
        print(f'({i},{j})')
(paths,lengths)=return_paths_multiple_interdictions(M_MISPIAU,G,R,S,T)
print('Path')
print(paths)
print('Adjusted Length')
print(lengths)
print('Evader Length')
print(M_MISPIAU.d[T].value)


#%% Bolstered Version

G=nx.DiGraph()
G.add_edge(0,1,length=2,interdicted_length=2)
G.add_edge(0,2,length=8,interdicted_length=1)
G.add_edge(1,3,length=5,interdicted_length=0)
G.add_edge(2,3,length=5,interdicted_length=0)

B=2
vdim=len(set(G.edges()))
R={}
R1={}

for (i,j) in set(G.edges):
    r=G[i][j]['interdicted_length']
    k=0
    for (u,v) in set(G.edges):
        if i==u and j==v:
            R[(i,j,k)]=r
            R1[(i,j,k)]=0.5*r
        k=k+1    
        
S=0
T=3
opt=SolverFactory('gurobi_direct')
M_BSPIAU=create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R,R1,S,T,vdim,B)
opt.solve(M_BSPIAU)

print('Interdict:')
for (i,j) in set(G.edges):
    if M_BSPIAU.x[(i,j)].value>=0.9:
        print(f'({i},{j})')
print('Bolster')
for (i,j) in set(G.edges):
    if M_BSPIAU.b[(i,j)].value>=0.9:
        print(f'({i},{j})')

