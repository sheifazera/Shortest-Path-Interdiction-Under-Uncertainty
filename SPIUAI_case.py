"""
SPI UAI: Case Study Sioux Falls
S. Punla-Green
"""

#Shortest path with Uncertain Asymmetric Information
import networkx as nx
import time
import matplotlib.pyplot as plt
from functions import create_uncertain_asymmetric_information_shortest_path_interdiction_nx, SPI_UAI_algorithm_mod
import random
import numpy as np
from pyomo.environ import *
import itertools
from pyomo.gdp import *
from pyomo.mpec import *

D={(1,2):(2,3),(2,4):(3,4),(1,3):(3,1),(3,4):(3,2)} #Make sure r is the amount added, not the final amount
D_perceived={(1,2):4,(2,4):5,(1,3):3,(3,4):3}

G=nx.DiGraph()

for (i,j) in D.keys():
    (l,r)=D[(i,j)]
    G.add_edge(i,j)
    G[i][j]['length']=l
    G[i][j]['interdicted_length']=r
    G[i][j]['perceived_length']=D_perceived[(i,j)]


r_perceived={} #THIS IS THE AMOUNT ADDED, NOT THE FINAL AMOUNT

r12=[3,4,5]
r24=[4,5,6]
r13=[1,2]
r34=[2,3,4]
'''
comb=list(itertools.product(*[r12,r24,r13,r34]))
Idim=len(comb)
for i in range(0,Idim):
    [r12,r24,r13,r34]=comb[i]
    r_perceived[(i,1,2)]=r12
    r_perceived[(i,2,4)]=r24
    r_perceived[(i,1,3)]=r13
    r_perceived[(i,3,4)]=r34
 
'''    
B=1
s=1
t=4 
'''
M=create_uncertain_asymmetric_information_shortest_path_interdiction_nx(G,r_perceived,s,t,Idim,B)
opt=SolverFactory('gurobi')
#M.constraint=Constraint(expr=M.x[(1,3)] ==1)
opt.solve(M)
M.x.pprint()
M.m.pprint()
'''


#%%
random.seed(a=631996)
R={}
R[(1,2)]=tuple(r12)
R[(2,4)]=tuple(r24)
R[(1,3)]=tuple(r13)
R[(3,4)]=tuple(r34)
for (i,j) in R.keys():
    R[(i,j)]=tuple([min(R[(i,j)]),max(R[(i,j)])])

(M,path, length, m, its, _ , _ , _)=SPI_UAI_algorithm_mod(G,R,s,t,B)

print('Interdicted Arcs')
for (i,j) in set(G.edges):
    if M.x[(i,j)].value >= 0.9:
        print(f'({i},{j})')
print(path)
print('Evader Length')
print(length)
print('Interdictor Length')
print(m)






    



