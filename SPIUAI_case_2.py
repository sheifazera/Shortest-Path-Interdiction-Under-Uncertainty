# -*- coding: utf-8 -*-
"""
SPI UAI: Case Study
"""

#SPI - UAI

import networkx as nx
import time
import matplotlib.pyplot as plt
from functions import return_path_asymmetric_information_SPI, create_SPI_asymmetric_information, return_path_nominal_SPI, create_shortest_path_interdiction_nx, create_uncertain_asymmetric_information_shortest_path_interdiction_nx, SPI_UAI_algorithm, SPI_UAI_regret_avoided, SPI_UAI_algorithm_mod
import random
import numpy as np
from pyomo.environ import *
import itertools
import time
import pickle

matrix = np.loadtxt('siouxfalls.txt', usecols=(0,1,3))

D = {(int(a),int(b)) : c for a,b,c in matrix}
N=set(range(1,25))
s=3
t=20

random.seed(a=631996)

opt=SolverFactory('gurobi')

def perturb(a):
    a_p=random.uniform(0.9*a,1.1*a)
    a_p=max(min(0.95,a_p),0.05)
    return a_p

def perturb_q(a):
    q_p=max(min(0.95,random.uniform(0.25*a,0.75*a)),0.05)
    return q_p
    

Prob = {(int(a),int(b)) : c for a,b,c in matrix}
for (i,j) in Prob.keys():
    if Prob[(i,j)] ==10:
        Prob[(i,j)] = (perturb(0.1),perturb_q(0.1)) #p_ij, q_ij 
    elif Prob[(i,j)] ==9 :
        Prob[(i,j)] = (perturb(0.2),perturb_q(0.2))
    elif Prob[(i,j)] ==8 :  
        Prob[(i,j)] = (perturb(0.3),perturb_q(0.3))
    elif Prob[(i,j)] ==7 :  
        Prob[(i,j)] = (perturb(0.4),perturb_q(0.4))
    elif Prob[(i,j)] ==6 :  
        Prob[(i,j)] = (perturb(0.5),perturb_q(0.5))
    elif Prob[(i,j)] ==5 :  
        Prob[(i,j)] = (perturb(0.6),perturb_q(0.6))
    elif Prob[(i,j)] ==4 :  
        Prob[(i,j)] = (perturb(0.7),perturb_q(0.7))
    elif Prob[(i,j)] ==3 :  
        Prob[(i,j)] = (perturb(0.8),perturb_q(0.8))
    elif Prob[(i,j)] ==2 :  
        Prob[(i,j)] = (perturb(0.9),perturb_q(0.9))    
    else:
        raise Exception('Original Sioux Falls value out of range 2-10')
G=nx.DiGraph()
possible_r=3
R={}
Q=4  
for (i,j) in Prob.keys():
    r_temp=[]
    (p,q)=Prob[(i,j)]
    p_perceived=max(min(0.95,p+random.uniform(-Q*0.1,Q*0.1)),0.05)
    G.add_edge(i,j,length=-np.log(p),interdicted_length=-np.log(q)+np.log(p), perceived_length=-np.log(p_perceived))
    q_perceived=max(min(0.95,0.5*p_perceived),0.05)
    r_temp.append(-np.log(q_perceived)+np.log(p_perceived))
    for l in range(1,possible_r):
        q_temp=max(min(random.uniform(0.1*p_perceived,0.9*p_perceived),0.95),0.05)
        r_temp.append(-np.log(q_temp)+np.log(p_perceived))
    R[(i,j)]=tuple([min(r_temp),max(r_temp)])
#%%

import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)
B=5
t0=time.time()
(M,path, length, m, its, Master_time, Sub_time, Sub2_time)=SPI_UAI_algorithm_mod(G,R,s,t,B)
t1=time.time()

print(f'Total Algorithm Time = {t1-t0} s')
print(f'Master Solve Time = {Master_time} s')
print(f'Subproblem Solve Time = {Sub_time} s')
print(f'Subproblem 2 Time = {Sub2_time} s')


print('Interdicted Arcs')
for (i,j) in set(G.edges):
    if M.x[(i,j)].value >= 0.9:
        print(f'({i},{j})')
print(path)
print('Evader PoE')
print(exp(-length))
print('Interdictor PoE')
print(exp(-m))
        



#%% Sioux Falls Asymmetric Information
for (i,j) in G.edges:
    p=exp(-G[i][j]['perceived_length'])
    q=0.5*p
    G[i][j]['perceived_interdicted_length']=-np.log(q)+np.log(p)
    
M_SPI=create_SPI_asymmetric_information(M,G,s,t, B)
opt.solve(M_SPI)
for (i,j) in set(G.edges):
    if M_SPI.x[(i,j)].value >= 0.9:
        print(f'({i},{j})')
        
(evad_paths,evad_lengths)=return_path_asymmetric_information_SPI(M_SPI,G, s,t)
print('SPI AI')
print(evad_paths)
print('Evader PoE')
print(exp(-evad_lengths))

print('Interdictor PoE')
print(exp(-value(M_SPI.Obj)))

(z1,z2, path, length)=SPI_UAI_regret_avoided(M_SPI,M, G,R,s,t)
RA=(exp(-z1)-exp(-z2))/exp(-z1) 
print(f'Regret Avoided={value(RA)}')
print('Worst-case Path')
print(path)
print(f'Interdictor PoE= {exp(-value(z1))}')
print(f'Evader PoE = {exp(-length)}')


#%% Sioux Falls Nominal
G=nx.DiGraph()
for (i,j) in Prob.keys():
    (p,q)=Prob[(i,j)]
    G.add_edge(i,j,length=-np.log(p),interdicted_length=-np.log(q)+np.log(p))

s=3
t=20
B=5
M=create_shortest_path_interdiction_nx(G,s,t, B)
opt.solve(M)
print('SPI')
print('Interdicted Arcs')
for (i,j) in set(G.edges):
    if M.x[(i,j)].value >= 0.9:
        print(f'({i},{j})')
        
(nom_paths,nom_lengths)=return_path_nominal_SPI(M,G,s,t)
print(nom_paths)
print('PoE')
print(exp(-nom_lengths))

    
    
    


 