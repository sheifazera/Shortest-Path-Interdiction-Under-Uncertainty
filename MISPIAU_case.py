"""
MISPIAU/BSPIAU: Case Study
S. Punla-Green


Note: This script is designed to find a network of interest for the case study
The first network found is the one used for the study. Exit the script after data has been saved.
"""

import numpy as np
import datetime
import itertools as it
import matplotlib.pyplot as plt
import random
import time
from pyomo.environ import *
#from prettytable import PrettyTable, ALL
from functions import create_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths, return_paths_bolstered
from functions import create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths_multiple_interdictions
from functions import create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx, return_interdicted_arcs
import networkx as nx
import pickle
import statistics

opt=SolverFactory('gurobi_direct')
opt.options['TimeLimit']=1800

Budgets={5,10}
Nodes={15}

def return_bolstered_arcs(M,G):
    bolstered=[]
    for (i,j) in set(G.edges):
            if M.b[(i,j)].value >= 0.9:
                bolstered.append((i,j))
    return bolstered

def return_interdicted_arcs_multiple_interdictions(M,G,B):
    interdicted=[]
    for (i,j) in set(G.edges):
        for b in range(1,B+1):
            if M.x[(i,j),b].value >= 0.9:
                interdicted.append((i,j))
    return interdicted

#%%
random.seed(a=631996) #Loops 1-5
#random.seed(a=5111994) #Loops 6-10
#random.seed(a=9182019) # Loops 11-15
#random.seed(a=5012020) #loop 16-20
data={}
data_S={}
data_B={}
data_MI={}
data_bolstered={(15,10):0,(15,5):0,(50,15):0,(50,20):0,(100,5):0,(100,10):0,(100,15):0,(100,20):0,(150,5):0,(150,10):0,(150,15):0,(150,20):0,(200,5):0,(200,10):0,(200,15):0,(200,20):0}
data_multiple_interdictions={(50,5):0,(50,10):0,(50,15):0,(50,20):0,(100,5):0,(100,10):0,(100,15):0,(100,20):0,(150,5):0,(150,10):0,(150,15):0,(150,20):0,(200,5):0,(200,10):0,(200,15):0,(200,20):0}
for loop in range(1,100):
    ct=datetime.datetime.now()
    print(f'Starting with loop {loop}')
    print("current time:", ct)
    for N in Nodes:
        print(f'Starting with {N} nodes')
        connected=False
        while connected is False:
            G=nx.random_geometric_graph(N, 1.5/sqrt(N), dim=2, p=2)
            connected=nx.is_connected(G)
        G=G.to_directed()
        
        for (i,j) in G.edges:
           p=random.random()
           q=random.uniform(0.25*p, 0.75*p)
           G[i][j]['length'] = -np.log(p)
           G[i][j]['interdicted_length']=-np.log(q)+np.log(p)
        
        
        
        s=random.choice(list(G.nodes))
        t=random.choice(list(set(G.nodes)-{s}))
        vdim=len(set(G.edges()))
        R_B={}
        R_B1={}
        R_S={}
        
        for (i,j) in set(G.edges):
            r=G[i][j]['interdicted_length']
            k=0
            for (u,v) in set(G.edges):
                if i==u and j==v:
                    R_B[(i,j,k)]=r
                    R_B1[(i,j,k)]=random.uniform(0.4*r,0.6*r)
                    R_S[(i,j,k)]=r
                k=k+1
        
        for B in Budgets:
            M_B=create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R_B,R_B1,s,t,vdim,B)
            start=time.time()
            results=opt.solve(M_B)
            end=time.time()
            if results.solver.termination_condition==TerminationCondition.optimal:
                interdicted_B=return_interdicted_arcs(M_B,G)
                bolstered=return_bolstered_arcs(M_B,G)
                (paths_B,lengths_B)=return_paths_bolstered(M_B,G,R_B,R_B1,s,t)
            else:
                lengths_B=0
                bolstered=[]
            data_B[(N,B,loop)]=(results.solver.wallclock_time)
            if len(bolstered)>0:
                data_bolstered[(N,B)]=data_bolstered[(N,B)]+1
                for (i,j) in bolstered:
                    for path in paths_B:
                        if i in path and j in path:
                            nx.write_edgelist(G, "test.edgelist", data=["length","interdicted_length"])
                            nx.draw(G,with_labels=True)
                            np.save('RB1_bolstering.npy', R_B1)
                            np.save('RB_bolstering.npy', R_B)
                            print(s)
                            print(t)
                            print('Intericted')
                            print(interdicted_B)
                            print('Bolstered)')
                            print(bolstered)
                            print(paths_B)
                            print(f'PoE={np.exp(-lengths_B)}')
                            print('Evader PoE')
                            print(np.exp(-M_B.d[t].value))
                            
                            #ADD SOMETHING HERE TO SOLVE mispiau AND SpiAU AS WELL
                            print(len(set(G.edges)))
                            k=0
                            for (i,j) in set(G.edges):
                                print(f'({i},{j}) & {np.exp(-G[i][j]["length"])} & {np.exp(-G[i][j]["interdicted_length"])} & {np.exp(-R_B1[i,j,k])} \\\ ')
                                k=k+1
                            
                            M_SPIAU=create_asymmetric_uncertainty_shortest_path_interdiction_nx(G, R_B,s,t, vdim, B)
                            opt.solve(M_SPIAU)
                            print('SPIAU')
                            interdicted=return_interdicted_arcs(M_SPIAU,G)
                            print('Interdicted')
                            print(interdicted)
                            (paths,lengths)=return_paths(M_SPIAU,G,R_B,s,t)
                            print('Path')
                            print(paths)
                            print('Adjusted Prob')
                            print(np.exp(-lengths))
                            print('Evader Prob')
                            print(np.exp(-M_SPIAU.d[t].value))
                            vdim_MI=B*len(set(G.edges()))
            
                            R_MI={}
                            for (i,j) in set(G.edges):
                                r=G[i][j]['interdicted_length']
                                k=0
                                for (u,v) in set(G.edges):
                                    if i==u and j==v:
                                        for l in range(1,B+1):
                                            R_MI[(i,j,k,l)]=r 
                                    k=k+1
                            M_MISPIAU=create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R_MI,s,t,vdim_MI,B)
                            opt.solve(M_MISPIAU)
                            print('Multiple Interdiction SPIAU')
                            interdicted=return_interdicted_arcs_multiple_interdictions(M_MISPIAU,G,B)
                            print('Interdicted')
                            print(interdicted)
                            (paths,lengths)=return_paths_multiple_interdictions(M_MISPIAU,G,R_MI,s,t)
                            print('Path')
                            print(paths)
                            print('Adjusted Prob')
                            print(np.exp(-lengths))
                            print('Evader Prob')
                            print(np.exp(-M_MISPIAU.d[t].value))
                            
                            print('SPI')
                            R={}
                            vdim=0
                            M_SPI=create_asymmetric_uncertainty_shortest_path_interdiction_nx(G, R,s,t, vdim, B)
                            opt.solve(M_SPI)
                            interdicted=return_interdicted_arcs(M_SPI,G)
                            print('Interdicted')
                            print(interdicted)
                            (paths,lengths)=return_paths(M_SPI,G,R_B,s,t)
                            print('Path')
                            print(paths)
                            print('Adjusted Prob')
                            print(np.exp(-lengths))
                            print('Evader Prob')
                            print(np.exp(-M_SPI.d[t].value))
                            
                        
                            
                            input("Press Enter to continue...")
ct=datetime.datetime.now()
print("current time:", ct)


