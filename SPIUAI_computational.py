
"""
SPI UAI: Computational Study
S. Punla-Green
"""

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import random
import time
from pyomo.environ import *
import networkx as nx
import pickle
from functions import return_path_asymmetric_information_SPI, create_SPI_asymmetric_information, return_path_nominal_SPI, create_shortest_path_interdiction_nx, create_uncertain_asymmetric_information_shortest_path_interdiction_nx, SPI_UAI_algorithm_mod

opt=SolverFactory('gurobi_direct')
#opt.options['TimeLimit']=1200

Budgets={3,5,7}
Nodes={75}
Q=4
possible_r=5
#%%
#random.seed(a=631996) #25 and 50 Nodes
random.seed(a=5111994) #75 nodes
#random.seed(a=12152015) #100 node 
#random.seed(a=7161951) #6 and 7 Monday into Tuesday
#random.seed(a=7171956) #8 and 9 and 10 Wednesday

data={}
data_dead={}
for N in Nodes:
    for B in Budgets:
        data_dead[(N,B)]=0

#%%
for N in Nodes:
    for loop in range(1,6):
        R={}
        connected=False
        while connected is False:
            G=nx.random_geometric_graph(N, 1.5/sqrt(N), dim=2, p=2)
            connected=nx.is_connected(G)
        G=G.to_directed()
        
        for (i,j) in G.edges:
           r_temp=[]
           p=random.uniform(0.05,0.95)
           q=max(min(0.95,random.uniform(0.25*p, 0.75*p)),0.05)
           p_perceived=max(min(0.95,p+random.uniform(-Q*0.1,Q*0.1)),0.05)
           G[i][j]['length'] = -np.log(p)
           G[i][j]['interdicted_length']=-np.log(q)+np.log(p)
           G[i][j]['perceived_length']=-np.log(p_perceived)
           q_perceived=0.5*p_perceived
           r_temp.append(-np.log(q_perceived)+np.log(p_perceived))
           for l in range(1,possible_r):
               q_temp=random.uniform(0.1*p_perceived,0.9*p_perceived)
               r_temp.append(-np.log(q_temp)+np.log(p_perceived))   
           R[(i,j)]=tuple([min(r_temp),max(r_temp)])
           
    
        
        s=random.choice(list(G.nodes))
        t=random.choice(list(set(G.nodes)-{s}))
        
        for B in Budgets:
            start=time.time()
            (M,path, length, m,its, Master_time, Sub_time, _)=SPI_UAI_algorithm_mod(G,R,s,t,B, max_it=15, time_limit=1200, tol=1e-6)
            end=time.time()
            if path==99999:
                #This loop timed out and should not have its data recorded
                data_dead[(N,B)]+=1
            else:
                data[(N,B,loop)]=(end-start, its, Master_time, Sub_time)
            print(f'Finished with B={B}')
        print(f'Finished with loop {loop}')
    print(f'Finished with {N} nodes')
    #input("Press Enter to continue...")
    

np.save('SPIUAI_data_1_6_Q4_always_warmstart_N75_withpictures.npy', data)

#%% Plots

data2=np.load("SPIUAI_data_1_10_Q4_always_warmstart.npy", allow_pickle=True)
data=data2.item()

Budgets={3,5,7,9}
Nodes={25,50}
for N in Nodes:
    fig1, ax1 = plt.subplots()
    plt.title('%i Nodes' % (N))
    ax1.set_xlabel('Budget')
    #ax1.set_ylabel('Time to Solve')
    #ax1.set_ylabel('Iterations')
    ax1.set_ylabel('Percent Time Solving Master')
    B3=0
    B5=0
    B7=0
    B9=0
    B3list=[]
    B5list=[]
    B7list=[]
    B9list=[]
    for loop in range(1,11):
        for B in Budgets:
            if B==3:
                if (N,B,loop) not in data.keys():
                    B3=B3+1
                else:
                    (time, its, Master, Sub) =data[(N,B,loop)]
                    if its >= 15:
                        B3=B3+1
                    else:
                        B3list.append(time)    
            elif B==5:
                if (N,B,loop) not in data.keys():
                    B5=B5+1
                else:
                    (time, its, Master, Sub) =data[(N,B,loop)]
                    if its >= 15:
                        B5=B5+1
                    else:
                        B5list.append(time)   
            elif B==7:
                if (N,B,loop) not in data.keys():
                    B7=B7+1
                else:
                    (time, its, Master, Sub) =data[(N,B,loop)]
                    if its >= 15:
                        B7=B7+1
                    else:
                        B7list.append(time)    
            
            elif B==9:
                if (N,B,loop) not in data.keys():
                    B9=B9+1
                else:
                    (time, its, Master, Sub) =data[(N,B,loop)]
                    if its >= 15:
                        B9=B9+1
                    else:
                        B9list.append(time)  
                       
    ax1.boxplot([B3list, B5list], labels=[3,5], showfliers=True)
    m=max(B3list+B5list+B7list+B9list)
    '''
    plt.text(1,.85*m,f'({B3})',fontsize='x-large')
    plt.text(2,.85*m,f'({B5})',fontsize='x-large')
    plt.text(3,.85*m,f'({B7})',fontsize='x-large')
    plt.text(4,.85*m,f'({B9})',fontsize='x-large')
    '''
    print(f'Nodes={N}')
    print(f'B3: mean={statistics.mean(B3list)}, SD={statistics.stdev(B3list)}')
    print(f'B5: mean={statistics.mean(B5list)}, SD={statistics.stdev(B5list)}')
    print(f'B7: mean={statistics.mean(B7list)}, SD={statistics.stdev(B7list)}')
    print(f'B9: mean={statistics.mean(B9list)}, SD={statistics.stdev(B9list)}')
    
#%% Data Analysis

#Check the ratio for sub to Master   

data2=np.load("SPIUAI_data_1_10_Q4_always_warmstart.npy", allow_pickle=True)
data25_59=data2.item() 

data2=np.load("SPIUAI_data_1_10_Q4_always_warmstart_N75.npy", allow_pickle=True)
data75=data2.item() 

data2=np.load("SPIUAI_data_1_10_Q4_always_warmstart_N100.npy", allow_pickle=True)
data100=data2.item() 
    

