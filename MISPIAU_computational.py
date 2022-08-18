"""
MISPIAU/BSPIAU: Computational Study
S. Punla-Green
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

Budgets={5,10,15,20}
Nodes={50,100,150,200}


def return_interdicted_arcs_multiple_interdictions(M,G,B):
    interdicted=[]
    for (i,j) in set(G.edges):
        for b in range(1,B+1):
            if M.x[(i,j),b].value >= 0.9:
                interdicted.append((i,j))
    return interdicted

def return_bolstered_arcs(M,G):
    bolstered=[]
    for (i,j) in set(G.edges):
            if M.b[(i,j)].value >= 0.9:
                bolstered.append((i,j))
    return bolstered

#%%
#random.seed(a=631996) #Loops 1-5
#random.seed(a=5111994) #Loops 6-10
#random.seed(a=9182019) # Loops 11-15
random.seed(a=5012020) #loop 16-20
data={}
data_S={}
data_B={}
data_MI={}
data_bolstered={(50,5):0,(50,10):0,(50,15):0,(50,20):0,(100,5):0,(100,10):0,(100,15):0,(100,20):0,(150,5):0,(150,10):0,(150,15):0,(150,20):0,(200,5):0,(200,10):0,(200,15):0,(200,20):0}
data_multiple_interdictions={(50,5):0,(50,10):0,(50,15):0,(50,20):0,(100,5):0,(100,10):0,(100,15):0,(100,20):0,(150,5):0,(150,10):0,(150,15):0,(150,20):0,(200,5):0,(200,10):0,(200,15):0,(200,20):0}
for loop in range(16,21):
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
                    
            M_MI=create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R_MI,s,t,vdim_MI,B)
            start=time.time()
            results=opt.solve(M_MI)
            end=time.time()
            if results.solver.termination_condition==TerminationCondition.optimal:
                interdicted=return_interdicted_arcs_multiple_interdictions(M_MI,G,B)
                (paths_MI,lengths_MI)=return_paths_multiple_interdictions(M_MI,G,R_MI,s,t)
            else:
                lengths_MI=0
                interdicted=[]
            data_MI[(N,B,loop)]=(results.solver.wallclock_time)
            if len(interdicted)>len(set(interdicted)):
                data_multiple_interdictions[(N,B)]=data_multiple_interdictions[(N,B)]+1
            
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
                    
            
            
            M_S=create_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R_S,s,t,vdim,B)
            start=time.time()
            results=opt.solve(M_S)
            end=time.time()
            if results.solver.termination_condition==TerminationCondition.optimal:
                #interdicted_S=return_interdicted_arcs(M_S,G)
                (paths_S,lengths_S)=return_paths(M_S,G,R_S,s,t)
            else:
                length_S=0
            data_S[(N,B,loop)]=(results.solver.wallclock_time)
            
            data[(N,B,loop)]=(-np.log(lengths_S),-np.log(lengths_B),-np.log(lengths_MI))
            
            print(f'Finished with {B} Budget')
ct=datetime.datetime.now()
print("current time:", ct)
np.save('data_multiple_interdictions_all_SPIAU_16_20.npy', data_S)  
np.save('data_multiple_interdictions_all_BSPIAU_16_20.npy', data_B) 
np.save('data_multiple_interdictions_all_MISPIAU_16_20.npy', data_MI) 
np.save('data_multiple_interdictions_all_SPIAU_poe_16_20.npy', data) 
np.save('data_multiple_interdictions_all_SPIAU_bolsters_16_20.npy', data_bolstered) 
np.save('data_multiple_interdictions_all_SPIAU_mult_ints_16_20.npy', data_multiple_interdictions) 
#%% Data Analytics
import numpy as np
data1=np.load("data_multiple_interdictions_all_SPIAU_1_5.npy", allow_pickle=True)
dataS1=data1.item()
data2=np.load("data_multiple_interdictions_all_SPIAU_6_10.npy", allow_pickle=True)
dataS2=data2.item()
data3=np.load("data_multiple_interdictions_all_SPIAU_11_15.npy", allow_pickle=True)
dataS3=data3.item()
data4=np.load("data_multiple_interdictions_all_SPIAU_16_20.npy", allow_pickle=True)
dataS4=data4.item()

dataS={**dataS1,**dataS2,**dataS3,**dataS4}
np.save('data_multiple_interdictions_all_SPIAU_1_20.npy',dataS)

data1=np.load("data_multiple_interdictions_all_BSPIAU_1_5.npy", allow_pickle=True)
dataS1=data1.item()
data2=np.load("data_multiple_interdictions_all_BSPIAU_6_10.npy", allow_pickle=True)
dataS2=data2.item()
data3=np.load("data_multiple_interdictions_all_BSPIAU_11_15.npy", allow_pickle=True)
dataS3=data3.item()
data4=np.load("data_multiple_interdictions_all_BSPIAU_16_20.npy", allow_pickle=True)
dataS4=data4.item()

dataS={**dataS1,**dataS2,**dataS3,**dataS4}
np.save('data_multiple_interdictions_all_BSPIAU_1_20.npy',dataS)


data1=np.load("data_multiple_interdictions_all_MISPIAU_1_5.npy", allow_pickle=True)
dataS1=data1.item()
data2=np.load("data_multiple_interdictions_all_MISPIAU_6_10.npy", allow_pickle=True)
dataS2=data2.item()
data3=np.load("data_multiple_interdictions_all_MISPIAU_11_15.npy", allow_pickle=True)
dataS3=data3.item()
data4=np.load("data_multiple_interdictions_all_MISPIAU_16_20.npy", allow_pickle=True)
dataS4=data4.item()

dataS={**dataS1,**dataS2,**dataS3,**dataS4}
np.save('data_multiple_interdictions_all_MISPIAU_1_20.npy',dataS)



data1=np.load("data_multiple_interdictions_all_SPIAU_poe_1_5.npy", allow_pickle=True)
dataS1=data1.item()
data2=np.load("data_multiple_interdictions_all_SPIAU_poe_6_10.npy", allow_pickle=True)
dataS2=data2.item()
data3=np.load("data_multiple_interdictions_all_SPIAU_poe_11_15.npy", allow_pickle=True)
dataS3=data3.item()
data4=np.load("data_multiple_interdictions_all_SPIAU_poe_16_20.npy", allow_pickle=True)
dataS4=data4.item()

dataS={**dataS1,**dataS2,**dataS3,**dataS4}
datanew={}
for (N,B,loop) in dataS.keys():
    [S,Bo,MI]=dataS[(N,B,loop)]
    S=exp(-(exp(-S)))
    MI=exp(-(exp(-MI)))
    Bo=exp(-(exp(-Bo)))
    datanew[(N,B,loop)]=(S,Bo,MI)

np.save('data_multiple_interdictions_all_SPIAU_poe_1_20.npy',datanew)

data1=np.load("data_multiple_interdictions_all_SPIAU_bolsters_1_5.npy", allow_pickle=True)
dataS1=data1.item()
data2=np.load("data_multiple_interdictions_all_SPIAU_bolsters_6_10.npy", allow_pickle=True)
dataS2=data2.item()
data3=np.load("data_multiple_interdictions_all_SPIAU_bolsters_11_15.npy", allow_pickle=True)
dataS3=data3.item()
data4=np.load("data_multiple_interdictions_all_SPIAU_bolsters_16_20.npy", allow_pickle=True)
dataS4=data4.item()

datanew={}
for (N,B) in dataS1.keys():
    datanew[(N,B)]=dataS1[(N,B)]+dataS2[(N,B)]+dataS3[(N,B)]+dataS4[(N,B)]
np.save('data_multiple_interdictions_all_SPIAU_bolsters_1_20.npy',datanew)

data1=np.load("data_multiple_interdictions_all_SPIAU_mult_ints_1_5.npy", allow_pickle=True)
dataS1=data1.item()
data2=np.load("data_multiple_interdictions_all_SPIAU_mult_ints_6_10.npy", allow_pickle=True)
dataS2=data2.item()
data3=np.load("data_multiple_interdictions_all_SPIAU_mult_ints_11_15.npy", allow_pickle=True)
dataS3=data3.item()
data4=np.load("data_multiple_interdictions_all_SPIAU_mult_ints_16_20.npy", allow_pickle=True)
dataS4=data4.item()

datanew={}
for (N,B) in dataS1.keys():
    datanew[(N,B)]=dataS1[(N,B)]+dataS2[(N,B)]+dataS3[(N,B)]+dataS4[(N,B)]
np.save('data_multiple_interdictions_all_SPIAU_mult_ints_1_20.npy',datanew)
#%% 
import numpy as np
infinity = float('inf')
data=np.load("data_multiple_interdictions_all_SPIAU_1_20.npy",allow_pickle=True)
dataS=data.item()
data=np.load("data_multiple_interdictions_all_BSPIAU_1_20.npy",allow_pickle=True)
dataB=data.item()
data=np.load("data_multiple_interdictions_all_MISPIAU_1_20.npy",allow_pickle=True)
dataMI=data.item()
data=np.load("data_multiple_interdictions_all_SPIAU_poe_1_20.npy",allow_pickle=True)
data_poe=data.item()
data=np.load("data_multiple_interdictions_all_SPIAU_bolsters_1_20.npy",allow_pickle=True)
data_bolsters=data.item()
data=np.load("data_multiple_interdictions_all_SPIAU_mult_ints_1_20.npy",allow_pickle=True)
data_mult=data.item()






#%%
Nodes={50,100,150,200}
#Nodes={150,200}
for N in Nodes:
    TO=np.zeros((4,3))
    B5list_S=[]
    B10list_S=[]
    B15list_S=[]
    B20list_S=[]
    B5list_B=[]
    B10list_B=[]
    B15list_B=[]
    B20list_B=[]
    B5list_MI=[]
    B10list_MI=[]
    B15list_MI=[]
    B20list_MI=[]

    for B in Budgets:
        PI_B=0
        PI_MI=0
        avg=0
        for loop in range(1,21):
            (S,Bo,MI)=data_poe[(N,B,loop)]
            if S==1:
                TO[int(B/5-1),0]+=1
            else:
                if B==5:
                    B5list_S.append(dataS[(N,B,loop)])    
                elif B==10:
                    B10list_S.append(dataS[(N,B,loop)])  
                elif B==15:
                    B15list_S.append(dataS[(N,B,loop)])  
                elif B==20:
                    B20list_S.append(dataS[(N,B,loop)])
                
            if Bo==1:
                TO[int(B/5-1),1]+=1
            else:
                if B==5:
                    B5list_B.append(dataB[(N,B,loop)])    
                elif B==10:
                    B10list_B.append(dataB[(N,B,loop)])  
                elif B==15:
                    B15list_B.append(dataB[(N,B,loop)])  
                elif B==20:
                    B20list_B.append(dataB[(N,B,loop)])
            if MI==1:
                TO[int(B/5-1),2]+=1
            else:   
                if B==5:
                    B5list_MI.append(dataMI[(N,B,loop)])    
                elif B==10:
                    B10list_MI.append(dataMI[(N,B,loop)])  
                elif B==15:
                    B15list_MI.append(dataMI[(N,B,loop)])  
                elif B==20:
                    B20list_MI.append(dataMI[(N,B,loop)])
                
            if S==1 or Bo==1 or MI==1:  #don't calculate PI if 
                PI_B+=0
                PI_MI+=0
            else:
                (poe_S,poe_B,poe_MI)=data_poe[(N,B,loop)]
                PI_B=PI_B+abs(poe_B-poe_S)/poe_S
                PI_MI=PI_MI+abs(poe_MI-poe_S)/poe_S
                avg=avg+1
                
        print(f'N={N}, B={B}, PI_B={PI_B/avg*100}, PI_MI={PI_MI/avg*100}')
        
    fig1, ax1 = plt.subplots()
    plt.title(' MISPIAU: %i Nodes' % (N))
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Solve Time (s)')
    ax1.boxplot([B5list_MI, B10list_MI,B15list_MI, B20list_MI], labels=[5,10,15,20], showfliers=True)
    m=max(B5list_MI+B10list_MI+B15list_MI+B20list_MI)
    plt.text(1,.85*m,f'({int(TO[0,2])})',fontsize='x-large')
    plt.text(2,.85*m,f'({int(TO[1,2])})',fontsize='x-large')
    plt.text(3,.85*m,f'({int(TO[2,2])})',fontsize='x-large')
    plt.text(4,.85*m,f'({int(TO[3,2])})',fontsize='x-large')
    
    fig1, ax1 = plt.subplots()
    plt.title(' BSPIAU: %i Nodes' % (N))
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Solve Time (s)')
    ax1.boxplot([B5list_B, B10list_B,B15list_B, B20list_B], labels=[5,10,15,20], showfliers=True)
    m=max(B5list_B+B10list_B+B15list_B+B20list_B)
    plt.text(1,.85*m,f'({int(TO[0,1])})',fontsize='x-large')
    plt.text(2,.85*m,f'({int(TO[1,1])})',fontsize='x-large')
    plt.text(3,.85*m,f'({int(TO[2,1])})',fontsize='x-large')
    plt.text(4,.85*m,f'({int(TO[3,1])})',fontsize='x-large')
    
    fig1, ax1 = plt.subplots()
    plt.title(' SPIAU: %i Nodes' % (N))
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Solve Time (s)')
    ax1.boxplot([B5list_S, B10list_S,B15list_S, B20list_S], labels=[5,10,15,20], showfliers=True)
    m=max(B5list_S+B10list_S+B15list_S+B20list_S)
    plt.text(1,.85*m,f'({int(TO[0,0])})',fontsize='x-large')
    plt.text(2,.85*m,f'({int(TO[1,0])})',fontsize='x-large')
    plt.text(3,.85*m,f'({int(TO[2,0])})',fontsize='x-large')
    plt.text(4,.85*m,f'({int(TO[3,0])})',fontsize='x-large')
    
    print(f"N={N}")
    print('SPIAU')
    print(f'B5: mean={statistics.mean(B5list_S)}, SD={statistics.stdev(B5list_S)}, TO={TO[0,0]}')
    print('BISPIAU')
    print(f'B5: mean={statistics.mean(B5list_B)}, SD={statistics.stdev(B5list_B)}, TO={TO[0,1]}')
    print('MISPIAU')
    print(f'B5: mean={statistics.mean(B5list_MI)}, SD={statistics.stdev(B5list_MI)}, TO={TO[0,2]}')
    
    print(f'S:B10: mean={statistics.mean(B10list_S)}, SD={statistics.stdev(B10list_S)}, TO={TO[1,0]}')
    print(f'B:B10: mean={statistics.mean(B10list_B)}, SD={statistics.stdev(B10list_B)}, TO={TO[1,1]}')
    print(f'MI:B10: mean={statistics.mean(B10list_MI)}, SD={statistics.stdev(B10list_MI)}, TO={TO[1,2]}')

    print(f'S:B15: mean={statistics.mean(B15list_S)}, SD={statistics.stdev(B15list_S)}, TO={TO[2,0]}')
    print(f'B:B15: mean={statistics.mean(B15list_B)}, SD={statistics.stdev(B15list_B)}, TO={TO[2,1]}')
    print(f'MI:B15: mean={statistics.mean(B15list_MI)}, SD={statistics.stdev(B15list_MI)}, TO={TO[2,2]}')

    print(f'S:B20: mean={statistics.mean(B20list_S)}, SD={statistics.stdev(B20list_S)}, TO={TO[3,0]}')
    print(f'B:B20: mean={statistics.mean(B20list_B)}, SD={statistics.stdev(B20list_B)}, TO={TO[3,1]}')
    print(f'MI:B20: mean={statistics.mean(B20list_MI)}, SD={statistics.stdev(B20list_MI)}, TO={TO[3,2]}')


#%% 
dataB=np.zeros((4))
dataMI=np.zeros((4))
for B in Budgets:
    avg=0
    PI_B=0
    PI_MI=0
    PI_S=0
    for N in Nodes:
        for loop in range(1,21):
            (S,Bo,MI)=data_poe[(N,B,loop)]
            if S==1 or Bo==1 or MI==1:  #don't calculate PI if 
                PI_B+=0
                PI_MI+=0
            else:
                (poe_S,poe_B,poe_MI)=data_poe[(N,B,loop)]
                #PI_B=PI_B+abs(poe_B-poe_S)/poe_S
                #PI_MI=PI_MI+abs(poe_MI-poe_S)/poe_S
                PI_S=PI_S+poe_S
                PI_B=PI_B+poe_B
                PI_MI=PI_MI+poe_MI
                avg=avg+1
    dataB[int(B/5-1)]=(PI_B/avg)/(PI_S/avg)
    dataMI[int(B/5-1)]=(PI_MI/avg)/(PI_S/avg)
dataS=[1,1,1,1]  
labels=[5,10,15,20]
         
width=0.3
plt.xticks(range(len(dataS)), labels)
plt.xlabel('Budget')
plt.ylabel('Normalized Probability of Evasion')
plt.title('BSPIAU and MISPIAU Improvement')
plt.bar(np.arange(len(dataS)), dataS, width=width)
plt.bar(np.arange(len(dataB))+ 1*width, dataB, width=width)
plt.bar(np.arange(len(dataMI))+ 2*width, dataMI, width=width)
colors = {'SPIAU':'blue','BSPIAU':'orange', 'MISPIAU':'green'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, loc='lower left')
plt.show()            


#%%
MI=[(16+14+15+18)/80, (20+19+19+20)/80,(20+20+19+19)/78, (20+19+14+13)/66]
B=[(6+2+2+5)/80,(10+5+3+7)/80, (11+7+5+9)/78,(16+8+6+5)/71]  
S=[0,0,0,0]
labels=[5,10,15,20]
         
width=0.3
plt.xticks(range(len(dataS)), labels)
plt.xlabel('Budget')
plt.ylabel('Percentage of Trials')
plt.title('Occurrences of Bolstering or Multiple Interdictions')
plt.bar(np.arange(len(S)), S, width=width)
plt.bar(np.arange(len(B)), B, width=width)
plt.bar(np.arange(len(MI))+ width, MI, width=width)
colors = {'BSPIAU':'orange', 'MISPIAU':'green'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.show()    

#%% 
data_B=np.zeros((4,4))
data_MI=np.zeros((4,4))
data_TO=np.zeros((4,4))
data_S=np.zeros((4,4))
for B in Budgets:
    for N in Nodes:
        avg=0
        time_B=0
        time_MI=0
        time_S=0
        for loop in range(1,21):
            (S,Bo,MI)=data_poe[(N,B,loop)]
            if S==1 or MI==1: #or Bo==1: #don't calculate if 
                time_B+=0
                time_MI+=0
            else:
                MI=dataMI[(N,B,loop)]
                S=dataS[(N,B,loop)]
                Bo=dataB[(N,B,loop)]
                #PI_B=PI_B+abs(poe_B-poe_S)/poe_S
                #PI_MI=PI_MI+abs(poe_MI-poe_S)/poe_S
                time_S=time_S+S
                time_B=time_B+B
                time_MI=time_MI+MI
                avg=avg+1
        data_B[int(B/5-1),int(N/50-1)]=(time_B/avg)/(time_S/avg)
        data_MI[int(B/5-1),int(N/50-1)]=(time_MI/avg)/(time_S/avg)
        data_S[int(B/5-1),int(N/50-1)]=1 
        data_TO[int(B/5-1),int(N/50-1)]=int(20-avg)
labels=[50,100,150,200]
        
width=0.2
plt.xticks([r+1.5*width for r in range(len(labels))], labels)
plt.xlabel('Nodes')
plt.ylabel('Normalized Solve Time')
plt.title('MISPIAU Computational Cost')
plt.bar(np.arange(len(labels)), data_MI[0,:], width=width, color='g',label='MISPIAU', edgecolor ='black')
#plt.bar(np.arange(len(labels)), data_S[0,:], width=width, color='b', label='SPIAU',edgecolor ='black')
plt.bar(np.arange(len(labels))+width, data_MI[1,:], width=width, color='g', edgecolor ='black')
#plt.bar(np.arange(len(labels))+width, data_S[1,:], width=width, color='b', edgecolor ='black')


plt.bar(np.arange(len(labels))+2*width, data_MI[2,:], width=width, color='g', edgecolor ='black')
#plt.bar(np.arange(len(labels))+2*width, data_S[2,:], width=width, color='b', edgecolor ='black')
plt.bar(np.arange(len(labels))+3*width, data_MI[3,:],  width=width, color='g', edgecolor ='black')
#plt.bar(np.arange(len(labels))+3*width, data_S[3,:], width=width, color='b', edgecolor ='black')
plt.axhline(y=1, color='b', linestyle='-',label='SPIAU')
plt.legend()
for B in Budgets:
    for N in Nodes:
        m=data_MI[int(B/5-1),int(N/50-1)]
        if data_TO[int(B/5-1),int(N/50-1)] >0:
            plt.text(N/50-1+(B/5-1-0.5)*width,m+0.5,f'({int(data_TO[int(B/5-1),int(N/50-1)])})',fontsize='large',color='k')  
        plt.text(N/50-1+(B/5-1-0.5)*width+0.05,-0.3,f'{B}',fontsize='small',color='k')
plt.show()




       
            
#%% Bolstered SPIAU
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import random
import time
from pyomo.environ import *
#from prettytable import PrettyTable, ALL
from shortestpath_networkx import create_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths
from shortestpath_networkx import create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths_multiple_interdictions
from shortestpath_networkx import create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx
import networkx as nx
import pickle

opt=SolverFactory('gurobi_direct')
opt.options['TimeLimit']=1800

Budgets={5,10,15,20}
Nodes={50,100,150,200}



#random.seed(a=631996) #Loops 1-10
random.seed(a=5111994) #Loops 11-20
data={}
for loop in range(11,21):
    print(f'Starting with loop {loop}')
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
        
        for B in Budgets:
            vdim=B*len(set(G.edges()))
            R={}
            R1={}
        
            for (i,j) in set(G.edges):
                r=G[i][j]['interdicted_length']
                k=0
                for (u,v) in set(G.edges):
                    if i==u and j==v:
                        R[(i,j,k)]=r
                        R1[(i,j,k)]=random.uniform(0.25*r,0.75*r)
                    k=k+1
            M=create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R,R1,s,t,vdim,B)
            start=time.time()
            results=opt.solve(M)
            end=time.time()
            data[(N,B,loop)]=(results.solver.wallclock_time)
            print(f'Finished with {B} Budget')
    print('')

np.save('data_BSPIAU_11_20.npy', data)  

#%% Data for BSPIAU


import numpy as np
data2=np.load("data_BSPIAU_1_10.npy", allow_pickle=True)
data10=data2.item()
data2=np.load("data_BSPIAU_11_20.npy", allow_pickle=True)
data20=data2.item()
data={**data10,**data20}


for N in Nodes:
    B5=0
    B10=0
    B15=0
    B20=0
    B5list=[]
    B10list=[]
    B15list=[]
    B20list=[]
    fig1, ax1 = plt.subplots()
    plt.title('BSPIAU: %i Nodes' % (N))
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Solve Time (s)')
    for loop in range(1,21):
        for B in Budgets:
            if B==5:
                if data[(N,B,loop)]>=1750:
                    B5=B5+1
                else:
                    B5list.append(data[(N,B,loop)])    
            elif B==10:
                if data[(N,B,loop)]>=1750:
                    B10=B10+1
                else:
                    B10list.append(data[(N,B,loop)])  
            elif B==15:
                if data[(N,B,loop)]>=1750:
                    B15=B15+1
                else:
                    B15list.append(data[(N,B,loop)])  
            elif B==20:
                if data[(N,B,loop)]>=1750:
                    B20=B20+1
                else:
                    B20list.append(data[(N,B,loop)])  
    ax1.boxplot([B5list, B10list,B15list, B20list], labels=[5,10,15,20], showfliers=True)
    m=max(B5list+B10list+B15list+B20list)
    plt.text(1,.85*m,f'({B5})',fontsize='x-large')
    plt.text(2,.85*m,f'({B10})',fontsize='x-large')
    plt.text(3,.85*m,f'({B15})',fontsize='x-large')
    plt.text(4,.85*m,f'({B20})',fontsize='x-large')
    
    print(f"N={N}")
    print(f'B5: mean={statistics.mean(B5list)}, SD={statistics.stdev(B5list)}')
    print(f'B10: mean={statistics.mean(B10list)}, SD={statistics.stdev(B10list)}')
    print(f'B15: mean={statistics.mean(B15list)}, SD={statistics.stdev(B15list)}')
    print(f'B20: mean={statistics.mean(B20list)}, SD={statistics.stdev(B20list)}')
    
    
#%% SPIAU (re do)

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import random
import time
from pyomo.environ import *
#from prettytable import PrettyTable, ALL
from shortestpath_networkx import create_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths
from shortestpath_networkx import create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx, return_paths_multiple_interdictions
from shortestpath_networkx import create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx
import networkx as nx
import pickle

opt=SolverFactory('gurobi_direct')
opt.options['TimeLimit']=1800

Budgets={5,10,15,20}
Nodes={50,100,150,200}



#random.seed(a=631996) #Loops 1-10
random.seed(a=5111994) #Loops 11-20
data={}
for loop in range(11,21):
    print(f'Starting with loop {loop}')
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
        R={}
        vdim=len(G.edges)
        
        for (i,j) in set(G.edges):
            r=G[i][j]['interdicted_length']
            k=0
            for (u,v) in set(G.edges):
                if i==u and j==v:
                    R[(i,j,k)]=r
                else:
                    R[(i,j,k)]=0
                k=k+1
        
        
        s=random.choice(list(G.nodes))
        t=random.choice(list(set(G.nodes)-{s}))
        
        for B in Budgets:
            
            M=create_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R,s,t,vdim,B)
            start=time.time()
            results=opt.solve(M)
            end=time.time()
            data[(N,B,loop)]=(results.solver.wallclock_time)
            print(f'Finished with {B} Budget')
    print('')

np.save('data_SPIAU_11_20_redo.npy', data) 
 

#%% 


import numpy as np
data2=np.load("data_SPIAU_1_10_redo.npy", allow_pickle=True)
data10=data2.item() 

data_real={}
for N in Nodes:
    for B in Budgets:
        for loop in range(1,11):
            data_real[(N,B,loop)]=data10[(N,B,loop)]
data2=np.load("data_SPIAU_11_20_redo.npy", allow_pickle=True)
data20=data2.item()
data={**data_real,**data20}


for N in Nodes:
    B5=0
    B10=0
    B15=0
    B20=0
    B5list=[]
    B10list=[]
    B15list=[]
    B20list=[]
    fig1, ax1 = plt.subplots()
    plt.title('SPIAU: %i Nodes' % (N))
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Solve Time (s)')
    for loop in range(1,21):
        for B in Budgets:
            if B==5:
                if data[(N,B,loop)]>=1750:
                    B5=B5+1
                else:
                    B5list.append(data[(N,B,loop)])    
            elif B==10:
                if data[(N,B,loop)]>=1750:
                    B10=B10+1
                else:
                    B10list.append(data[(N,B,loop)])  
            elif B==15:
                if data[(N,B,loop)]>=1750:
                    B15=B15+1
                else:
                    B15list.append(data[(N,B,loop)])  
            elif B==20:
                if data[(N,B,loop)]>=1750:
                    B20=B20+1
                else:
                    B20list.append(data[(N,B,loop)])  
    ax1.boxplot([B5list, B10list,B15list, B20list], labels=[5,10,15,20], showfliers=True)
    m=max(B5list+B10list+B15list+B20list)
    plt.text(1,.85*m,f'({int(B5)})',fontsize='x-large')
    plt.text(2,.85*m,f'({B10})',fontsize='x-large')
    plt.text(3,.85*m,f'({B15})',fontsize='x-large')
    plt.text(4,.85*m,f'({B20})',fontsize='x-large')
    
    print(f"N={N}")
    print(f'B5: mean={statistics.mean(B5list)}, SD={statistics.stdev(B5list)}')
    print(f'B10: mean={statistics.mean(B10list)}, SD={statistics.stdev(B10list)}')
    print(f'B15: mean={statistics.mean(B15list)}, SD={statistics.stdev(B15list)}')
    print(f'B20: mean={statistics.mean(B20list)}, SD={statistics.stdev(B20list)}')    

