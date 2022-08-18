"""
Functions
S. Punla-Green
"""

from pyomo.environ import *
import numpy as np
from numpy import linalg as LA
import random
import networkx as nx
infty=float('inf')
from itertools import islice, chain, combinations
from pyomo.gdp import *
from pyomo.mpec import *
import time
import matplotlib.pyplot as plt

#%% SPIAU
def create_shortest_path_nx(G,s,t):  #G is a directed NetworkX graph
    M=ConcreteModel()
    #Check if s and t are in the set of nodes
    if s not in list(G.nodes):
        raise Exception('{s} is not in the provided set of nodes')
    if t not in list(G.nodes):
        raise Exception('{t} is not in the provided set of nodes')
    
    M.d=Var(set(G.nodes),within=NonNegativeReals) #distance from s to a node
    M.Obj=Objective(expr=M.d[t], sense=maximize)
    M.c0=Constraint(expr= M.d[s]==0) #distance from s to s is 0
    
    def C(M,i,j):
        
        return M.d[j] <= M.d[i] + G[i][j]['length']
    M.c=Constraint(set(G.edges),rule=C) #Each directed edge should have a constraint           
    M.dual=Suffix(direction=Suffix.IMPORT) #Request the dual variables back from the solver
    return M



#Shortest Path with Interdiction
def create_shortest_path_interdiction_nx(G,s,t, B): #G is a directed NetworkX graph with length and interdicted_length
    M=ConcreteModel()
    #Check if s and t are in the set of nodes
    if s not in list(G.nodes):
        raise Exception('{s} is not in the provided set of nodes')
    if t not in list(G.nodes):
        raise Exception('{t} is not in the provided set of nodes') 
    M.d=Var(set(G.nodes),within=NonNegativeReals) #distance from s to a node
    M.x=Var(set(G.edges),within=Binary) #whether or not an edge is interdicted
    
    M.Obj=Objective(expr=M.d[t], sense=maximize)
    M.c0=Constraint(expr= M.d[s]==0) #distance from s to s is 0
    M.budget=Constraint(expr = sum(M.x[(u,v)] for (u,v) in set(G.edges)) <= B)
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['length'] + M.x[(i,j)]* G[i][j]['interdicted_length']
    M.c=Constraint(set(G.edges),rule=C) #Each directed edge should have a constraint       
    return M




#Shortest Path with Asymmetric Robust Uncertainty 
def create_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R,s,t,vdim,B):
    M=ConcreteModel()
    #if len(set(G.edges))*vdim != len(R):
    #    raise Exception('R is not appropriately dimensioned')
    #Check if s and t are in the set of nodes
    if s not in list(G.nodes):
        raise Exception('{s} is not in the provided set of nodes')
    if t not in list(G.nodes):
        raise Exception('{t} is not in the provided set of nodes') 
    M.d=Var(set(G.nodes),within=PositiveReals) #distance from s to a node
    M.x=Var(set(G.edges),within=Binary) #whether or not an edge is interdicted
    M.budget=Constraint(expr = sum(M.x[(u,v)] for (u,v) in set(G.edges)) <= B)
    M.V=RangeSet(1,vdim)
    M.v=Var(M.V,within=Reals)
    M.t=Var(M.V,within=Reals)
    M.w=Var(set(G.edges),within=NonNegativeReals)
    M.z=Var(set(G.edges),within=NonNegativeReals)
    M.s=Var(within=NonNegativeReals)
    
    def Obj(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        value=value -M.s
        return value
    M.Obj=Objective(rule=Obj,sense=maximize)
    
    
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['length'] + M.x[(i,j)]* G[i][j]['interdicted_length']
    M.c=Constraint(set(G.edges),rule=C) #Distance-based shortest path
    M.c0=Constraint(expr = M.d[s]==0) #Distance-based shortest path
    def c1(M,k):
        value=0
        for (i,j) in set(G.edges):
            if (i,j,k-1) in R.keys():
                value=value+R[(i,j,k-1)]*M.w[(i,j)]
        return M.t[k]==value
    M.c1=Constraint(M.V, rule=c1) # R^Tw=t
    
    M.c2=Constraint(expr= sum(M.t[i]*M.t[i] for i in M.V)<= M.s*M.s) # t^T t <= s^2
    def c3(M,i,j):
        return M.z[(i,j)] + M.x[(i,j)] <=1 
    M.c3=Constraint(set(G.edges),rule=c3) #zk + xk <= 1
    def c4(M,i,j):
        return M.w[(i,j)] - M.x[(i,j)] <= 0
    M.c4=Constraint(set(G.edges),rule=c4) #wk - xk <= 0
    def c5(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value+ l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return M.d[t]==value
    M.c5=Constraint(rule=c5)
    
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j)] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
  
    return M



#Shortest Path with Asymmetric Robust Uncertainty for multiple (S,T) pairs

def create_asymmetric_uncertainty_shortest_path_interdiction_multiple_ST_nx(G,R,I,vdim,B):
    M=ConcreteModel()
    if  len(set(G.edges))*vdim != len(R):
        raise Exception('R is not appropriately dimensioned')   
    M.x=Var(set(G.edges),within=Binary) #whether or not an edge is interdicted
    M.budget=Constraint(expr = sum(M.x[(u,v)] for (u,v) in set(G.edges)) <= B)
    M.V=RangeSet(1,vdim)
    M.m=Var(within=Reals)
    
    M.Obj=Objective(expr=M.m, sense=maximize)
    
    M.STBlocks=Block(I)
    
    #Loop for all possible (S,T) pairs
    for (S,T) in I:  
        M.STBlocks[(S,T)].d=Var(set(G.nodes),within=PositiveReals)
        M.STBlocks[(S,T)].t=Var(M.V,within=Reals)
        M.STBlocks[(S,T)].w=Var(set(G.edges),within=NonNegativeReals)
        M.STBlocks[(S,T)].z=Var(set(G.edges),within=NonNegativeReals)
        M.STBlocks[(S,T)].s=Var(within=NonNegativeReals)
    #Constraints
        M.STBlocks[(S,T)].C=ConstraintList()
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            M.STBlocks[(S,T)].C.add(M.STBlocks[(S,T)].d[j] <= M.STBlocks[(S,T)].d[i] + l + r*M.x[(i,j)])
        M.STBlocks[(S,T)].c0=Constraint(expr = M.STBlocks[(S,T)].d[S]==0)
        M.STBlocks[(S,T)].c1=ConstraintList()
        for k in M.V:
            M.STBlocks[(S,T)].c1.add(M.STBlocks[(S,T)].t[k]==sum(R[(i,j,k-1)]*M.STBlocks[(S,T)].w[(i,j)] for (i,j) in set(G.edges)))
        M.STBlocks[(S,T)].c2=Constraint(expr=sum(M.STBlocks[(S,T)].t[i]*M.STBlocks[(S,T)].t[i] for i in M.V)<= M.STBlocks[(S,T)].s*M.STBlocks[(S,T)].s)
        M.STBlocks[(S,T)].c3=ConstraintList()
        for (i,j) in set(G.edges):
            M.STBlocks[(S,T)].c3.add(M.STBlocks[(S,T)].z[(i,j)] + M.x[(i,j)] <=1 )
        M.STBlocks[(S,T)].c4=ConstraintList()
        for (i,j) in set(G.edges):
            M.STBlocks[(S,T)].c4.add(M.STBlocks[(S,T)].w[(i,j)] - M.x[(i,j)] <= 0)
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value+ l*M.STBlocks[(S,T)].z[(i,j)] + (l+r)*M.STBlocks[(S,T)].w[(i,j)]
        M.STBlocks[(S,T)].c5=Constraint(expr=M.STBlocks[(S,T)].d[T]==value)
        M.STBlocks[(S,T)].c6=ConstraintList()
        for J in set(G.nodes):
            if J==S:
                rhs=1
            elif J==T:
                rhs=-1
            else:
                rhs=0
            RS=0
            FS=0
            for (i,j) in set(G.out_edges(J)):
                FS= FS + M.STBlocks[(S,T)].w[(i,j)] + M.STBlocks[(S,T)].z[(i,j)]
            for (i,j) in set(G.in_edges(J)):
                RS= RS + M.STBlocks[(S,T)].w[(i,j)] + M.STBlocks[(S,T)].z[(i,j)]
                   
            M.STBlocks[(S,T)].c6.add(FS-RS == rhs )        
        
    
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.STBlocks[(S,T)].z[(i,j)] + (l+r)*M.STBlocks[(S,T)].w[(i,j)]
        value=value -M.STBlocks[(S,T)].s
        M.STBlocks[(S,T)].c7=Constraint(expr=value >= M.m)
 
    return M


#Return path




def return_paths(M,G,R,S,T):
    tol=1e-5
    G_adj=nx.DiGraph()
    norm=np.sqrt(sum(M.t[i].value*M.t[i].value for i in M.V))
    for (i,j) in set(G.edges):
        if (M.w[(i,j)].value >= 1/len(list(G.nodes))) or (M.z[(i,j)].value >= 1/len(list(G.nodes))):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            if M.x[(i,j)].value >=0.9:
                length=l+r
                adjustment=0
                if norm>tol: # Tune this cut off value
                    for k in M.V:
                        if (i,j,k-1) in R.keys():
                            adjustment=adjustment+(-1/norm)*(R[(i,j,k-1)]*M.t[k].value)
                    
                length=length+adjustment
            else:
                length=l
            G_adj.add_edge(i,j,adjusted_length=length)
    #Finished creating a network with only edges that are traversed in the problem with adjusted edges lengths
    #Use Network_X to find ALL shortest paths
    #print(G_adj[3][342]['adjusted_length'])
    #print(G_adj[3][352]['adjusted_length'])
    paths=nx.all_shortest_paths(G_adj, source=S, target=T, weight='adjusted_length')
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='adjusted_length', method='dijkstra')
    return paths,lengths

def return_interdicted_arcs(M,G):
    interdicted=[]
    for (i,j) in set(G.edges):
        if M.x[(i,j)].value >= 0.9:
            interdicted.append((i,j))
    return interdicted
        
def return_paths_multiple_ST(M,G,R,S,T):
    tol=1e-5
    G_adj=nx.DiGraph()
    norm=np.sqrt(sum(M.STBlocks[(S,T)].t[i].value*M.STBlocks[(S,T)].t[i].value for i in M.V))
    for (i,j) in set(G.edges):
        if (M.STBlocks[(S,T)].w[(i,j)].value >= 1/len(list(G.nodes))) or (M.STBlocks[(S,T)].z[(i,j)].value >= 1/len(list(G.nodes))):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            if M.x[(i,j)].value >=0.9:
                length=l+r
                if norm>tol: # Tune this cut off value
                    adjustment=(-1/norm)*(sum(R[(i,j,k-1)]*M.STBlocks[(S,T)].t[k].value for k in M.V))
                    length=length+adjustment
            else:
                length=l
            G_adj.add_edge(i,j,adjusted_length=length)
    #Finished creating a network with only edges that are traversed in the problem with adjusted edges lengths
    #Use Network_X to find ALL shortest paths
    paths=nx.all_shortest_paths(G_adj, source=S, target=T, weight='adjusted_length')
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='adjusted_length', method='dijkstra')
    return paths, lengths


def similar_length(M,G,R,S,T,cutoff): #cutoff=percentage of shortest path longer ie 110% would be cutoff=1.10
    tol=1e-5    
    (paths, lengths)=return_paths(M,G,R,S,T)   
    
    G_adj=nx.Graph()
    norm=np.sqrt(sum(M.t[i].value*M.t[i].value for i in M.V))
    for (i,j) in G.edges:
        l=G[i][j]['length']
        r=G[i][j]['interdicted_length']
        if M.x[(i,j)].value >=0.9:
            length=l+r
            if norm>tol:
                adjustment=(-1/norm)*(sum(R[(i,j,k-1)]*M.t[k].value for k in M.V))
                length=length+adjustment
        else:
            length=l
        if G_adj.has_edge(j,i):
            #Check which length to use
            if G_adj[j][i]['adjusted_length'] <= length: #Shorter Lengths indicate that the edge has been interdicted so we shoudl use that value for both directions
                G_adj[i][j]['adjusted_length']=length
        else:
            G_adj.add_edge(i,j,adjusted_length=length)
    
    similar_paths=[]
    all_paths=list(islice(nx.shortest_simple_paths(G_adj, S, T, weight='adjusted_length'), 30))
    
    for path in all_paths:
        if nx.path_weight(G_adj, path, weight='adjusted_length') <= cutoff*lengths:
            similar_paths.append(path)
        else:
            break
    no_of_similar_paths=len(similar_paths)
    return (similar_paths, no_of_similar_paths)

#%% MISPIAU and BSPIAU

def create_multiple_interdictions_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R,s,t,vdim,B):
    M=ConcreteModel()
    '''
    if B*len(set(G.edges))*vdim != len(R):
        raise Exception('R is not appropriately dimensioned')
    '''
    #Check if s and t are in the set of nodes
    if s not in list(G.nodes):
        raise Exception('{s} is not in the provided set of nodes')
    if t not in list(G.nodes):
        raise Exception('{t} is not in the provided set of nodes') 
    M.d=Var(set(G.nodes),within=PositiveReals) #distance from s to a node
    M.B=RangeSet(1,B)
    
    M.x=Var(Any,dense=False,within=Binary) #whether or not an edge is interdicted (indexed by (i,j), b)
    M.budget=Constraint(expr = sum(sum(M.x[(u,v),b] for (u,v) in set(G.edges)) for b in M.B) <= B)
    M.V=RangeSet(1,vdim)
    M.v=Var(M.V,within=Reals)
    M.t=Var(Any,dense=False,within=Reals) #t[vdim,b]
    M.w0=Var(set(G.edges),within=NonNegativeReals, bounds=(0,1))
    M.w=Var(Any,dense=False,within=NonNegativeReals, bounds=(0,1))
    M.z=Var(set(G.edges),within=NonNegativeReals, bounds=(0,1))
    M.s=Var(within=NonNegativeReals)
    
    def Obj(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.z[(i,j)] + l*M.w[(i,j),1]
            value=value+r*sum(M.w[(i,j),b] for b in M.B)
        value=value -M.s
        return value
    M.Obj=Objective(rule=Obj,sense=maximize)
    
    
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['length'] + sum(M.x[(i,j),b] for b in M.B)* G[i][j]['interdicted_length']
    M.c=Constraint(set(G.edges),rule=C) #Distance-based shortest path
    M.c0=Constraint(expr = M.d[s]==0) #Distance-based shortest path
    

    def c1(M,k,b):
        value=0
        for (i,j) in set(G.edges):
            if (i,j,k-1,b) in R.keys():
                value=R[(i,j,k-1,b)*M.w[(i,j),b]]
        return M.t[k,b]==value
    M.c1=ConstraintList()
    for k in M.V:
        for b in M.B:
            value=0
            for (i,j) in set(G.edges):
                if (i,j,k-1,b) in R.keys():
                    value=R[(i,j,k-1,b)]*M.w[(i,j),b]
            M.c1.add(M.t[k,b]==value)
    
    M.c2=Constraint(expr= sum(sum(M.t[i,b]*M.t[i,b] for i in M.V) for b in M.B)<= M.s*M.s) # t^T t <= s^2
    M.c3=ConstraintList()
    for (i,j) in G.edges:
        #M.c3.add(M.w0[(i,j)] - sum(M.w[(i,j),b] for b in M.B) <= 0)
        for b in M.B:
            M.c3.add(M.z[(i,j)] + M.x[(i,j),b] <=1 )
            M.c3.add(M.w[(i,j),b] - M.x[(i,j),b] <= 0)
            
        for b in range(1,B):
            M.c3.add(M.x[(i,j),b+1]-M.x[(i,j),b]<=0)
            M.c3.add(M.w[(i,j),b+1]-M.w[(i,j),1] <=0)
            M.c3.add(M.w[(i,j),b+1] >= M.x[(i,j),b+1] + M.w[(i,j),1]-1)
    
    def c5(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value+ l*M.z[(i,j)] + l*M.w[(i,j),1] + sum(M.w[(i,j),b] for b in M.B)*r
        return M.d[t]==value
    M.c5=Constraint(rule=c5)
    '''
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w0[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j),1] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
    '''
    M.c7=ConstraintList()
    for J in set(G.nodes):
            if J==s:
                rhs=1
            elif J==t:
                rhs=-1
            else:
                rhs=0
            RS=0
            FS=0 
            for (i,j) in set(G.out_edges(J)):
                FS= FS + M.w[(i,j),1] + M.z[(i,j)]    
            for (i,j) in set(G.in_edges(J)):
                RS= RS + M.w[(i,j),1] + M.z[(i,j)]
            M.c7.add(FS-RS == rhs)
    
    return M

def return_paths_multiple_interdictions(M,G,R,S,T):
    tol=1e-5
    G_adj=nx.DiGraph()
    norm=np.sqrt(sum(sum(M.t[i,b].value*M.t[i,b].value for i in M.V) for b in M.B))
    for (i,j) in set(G.edges):
        if (M.w[(i,j),1].value >= 1/len(list(G.nodes))) or (M.z[(i,j)].value >= 1/len(list(G.nodes))):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            length=l
            for b in M.B:
                if M.x[(i,j),b].value >=0.9:
                    length=length+r
                    if norm>tol: # Tune this cut off value
                        value=0
                        for k in M.V:
                            if (i,j,k-1,b) in R.keys():
                                value=value+R[(i,j,k-1,b)]*M.t[k,b].value
                        adjustment=(-1/norm)*value
                        length=length+adjustment
            G_adj.add_edge(i,j,adjusted_length=length)
    #Finished creating a network with only edges that are traversed in the problem with adjusted edges lengths
    #Use Network_X to find ALL shortest paths
    #print(G_adj[3][342]['adjusted_length'])
    #print(G_adj[3][352]['adjusted_length'])
    paths=nx.all_shortest_paths(G_adj, source=S, target=T, weight='adjusted_length')
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='adjusted_length', method='dijkstra')
    return paths,lengths

def return_paths_bolstered(M,G,R,R1,S,T):
    tol=1e-5
    G_adj=nx.DiGraph()
    norm=np.sqrt(sum(M.t[i].value*M.t[i].value for i in M.V))
    for (i,j) in set(G.edges):
        if (M.w[(i,j)].value >= 1/len(list(G.nodes))) or (M.z[(i,j)].value >= 1/len(list(G.nodes))):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            if M.x[(i,j)].value >=0.9:
                length=l+r
                if norm>tol: # Tune this cut off value
                    if M.b[(i,j)].value >=0.9:
                        adjustment2=0
                        for k in M.V:
                           if (i,j,k-1) in R1.keys(): 
                               adjustment2=adjustment2+(-1/norm)*(R1[(i,j,k-1)]*M.t[k].value)
                    else:
                        adjustment2=0
                    adjustment=0
                    for k in M.V:
                        if (i,j,k-1) in R.keys():
                            adjustment=adjustment+(-1/norm)*(R[(i,j,k-1)]*M.t[k].value)
                    length=length+adjustment-adjustment2
            else:
                length=l
            G_adj.add_edge(i,j,adjusted_length=length)
    #Finished creating a network with only edges that are traversed in the problem with adjusted edges lengths
    #Use Network_X to find ALL shortest paths
    #print(G_adj[3][342]['adjusted_length'])
    #print(G_adj[3][352]['adjusted_length'])
    paths=nx.all_shortest_paths(G_adj, source=S, target=T, weight='adjusted_length')
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='adjusted_length', method='dijkstra')
    return paths,lengths


def create_bolstered_asymmetric_uncertainty_shortest_path_interdiction_nx(G,R, R1, s,t,vdim,B, cI=1, cB=1):
    M=ConcreteModel()
    #Check if s and t are in the set of nodes
    if s not in list(G.nodes):
        raise Exception('{s} is not in the provided set of nodes')
    if t not in list(G.nodes):
        raise Exception('{t} is not in the provided set of nodes') 
    M.d=Var(set(G.nodes),within=PositiveReals) #distance from s to a node
    M.x=Var(set(G.edges),within=Binary) #whether or not an edge is interdicted
    M.b=Var(set(G.edges),within=Binary) #whether or not an edge is bolstered
    M.budget=Constraint(expr = sum(cI*M.x[(u,v)]+ cB*M.b[(u,v)] for (u,v) in set(G.edges)) <= B)
    M.V=RangeSet(1,vdim)
    M.v=Var(M.V,within=Reals)
    M.t=Var(M.V,within=Reals)
    M.w=Var(set(G.edges),within=NonNegativeReals)
    M.wb=Var(set(G.edges),within=NonNegativeReals)
    M.z=Var(set(G.edges),within=NonNegativeReals)
    M.s=Var(within=NonNegativeReals)
    
    def Obj(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        value=value -M.s
        return value
    M.Obj=Objective(rule=Obj,sense=maximize)
    
    
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['length'] + M.x[(i,j)]* G[i][j]['interdicted_length']
    M.c=Constraint(set(G.edges),rule=C) #Distance-based shortest path
    M.c0=Constraint(expr = M.d[s]==0) #Distance-based shortest path
    def c1(M,k):
        value=0
        for (i,j) in set(G.edges):
            if (i,j,k-1) in R.keys():
                value=value+ R[(i,j,k-1)]*M.w[(i,j)]
            if (i,j,k-1) in R1.keys():
                value=value- R1[(i,j,k-1)]*M.wb[(i,j)]
        return M.t[k]==value
    M.c1=Constraint(M.V, rule=c1) # R^Tw - R1^Tw^b=t
    
    M.c2=Constraint(expr= sum(M.t[i]*M.t[i] for i in M.V)<= M.s*M.s) # t^T t <= s^2
    def c3(M,i,j):
        return M.z[(i,j)] + M.x[(i,j)] <=1 
    M.c3=Constraint(set(G.edges),rule=c3) #zk + xk <= 1
    def c4(M,i,j):
        return M.w[(i,j)] - M.x[(i,j)] <= 0
    M.c4=Constraint(set(G.edges),rule=c4) #wk - xk <= 0
    def c4a(M,i,j):
        return M.wb[(i,j)] - M.b[(i,j)] <= 0
    M.c4a=Constraint(set(G.edges),rule=c4a) #wk - xk <= 0
    
    def c4b(M,i,j):
        return M.b[(i,j)] - M.x[(i,j)] <= 0
    M.c4b=Constraint(set(G.edges),rule=c4b) #wk - xk <= 0
    
    def c4c(M,i,j):
        return M.wb[(i,j)] - M.w[(i,j)] <= 0
    M.c4c=Constraint(set(G.edges),rule=c4c) #wk - xk <= 0
    
    def c5(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value+ l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return M.d[t]==value
    M.c5=Constraint(rule=c5)
    
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j)] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
    
    def c7(M,i,j):
        return M.wb[(i,j)] >= M.x[(i,j)] + M.w[(i,j)]-1
    M.c7=Constraint(set(G.edges),rule=c7)
  
    return M

#%% SPI UAI

def SPI_UAI_algorithm_mod(G,R,S,T,B, max_it=100, time_limit=3600, tol=1e-6):
    
    import logging
    data=[]

    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    iteration=0
    #flag_time_out=0
    Master_time=0
    Sub_time=0
    Sub2_time=0
    Master_write_time=0
    Sub_write_time=0
    #G - graph with length, interdicted_length, and perceived_length
    #R[(i,j)]=(possible perceived values of r_ij)
    r_perceived={}
    #opt=SolverFactory('gurobi_persistent')
    opt=SolverFactory('gurobi')
    opt.options['TimeLimit']=time_limit
    #opt.options["tee"]=False
    #Initialize
    
    r_temp={}
    for (i,j) in set(G.edges):
        r_temp[('upper',i,j)]=max(list(R[(i,j)]))
        r_temp[('lower',i,j)]=min(list(R[(i,j)]))
    k=0   
    for (i,j) in set(G.edges): #the i,j one is the lower point, all other edges at upper
        for (m,n) in set(G.edges):
            if (m,n)==(i,j):
                r_perceived[(k,m,n)]=r_temp[('lower',m,n)]
            else:
                r_perceived[(k,m,n)]=r_temp[('upper',m,n)]
        k=k+1
    for(m,n) in set(G.edges):
        r_perceived[(k,m,n)]=r_temp[('upper',m,n)]           
      
    flag=0
    Idim0=int(len(r_perceived)/len(set(G.edges())))
    Idim=Idim0
    x={}
    sub2_flag=1
    time_out_flag=0
    while flag!=1 and iteration <= max_it:
        #Solve the Master Problem
        t0=time.time()
        M=create_uncertain_asymmetric_information_shortest_path_interdiction_nx(G,r_perceived,S,T,Idim,B)
        #if x:
        #    for (i,j) in G.edges():
        #        M.x[(i,j)] = x[(i,j)]
        t1=time.time()
        Master_write_time=Master_write_time+(t1-t0)
        t0=time.time()
        #opt.set_instance(M)
        #results=opt.solve(M, warmstart=True, tee=True)
        #results=opt.solve(M, tee=True)
        results=opt.solve(M)
        t1=time.time()
        Master_time+=t1-t0
        if results.solver.termination_condition == TerminationCondition.optimal:
            m_star=value(M.m)
            #print(m_star)
            time_out_flag=0
        else:
            m_star=results.problem.upper_bound
            print(f'Time Out at iteration={iteration}')
            #print(m_star)
            time_out_flag+=1
            
        x_old=x
        x_new={}
        x_keys=[]
        for (i,j) in set(G.edges()):
            if value(M.x[(i,j)])>0.9:
                x_new[(i,j)]=1
                x_keys.append((i,j))
            else:
                x_new[(i,j)]=0        
    
        x=x_new
        #testing
        #m_star=results.problem.upper_bound

            
            
            
        
        if iteration >= 1:
            sub2_flag=1
            count=0
            for (i,j) in set(G.edges):
                if x_new[(i,j)]==x_old[(i,j)]:
                    count+=1
            if count==len(set(G.edges)):
                #print(f'Iteration={iteration}: x did not change with new R')
                #raise RuntimeError('x did not change with new R')
                #DEBUG HERE
                #(f'M_sub Object={value(M_sub.Obj)}')
                #print(f'm_star={m_star}')
                #Get Subproblem Path & Length
                #(path,length)=return_path_uncertain_asymmetric_information(M_sub,x,G,S,T)
                #print('Subproblem')
                #print(path)
                #print(length)
                #Get Master Problem Path & Length
                #print('Master')
                sub2_flag=0
        
        #Solve the Subproblem
        t0=time.time()
        #M_sub=create_SPI_UAI_subproblem_all_r(G, R, x, S, T)
        M_sub=create_SPI_UAI_subproblem(G, R, x, S, T)
        t1=time.time()
        Sub_write_time+=t1-t0
        t0=time.time()
        opt.solve(M_sub)
        t1=time.time()
        Sub_time+=t1-t0
        data.append((value(M_sub.Obj)-m_star)/m_star)
        print(f'Iteration={iteration}: {(value(M_sub.Obj)-m_star)/m_star}')
        if abs(value(M_sub.Obj)-m_star) <= tol:
            flag=1
            #break
        elif time_out_flag>0: 
            #CHECK IF WE FOUND A NEW R
            for I in range(0,Idim):
                new_r=0
                for (i,j) in x_keys:
                    if value(M_sub.r[(i,j)]) == r_perceived[(I,i,j)]:
                        new_r+=1
                if new_r==B:
                    #This R has been found before
                    opt.options["warmstart"]=True
                    new_r_flag=1
                    
                    if time_out_flag>=2:
                        return M, 99999, 99999 , 99999, iteration, Master_time, Sub_time, Sub2_time
                        #raise RuntimeError('Two consecutive Master Problems could not be solved in the time limit. Consider raising the time limit or reducing your problem size. ')
                    print('Master problem timed out. Subproblem did not find new instance. Trying again with warmstart.')
                    break
                else: new_r_flag=0
            if new_r_flag==0:
                for (i,j) in set(G.edges()):
                    if (i,j) in x_keys:
                        r_perceived[(Idim,i,j)]=value(M_sub.r[(i,j)])
                        r_perceived[(Idim+1,i,j)]=value(M_sub.r[(i,j)])
                        r_perceived[(Idim+2,i,j)]=value(M_sub.r[(i,j)])
                    else:
                        r_perceived[(Idim,i,j)]=r_temp[('upper',i,j)]  
                        r_perceived[(Idim+1,i,j)]=r_temp[('lower',i,j)]  
                        r_perceived[(Idim+2,i,j)]=random.choice([r_temp[('upper',i,j)], r_temp[('lower',i,j)]])
                Idim=Idim+3
                #THIS HAS BEEN CHANGED TO WARMSTART ALWAYS
                opt.options["warmstart"]=True
                time_out_flag=0
                
                
        else:
            for (i,j) in set(G.edges()):
                if (i,j) in x_keys:
                    r_perceived[(Idim,i,j)]=value(M_sub.r[(i,j)])
                    r_perceived[(Idim+1,i,j)]=value(M_sub.r[(i,j)])
                    r_perceived[(Idim+2,i,j)]=value(M_sub.r[(i,j)])
                else:
                    r_perceived[(Idim,i,j)]=r_temp[('upper',i,j)]  
                    r_perceived[(Idim+1,i,j)]=r_temp[('lower',i,j)]  
                    r_perceived[(Idim+2,i,j)]=random.choice([r_temp[('upper',i,j)], r_temp[('lower',i,j)]])
            Idim=Idim+3
            opt.options["warmstart"]=False
                      
         
        '''
        if sub2_flag==1:
            
            #Subproblem 2
            t0=time.time()
                   
            H=list(powerset(x_keys))
            for high in H:
                M_sub2=create_subproblem_2(G,high,r_temp,S,T)
                opt.solve(M_sub2)
                #Take the w,z values and find the true path length
                value0=0
                for (i,j) in set(G.edges):
                    l=G[i][j]['length']
                    r=G[i][j]['interdicted_length']
                    value0=value0+ l*value(M_sub2.z[(i,j)]) + (l+r)*value(M_sub2.w[(i,j)])
                    
                if value0 < m_star:
                    for (i,j) in set(G.edges()):
                        if (i,j) in x_keys:
                            if (i,j) in high:
                                r_perceived[(Idim,i,j)]=r_temp[('upper',i,j)]
                                r_perceived[(Idim+1,i,j)]=r_temp[('upper',i,j)]
                                r_perceived[(Idim+2,i,j)]=r_temp[('upper',i,j)]
                            else:
                                r_perceived[(Idim,i,j)]=r_temp[('lower',i,j)]
                                r_perceived[(Idim+1,i,j)]=r_temp[('lower',i,j)]
                                r_perceived[(Idim+2,i,j)]=r_temp[('lower',i,j)]
                                
                        else:
                            r_perceived[(Idim,i,j)]=r_temp[('upper',i,j)]  
                            r_perceived[(Idim+1,i,j)]=r_temp[('lower',m,n)]  
                            r_perceived[(Idim+2,i,j)]=random.choice([r_temp[('upper',m,n)], r_temp[('lower',m,n)]])
                Idim=Idim+3
            
            
            #Alternate Subproblem
            
            max_R=min(2**B,75)
            (r_perceived, Idim)=subproblem2_networkx(G,x_keys,r_temp,S,T, max_R, m_star, r_perceived, Idim)
            #
            t1=time.time()  
            Sub2_time+=t1-t0
            '''
                    
        iteration+=1 
        #End of While Loop
        
    if iteration>max_it:
        print('Max Iterations Reached')
        paths=[]
        lengths=[]
    #elif flag_time_out==1:
    #    print('Ultimate Iteration Timed Out: Results Suspect')
    else:
        print(f'{iteration} iterations to find optimal solution')
        (paths,lengths)=return_path_uncertain_asymmetric_information(M_sub,x,G,S,T)
        #M_sub.r.pprint()
    '''
    print(f'Master Solve Time = {Master_time} s')
    print(f'Master Time to Create Model = {Master_write_time} s')
    print(f'Subproblem Solve Time = {Sub_time} s')
    print(f'Subproblem Time to Create Model = {Sub_write_time} s')
    print(f'Subproblem 2 Time = {Sub2_time} s')
    '''
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Sub_obj-m*')
    ax1.plot(data, 'g-o')
    
    return M, paths, lengths , m_star, iteration, Master_time, Sub_time, Sub2_time



def return_path_uncertain_asymmetric_information(M,x,G,S,T):
    
    G_adj=nx.DiGraph()
    for (i,j) in set(G.edges):
        l=G[i][j]['perceived_length']
        r=M.r[(i,j)].value
        length=l+r*x[(i,j)]
        G_adj.add_edge(i,j,adjusted_length=length)
    paths=nx.all_shortest_paths(G_adj, source=S, target=T, weight='adjusted_length')
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='adjusted_length', method='dijkstra')
    return paths,lengths
        
def return_path_nominal_SPI(M,G,S,T):
    G_adj=nx.DiGraph()
    for (i,j) in set(G.edges):
        l=G[i][j]['length']
        r=G[i][j]['interdicted_length']
        G_adj.add_edge(i,j,length=l+r*M.x[(i,j)].value)
    paths=nx.all_shortest_paths(G_adj,source=S,target=T, weight='length' )
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='length', method='dijkstra')
    return paths, lengths


def create_SPI_asymmetric_information(M,G,s,t,B):
    M=ConcreteModel()
    M.d=Var(set(G.nodes),within=PositiveReals) #distance from s to a node
    M.x=Var(set(G.edges),within=Binary) #whether or not an edge is interdicted
    M.budget=Constraint(expr = sum(M.x[(u,v)] for (u,v) in set(G.edges)) <= B)
    M.w=Var(set(G.edges),within=NonNegativeReals)
    M.z=Var(set(G.edges),within=NonNegativeReals)
    
    def Obj(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return value
    M.Obj=Objective(rule=Obj,sense=maximize)
    
    
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['perceived_length'] + M.x[(i,j)]* G[i][j]['perceived_interdicted_length']
    M.c=Constraint(set(G.edges),rule=C) #Distance-based shortest path
    M.c0=Constraint(expr = M.d[s]==0) #Distance-based shortest path

    
    def c3(M,i,j):
        return M.z[(i,j)] + M.x[(i,j)] <=1 
    M.c3=Constraint(set(G.edges),rule=c3) #zk + xk <= 1
    def c4(M,i,j):
        return M.w[(i,j)] - M.x[(i,j)] <= 0
    M.c4=Constraint(set(G.edges),rule=c4) #wk - xk <= 0
    def c5(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['perceived_length']
            r=G[i][j]['perceived_interdicted_length']
            value=value+ l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return M.d[t]==value
    M.c5=Constraint(rule=c5)
    
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j)] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
    
    return M


def return_path_asymmetric_information_SPI(M,G,S,T):
    G_adj=nx.DiGraph()
    for (i,j) in set(G.edges):
        l=G[i][j]['perceived_length']
        r=G[i][j]['perceived_interdicted_length']
        G_adj.add_edge(i,j,length=l+r*M.x[(i,j)].value)
    paths=nx.all_shortest_paths(G_adj,source=S,target=T, weight='length' )
    paths=list(paths)
    lengths=nx.shortest_path_length(G_adj, source=S, target=T, weight='length', method='dijkstra')
    return paths, lengths

def SPI_UAI_regret_avoided(M_SPI,M, G,R,S,T):
    x={}
    for (i,j) in set(G.edges()):
        if value(M_SPI.x[(i,j)])>0.9:
            x[(i,j)]=1
        else:
            x[(i,j)]=0
    M_sub=create_SPI_UAI_subproblem(G,R,x,S,T)
    opt=SolverFactory('gurobi')
    opt.solve(M_sub)
    (paths,length)=return_path_uncertain_asymmetric_information(M_sub,x,G,S,T)
    return (M_sub.Obj, M.Obj, paths, length)
    




def create_SPI_UAI_subproblem_all_r(G,R,x,s,t):
    
    M=ConcreteModel()
    
    def _bounds_rule(M, i, j):
        r_max=max(list(R[(i,j)]))
        return (0, r_max)
    
    M.r = Var(set(G.edges), bounds=_bounds_rule)
    #M.r=Var(set(G.edges),within=NonNegativeReals)
    M.w=Var(set(G.edges), bounds=(0,1))
    M.z=Var(set(G.edges), bounds=(0,1))
    value=0
    for (i,j) in G.edges:
        value=value+G[i][j]['length']+max(list(R[(i,j)]))
    M.d=Var(set(G.nodes), bounds=(0,2*value))
    
    def Obj(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return value
    M.Obj=Objective(rule=Obj, sense=minimize)    
    
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['perceived_length'] + x[(i,j)]*M.r[(i,j)]
    M.c=Constraint(set(G.edges),rule=C) #Distance-based shortest path
    M.c0=Constraint(expr = M.d[s]==0) #Distance-based shortest path
    
    def c3(M,i,j):
        return M.z[(i,j)] + x[(i,j)] <=1 
    M.c3=Constraint(set(G.edges),rule=c3) #zk + xk <= 1
    def c4(M,i,j):
        return M.w[(i,j)] - x[(i,j)] <= 0
    M.c4=Constraint(set(G.edges),rule=c4) #wk - xk <= 0
    
    
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j)] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
    
    #ONLY DO r as a variable if x[(i,j)=1]
    '''
    M.set_r=ConstraintList()
    for (i,j) in set(G.edges):
        if (i,j) not in x_keys:
            (l,h)=R[(i,j)]
            M.set_r.add(expr=M.r[(i,j)]==h)
    '''   
    
    #R IS FROM INSTANCES
    M.Disjunction=Block(set(G.edges))
    for (i,j) in set(G.edges):
        r_list=R[(i,j)]
        M.Disjunction[(i,j)].rw=Var(bounds=(0,max(r_list)))
        M.Disjunction[(i,j)].RBlocks=Disjunct(range(0,len(r_list)))
        r0=0
        for r in r_list:
            M.Disjunction[(i,j)].RBlocks[r0].c1=Constraint(expr = M.r[(i,j)] == r)
            M.Disjunction[(i,j)].RBlocks[r0].c2=Constraint(expr = M.Disjunction[(i,j)].rw==r*M.w[(i,j)] )
            r0=r0+1
        M.Disjunction[(i,j)].Disj=Disjunction(expr = [M.Disjunction[(i,j)].RBlocks[r1] for r1 in range(0,len(r_list))])
    
    
    value=0
    for (i0,j0) in set(G.edges):
        l=G[i0][j0]['perceived_length']
        value=value+ l*M.z[(i0,j0)] + (l)*M.w[(i0,j0)] + M.Disjunction[(i0,j0)].rw

        
    M.c5=Constraint(expr = value == M.d[t])
    
    TransformationFactory('mpec.simple_disjunction').apply_to(M)
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(M)
    return M 



#%% Modified Algorithm
def create_subproblem_2(G,high,r_temp,s,t):
    M=ConcreteModel()
    
    M.w=Var(set(G.edges), bounds=(0,1))
    M.z=Var(set(G.edges), bounds=(0,1))
    
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j)] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
    
    def c5(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['perceived_length']
            if (i,j) in high:
                r=r_temp[('upper',i,j)]
            else:
                r=r_temp[('lower',i,j)]
            value=value+ l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return value
    M.obj=Objective(rule=c5,sense=minimize)
    
    return M


def subproblem2_networkx(G,x_keys,r_temp,s,t, max_R, m_star, r_perceived, Idim):
    #Create modified network
    
    G_mod = G.copy()
    for (i,j) in x_keys:
        #Remove edges from (i,j) in x.keys():
        l=G[i][j]['perceived_length']
        G_mod.remove_edge(i,j)
        #Add dummy nodes iUj and iLj
        G_mod.add_node(f"{i}"+"U"+f"{j}")
        G_mod.add_node(f"{i}"+"L"+f"{j}")
        #Connect dummy nodes with appropriate lengths
        G_mod.add_edge(i,f"{i}"+"U"+f"{j}", perceived_length=l+r_temp[('upper',i,j)])
        G_mod.add_edge(i,f"{i}"+"L"+f"{j}", perceived_length=l+r_temp[('lower',i,j)])
        G_mod.add_edge(f"{i}"+"U"+f"{j}",j, perceived_length=0)
        G_mod.add_edge(f"{i}"+"L"+f"{j}",j, perceived_length=0)
    #Find all paths through network in decreasing length
    all_paths=list(islice(nx.shortest_simple_paths(G_mod, s, t, weight='perceived_length'), max_R))
    
    r_add=[]
    for path in all_paths:
        if nx.path_weight(G_mod, path, weight='perceived_length') < m_star:
            #Extract r information. 
            r_add_temp=[]
            for node in path: #add these to a set to avoid duplicates
                for (i,j) in x_keys:
                    if f"{i}"+"U"+f"{j}" in str(node):
                        r_add_temp.append((i,j,'u'))
                    elif f"{i}"+"L"+f"{j}" in str(node):
                        r_add_temp.append((i,j,'l'))
            if r_add_temp not in r_add:
                r_add.append(r_add_temp)
        else:
            break
    #Turn my list of parameters into parameters for the Master problem
    for r_list in r_add:
        for (i,j) in set(G.edges):
            if (i,j,'u') in r_list:
                r_perceived[(Idim,i,j)]=r_temp[('upper',i,j)]
                r_perceived[(Idim+1,i,j)]=r_temp[('upper',i,j)]
                r_perceived[(Idim+2,i,j)]=r_temp[('upper',i,j)]
            elif (i,j,'l') in r_list:
                r_perceived[(Idim,i,j)]=r_temp[('lower',i,j)]
                r_perceived[(Idim+1,i,j)]=r_temp[('lower',i,j)]
                r_perceived[(Idim+2,i,j)]=r_temp[('lower',i,j)]
            else:
                r_perceived[(Idim,i,j)]=r_temp[('upper',i,j)]  
                r_perceived[(Idim+1,i,j)]=r_temp[('lower',i,j)]  
                r_perceived[(Idim+2,i,j)]=random.choice([r_temp[('upper',i,j)], r_temp[('lower',i,j)]])
        Idim+=3     

    return (r_perceived, Idim)


def create_uncertain_asymmetric_information_shortest_path_interdiction_nx(G,r_perceived,s,t,Idim,B):

    M=ConcreteModel()
    if len(set(G.edges))*Idim != len(r_perceived):
        raise Exception('r_perceived is not appropriately dimensioned')
    #Check if s and t are in the set of nodes
    if s not in list(G.nodes):
        raise Exception('{s} is not in the provided set of nodes')
    if t not in list(G.nodes):
        raise Exception('{t} is not in the provided set of nodes') 
    
    M.x=Var(set(G.edges),within=Binary) #whether or not an edge is interdicted
    M.budget=Constraint(expr = sum(M.x[(u,v)] for (u,v) in set(G.edges)) <= B)
    M.m=Var(within=Reals)
    M.Obj=Objective(expr=M.m,sense=maximize)
    
    M.IBlocks=Block(set(range(0,Idim)))
    for I in set(range(0,Idim)):
        M.IBlocks[I].d=Var(set(G.nodes),within=PositiveReals) #distance from s to a node
        M.IBlocks[I].w=Var(set(G.edges),within=NonNegativeReals)
        M.IBlocks[I].z=Var(set(G.edges),within=NonNegativeReals)
       
        
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.IBlocks[I].z[(i,j)] + (l+r)*M.IBlocks[I].w[(i,j)]
        M.IBlocks[I].ObjConstraint=Constraint(expr= value >= M.m)
        
        M.IBlocks[I].C=ConstraintList()
        for (i,j) in set(G.edges):
            M.IBlocks[I].C.add(M.IBlocks[I].d[j] <= M.IBlocks[I].d[i] + G[i][j]['perceived_length'] + M.x[(i,j)]* r_perceived[(I,i,j)])
            M.IBlocks[I].C.add(M.IBlocks[I].z[(i,j)] + M.x[(i,j)] <=1)
            M.IBlocks[I].C.add(M.IBlocks[I].w[(i,j)] - M.x[(i,j)] <= 0)
        M.IBlocks[I].c0=Constraint(expr = M.IBlocks[I].d[s]==0) #Distance-based shortest path
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['perceived_length']
            r=r_perceived[(I,i,j)]
            value=value+ l*M.IBlocks[I].z[(i,j)] + (l+r)*M.IBlocks[I].w[(i,j)]
        M.IBlocks[I].c5=Constraint(expr=M.IBlocks[I].d[t]==value)
        
        M.IBlocks[I].c6=ConstraintList()
        for J in set(G.nodes):
            if J==s:
                rhs=1
            elif J==t:
                rhs=-1
            else:
                rhs=0
            RS=0
            FS=0
            for (i,j) in set(G.out_edges(J)):
                FS= FS + M.IBlocks[I].w[(i,j)] + M.IBlocks[I].z[(i,j)]    
            for (i,j) in set(G.in_edges(J)):
                RS= RS + M.IBlocks[I].w[(i,j)] + M.IBlocks[I].z[(i,j)]
            M.IBlocks[I].c6.add(FS-RS == rhs )
    return M

def create_SPI_UAI_subproblem(G,R,x,s,t):
    
    M=ConcreteModel()
    
    def _bounds_rule(M, i, j):
        r_max=max(list(R[(i,j)]))
        return (0, r_max)
    
    x_keys=[]
    
    for (i,j) in set(G.edges):
        if x[(i,j)] == 1:
            x_keys.append((i,j))
    M.r = Var(set(G.edges), bounds=_bounds_rule)
    #M.r=Var(set(G.edges),within=NonNegativeReals)
    M.w=Var(set(G.edges), bounds=(0,1))
    M.z=Var(set(G.edges), bounds=(0,1))
    value=0
    for (i,j) in G.edges:
        value=value+G[i][j]['length']+max(list(R[(i,j)]))
    M.d=Var(set(G.nodes), bounds=(0,2*value))
    
    def Obj(M):
        value=0
        for (i,j) in set(G.edges):
            l=G[i][j]['length']
            r=G[i][j]['interdicted_length']
            value=value + l*M.z[(i,j)] + (l+r)*M.w[(i,j)]
        return value
    M.Obj=Objective(rule=Obj, sense=minimize)    
    
    def C(M,i,j):
        return M.d[j] <= M.d[i] + G[i][j]['perceived_length'] + x[(i,j)]*M.r[(i,j)]
    M.c=Constraint(set(G.edges),rule=C) #Distance-based shortest path
    M.c0=Constraint(expr = M.d[s]==0) #Distance-based shortest path
    
    def c3(M,i,j):
        return M.z[(i,j)] + x[(i,j)] <=1 
    M.c3=Constraint(set(G.edges),rule=c3) #zk + xk <= 1
    def c4(M,i,j):
        return M.w[(i,j)] - x[(i,j)] <= 0
    M.c4=Constraint(set(G.edges),rule=c4) #wk - xk <= 0
    
    
    def c6(M,J):
        if J==s:
            rhs=1
        elif J==t:
            rhs=-1
        else:
            rhs=0
        RS=0
        FS=0 
        for (i,j) in set(G.out_edges(J)):
            FS= FS + M.w[(i,j)] + M.z[(i,j)]    
        for (i,j) in set(G.in_edges(J)):
            RS= RS + M.w[(i,j)] + M.z[(i,j)]
        return FS-RS == rhs 
    M.c6=Constraint(set(G.nodes),rule=c6)
    
    #ONLY DO r as a variable if x[(i,j)=1]
    M.set_r=ConstraintList()
    for (i,j) in set(G.edges):
        if (i,j) not in x_keys:
            (l,h)=R[(i,j)]
            M.set_r.add(expr=M.r[(i,j)]==random.choice([l,h]))
       
    
    #R IS FROM INSTANCES
    M.Disjunction=Block(set(x_keys))
    for (i,j) in set(x_keys):
        r_list=R[(i,j)]
        M.Disjunction[(i,j)].rw=Var(bounds=(0,max(r_list)))
        M.Disjunction[(i,j)].RBlocks=Disjunct(range(0,len(r_list)))
        r0=0
        for r in r_list:
            M.Disjunction[(i,j)].RBlocks[r0].c1=Constraint(expr = M.r[(i,j)] == r)
            M.Disjunction[(i,j)].RBlocks[r0].c2=Constraint(expr = M.Disjunction[(i,j)].rw==r*M.w[(i,j)] )
            r0=r0+1
        M.Disjunction[(i,j)].Disj=Disjunction(expr = [M.Disjunction[(i,j)].RBlocks[r1] for r1 in range(0,len(r_list))])
    
    
    value=0
    for (i0,j0) in set(G.edges):
        l=G[i0][j0]['perceived_length']
        if (i0,j0) in set(x_keys):
            value=value+ l*M.z[(i0,j0)] + (l)*M.w[(i0,j0)] + M.Disjunction[(i0,j0)].rw
        else:
            value=value+ l*M.z[(i0,j0)]
        
    M.c5=Constraint(expr = value == M.d[t])
    
    TransformationFactory('mpec.simple_disjunction').apply_to(M)
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(M)
    return M   