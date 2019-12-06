#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:14:03 2019

@author: sreekuk1
"""
#Libraries required
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import pca
import scipy.sparse.linalg as la
from scipy.sparse import csr_matrix 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from collections import Counter
import scipy.sparse as sps
import scipy
import sys


#reading graph file into csv file- contains only edges
f=open("roadNet-CA.txt")
graph=np.array(pd.read_csv(f,sep=' ',header=None,skiprows=1))

#reading the number of vertices,edges and partitions required
with open("roadNet-CA.txt") as f: 
     first_line = f.readline()
graph_key=[int(s) for s in first_line.split() if s.isdigit()]
 
n_vert=graph_key[0]
n_edge=graph_key[1]
n_partition=graph_key[2]
#Constructing graph
G=nx.Graph()
G.add_nodes_from(range(n_vert))
G.add_edges_from(graph)

#Computing laplacian 
L=nx.laplacian_matrix(G)
L=csr_matrix.astype(L,dtype='f')

# Computing eigen vectors
eig_val,eig_vec = la.eigsh(L,2,which='SM') 
# Computing k-means
km = KMeans(n_clusters=n_partition, init='k-means++', max_iter=1000, n_init=25)
#km=KMeans(n_clusters=n_partition,max_iter=5000,init='kmeans++', random_state=0)
km.fit(eig_vec)
#objective function
cluster=[]
cluster.append(km.labels_)
cluster=list(cluster[0])
denom=Counter(km.labels_).values()


phi=0
a=[0.0]*n_partition
phi=0

for node in range(n_vert):
    for n in G.neighbors(node):
        if(cluster[n]!=cluster[node]):
            a[cluster[node]]+=1
                
phi=sum(np.array(a)/np.array(denom))
    
def save_in_file(f_name,clusters,first_line):
    f= open(f_name+".txt","w+")
    f.write(first_line)
    for i in range(len(clusters)):
        str_out = ""+str(i)+" "+str(clusters[i])+"\n"
        f.write(str_out)     

save_in_file("roadNet-CA.txt",cluster,first_line)






