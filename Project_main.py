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



def obj_func(graph,clusters):
    n_partition = max(clusters)+1
    n_vert = len(clusters)
    
    nodes_in_cluster = [0.0]*n_partition
    edge_out=[0.0]*n_partition #Array to compute number of edges moving out of a cluster to another
    
    for node in range(n_vert):
        nodes_in_cluster[clusters[node]] += 1
        for n in graph.neighbors(node):
                if(clusters[n]!=clusters[node]):
                    edge_out[clusters[node]]+=1
    #print("edge outs:", np.array(edge_out))
    phi = np.sum(np.array(edge_out)/np.array(nodes_in_cluster)) 
    
    return phi

def save_in_file(f_name,clusters,first_line):
    f= open(f_name+".txt","w+")
    f.write(first_line)
    for i in range(len(clusters)):
        str_out = ""+str(i)+" "+str(clusters[i])+"\n"
        f.write(str_out)     


#reading graph file into csv file- contains only edges
f_name = "ca-GrQc.txt"
f=open(f_name)
graph=np.array(pd.read_csv(f,sep=' ',header=None,skiprows=1))

#reading the number of vertices,edges and partitions required
with open(f_name) as f: 
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
cluster=km.labels_

    

        
phi = obj_func(G,cluster)
save_in_file(f_name,cluster,first_line)
print(phi)





