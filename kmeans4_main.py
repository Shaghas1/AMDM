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

def distance(p1, p2): 
    return np.sum((p1 - p2)**2) 

# initialisation algorithm K++
def initialize(data, k): 
    ''' 
    intialized the centroids for K-means++ 
    inputs: 
        data - numpy array of data points having shape (200, 2) 
        k - number of clusters  
    '''
    ## initialize the centroids list and add 
    ## a randomly selected data point to the list 
    centroids = [] 
    centroids.append(data[np.random.randint( 
            data.shape[0]), :]) 
   
    ## compute remaining k - 1 centroids 
    for c_id in range(k - 1): 
          
        ## initialize a list to store distances of data 
        ## points from nearest centroid 
        dist = [] 
        for i in range(data.shape[0]): 
            point = data[i, :] 
            d = sys.maxsize 
              
            ## compute distance of 'point' from each of the previously 
            ## selected centroid and store the minimum distance 
            for j in range(len(centroids)): 
                temp_dist = distance(point, centroids[j]) 
                d = min(d, temp_dist) 
            dist.append(d) 
              
        ## select data point with maximum distance as our next centroid 
        dist = np.array(dist) 
        next_centroid = data[np.argmax(dist), :] 
        centroids.append(next_centroid) 
        dist = [] 
    return centroids 


def k_means_modified4_plus(X, n_partition, num_iters):    
    
    
    dim = X.shape[1]
    n_data = X.shape[0]
    max_cluster_size = n_data/n_partition
    
    #Initialize centeriods using ++ method
    centroids = initialize(X, n_partition)
    
    #
    clusters = np.array([0]*n_data)
    
    k_means_errors=[]
    for it in range(num_iters):
        
        data_per_cluster = [0.0]*n_partition
        
        dist_to_cluster = np.zeros([n_data,n_partition])
        priority_cluster = np.zeros([n_data,n_partition])
        
        for i in range(n_data):      
            dist_to_cluster[i,:] = [(np.linalg.norm(X[i] - centroids[j])) for j in range(n_partition)]
            for j in range(n_partition):
                priority_cluster[i,j] = dist_to_cluster[i,j] - np.max(dist_to_cluster[i,:])
                
        priority = priority_cluster.min(1)                                         
        high_priority = np.argsort(priority)
            
        
        for i in range(n_data):
            curr_data = high_priority[i]
            data_cluser_priority = np.argsort(priority_cluster[curr_data,:])
            assigned_flag = False  
            for candidate in data_cluser_priority:
                if assigned_flag == False and data_per_cluster[candidate] < max_cluster_size:
                    clusters[curr_data] = candidate
                    data_per_cluster[candidate] += 1
                    assigned_flag = True 
                    continue

        
        #update means
        new_centroids = np.zeros((n_partition,dim))
        for i in range(n_partition):
            cluster_points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            if len(cluster_points) != 0:
                new_centroids[i] = np.mean(cluster_points, axis = 0)
        centroids = new_centroids

        #k_means_errors.append(obj_func(G,clusters))

    return centroids,clusters


    
def save_in_file(f_name,clusters,first_line):
    f= open(f_name+".txt","w+")
    f.write(first_line)
    for i in range(len(clusters)):
        str_out = ""+str(i)+" "+str(clusters[i])+"\n"
        f.write(str_out)   
        
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


centroids,clusters = k_means_modified4_plus(eig_vec, n_partition, 1000)
phi = obj_func(G,clusters)
save_in_file(f_name+"results",clusters,first_line)