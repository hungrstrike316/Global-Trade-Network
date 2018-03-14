import numpy as np
import pandas as pd
import sys
import networkx as nx
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
from utils.network_manipulation import modularity



dirPre = dm.set_dir_tree()
dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')

adj = dm.load_adjacency_npz_year(dirIn, 1969, 263)
adj_mtrx = adj[0]
adj_graph = nx.from_numpy_matrix(adj_mtrx, create_using=nx.DiGraph())

nodelist = list(adj_graph)

A = nx.to_scipy_sparse_matrix(adj_graph, nodelist=nodelist, weight='weight', format='csr')


A_prime = nx.to_numpy_matrix(adj_graph)

if (A.todense() == adj_mtrx).all():
    print('Scipy_Sparse is equal to our npz')
else:
    print('nx.to_scipy_sparce_matrix is different than npz adj')

"""
Key note for nx.directed_modularity_matrix use.  This uses the nx.to_scipy_sparce_matrix call, which creates a csr matrix.
To preserve the adjacency matrix weights, we need to set nodelist and weight parameters as seen below.
nodelist is just list(G) and weight is the string 'weight.'

"""
nx_mtrx = nx.directed_modularity_matrix(adj_graph, nodelist=nodelist, weight='weight')
# test_mod = modularity(np.array(A_prime).reshape(261,261))
# test_2 = modularity(np.array(A.todense()).reshape(261,261))
our_mod = modularity(adj_mtrx)
#
""" Testing scripts for modularities on different matrix types {'csr', 'np ndarray'}
"""
# #
# print("nx_mod - our_mod",np.subtract(nx_mtrx, our_mod))
# print('nx_mod =', mod_mtrx )
# # mod_diff = np.subtract(test_mod, our_mod)
# mod_diff2 = np.subtract(test_2, our_mod)
# print('mod diff2 = ', sum(mod_diff2))
# print('mod diff= ', sum(mod_diff))
# if (mod_diff == np.zeros(our_mod.shape)).all():
#     print('computed mod with scipy = nx func')
# else:
#     print("nx mod still different than computed")
#
# diff_2 = adj_mtrx - A_prime
# diff = adj_mtrx - A.todense()
#
#
# print("npz adj - nx.scipy = ", diff)
# print("npz - nx.numpy = ", diff_2)
# print("Adjacency npz = ", adj_mtrx)
# print("nx.to_numpy = ", A_prime  )
# print("nx.Scipy_Sparse = ", A)
#
in_degree = np.sum(adj_mtrx, axis=0)
out_degree = np.sum(adj_mtrx, axis=1)
# print(out_degree.shape)
# B = adj_mtrx - (np.multiply.outer(in_degree, out_degree.reshape(1,261))) / np.sum(adj_mtrx)
B = adj_mtrx - (in_degree * out_degree.reshape(261,1)) / np.sum(adj_mtrx)

"""  Important to reshape out_degree, and use np matrix multiplication NOT multiply.outer.
multipy.outer produces a 3d array, not a 2d array. """
# if return_symmetric:
#     return B + B.T


print(B.shape)




""" Following code is to examine nx implementation and compare it to our computation.
B is our modularity matrix, N and all other following variables are for the nx implementation
on the nx sparse matrix.  """
#
#
k_in = A.sum(axis=0)

# print('k_in - in_deg ', k_in - in_degree)
k_out = A.sum(axis=1)
# print('k_out - out_degree ', k_out - out_degree.reshape(261,1))
m = k_in.sum()
# Expected adjacency matrix
X = k_out * k_in / m
N =  A - X
print(N.shape)
print(B - N)
