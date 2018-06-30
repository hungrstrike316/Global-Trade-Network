import numpy as np
import pandas as pd
import sys
import networkx as nx
import matplotlib.pyplot as plt
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.network_manipulation as nm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

dirPre = dm.set_dir_tree()
dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
sym = 'sym' # 'sym' if undirected or '' if directed.
weight = 'weight' # 'weight' if weighted graph or None if unweighted graph
norm = '' # 'norm' if norm, '' if not.

for year in range(1962,1963):

	## (-) First the adjacency matrix is loaded from the adj_npz.
	# Then the adj_matrix is converted into a NetworkX DiGraph object.
	# Finally the DiGraph is used to create a laplacian matrix, using the  NetworkX
	# laplacian_matrix functions, selected in our network manipulation suit.
	adj_npz, _, _ = dm.load_adjacency_npz_year(dirIn, year, num_countries, sym)

    # For laplacian
	adj_graph = nm.construct_ntwrkX_Graph(dirPre, year, sym)
	lap_ntwkX = nm.networkX_laplacian(adj_graph,sym,norm,weight)

    ## (-) Flags for which method we're testing.
    #  Implement for all methods later.
    if adjacency:
        trade_ntwrk = adj_npz
        if sym == 'sym':
            assert np.all(trade_ntwrk - trade_ntwrk.T == 0), "Undirected Adjacency not symmetric!"

    if laplacian:
        trade_ntwrk = lap_ntwkX

	Ui, Si, svd_Vi = np.linalg.svd( trade_ntwrk, full_matrices=True, compute_uv=True )
	Wi, Vi = np.linalg.eig(trade_ntwrk)


    # View Ui - Vi (from eig)
    if False:
        plt.imshow(Ui - Vi)
        plt.colorbar()
        plt.show()

    #  View Ui - svd_Vi.T
    if False:
        plt.imshow(Ui - svd_Vi.T)
        plt.colorbar()
        plt.show()

    # Check that Wi and Si are same
    if False:
        equal = np.all(Wi - Si == 0)
        print('Eigenvalues are equal!' if equal else "Eigenvalues are not equal!")
