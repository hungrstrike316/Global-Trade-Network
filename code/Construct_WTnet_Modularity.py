import numpy as np
from utils.network_manipulation import modularity

for i in range(1962, 2015):
    try:
        adj = np.load("../adjacency_ntwrk_npz_files/adjacency_ntwrk_" + str(i) +
                      "_263countries.npz")
        B = modularity(adj['netwrk'])
        np.savez("../modularity_ntwrk_npz_files/modularity_ntwrk_" + str(i) +
                 "_263countries.npz", B)
    except FileNotFoundError:
        pass
