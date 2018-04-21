import numpy as np
import utils.network_manipulation as nm

for i in range(1962,1963): # 2015):
    try:
        adj = np.load("../adjacency_ntwrk_npz_files/adjacency_ntwrk_" + str(i) +
                    "_261countries.npz")

        B = nm.modularity(adj['netwrk'].astype(bool))

        np.savez("../modularity_ntwrk_npz_files/modularity_ntwrk_" + str(i) +
                	"_261countries.npz", B)
    except FileNotFoundError:
        print('uuh')
        pass
