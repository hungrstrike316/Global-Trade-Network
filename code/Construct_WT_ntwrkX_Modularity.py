
# This function takes npz files of adjacency matrices, converts them into a graph object, then creates
# a modularity matrix of the adjacency.



# Import a bunch of python packages that I use below.
import numpy as np
import pandas as pd
import sys
import networkx as nx
import matplotlib.pyplot as plt
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.network_manipulation as nm

sym = '' # 'sym' if undirected or '' if directed.
weight = 'weight' # 'weight' if weighted graph or None if unweighted graph

# (-) Obtain accurate directory locations for both input and output files.
dirPre = dm.set_dir_tree()
dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
dirOut = str(dirPre + 'modularity_ntwrkX_npz_files/')


## (-) Load country names that align with 3 letter acronyms used in origin destination file
countriesLL = dm.load_country_lat_lon_csv(dirPre)
num_countries = countriesLL.shape[0]


# (-) Loop through different years and compute modularity using NetworkX 
for year in range(1962,2015):

	## (-) First the adjacency matrix is loaded from the adj_npz.
	# Then the adj_matrix is converted into a NetworkX DiGraph object.
	# Finally the DiGraph is used to create a modularity matrix, using the built in NetworkX
	# modularity_matrix function.
	adj_npz, _, _ = dm.load_adjacency_npz_year(dirIn, year, num_countries, sym)
	adj_graph = nm.construct_ntwrkX_Graph(dirPre, year, sym)
	mod_ntwkX = nm.networkX_modularity(adj_graph,sym,weight)


	# (-) Uncomment these lines to check if Our implemented modularity matches results of NetworkX version.
	dirMod = str(dirPre + 'modularity_ntwrk_npz_files/')
	mod_npz = dm.load_modularity_npz_year(dirMod, year, num_countries, sym)
	diff = mod_ntwkX - mod_npz
	claim = (diff).any()
	print('Claim differences in 2 modularity calcs.', claim )
	if claim:
		plt.imshow(diff)
		plt.colorbar()
		plt.show()

	np.savez(str(dirOut + sym + 'modularity_ntwrkX_' + str(year) + '_' + str(num_countries) + 'countries.npz'),
		netwrk = mod_ntwkX)