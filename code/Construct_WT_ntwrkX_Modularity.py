
# This function takes npz files of adjacency matrices, converts them into a graph object, then creates
# a modularity matrix of the adjacency.
#
# Documentation on the NetworkX modularity_matrix module can be found here:

#  (1). https://pypkg.com/pypi/networkx/f/networkx/linalg/modularitymatrix.py



# Import a bunch of python packages that I use below.
import numpy as np
import pandas as pd
import sys
import networkx as nx
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions


# Define the function to construct a trade network adjacency matrix for single year. Run this in a for loop below
def Construct_WT_ntwrkX_Modularity(year):

	dirPre = dm.set_dir_tree()

	## (1) Load country names that align with 3 letter acronyms used in origin destination file
	countriesLL = dm.load_country_lat_lon_csv(dirPre)
	num_countries = countriesLL.shape[0]


	# (2) Obtain accurate directory locations for both input and output files.

	dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
	dirOut = str(dirPre + 'modularity_ntwrkX_npz_files/')


	## (4) First the adjacency matrix is loaded from the adj_npz.
	# Then the adj_matrix is converted into a NetworkX DiGraph object.
	# Finally the DiGraph is used to create a modularity matrix, using the built in NetworkX
	# modularity_matrix function.
	adj_npz = dm.load_adjacency_npz_year(dirIn, year, num_countries)
	adj_graph = nx.from_numpy_matrix(adj_npz[0], create_using=nx.DiGraph())
	mod_mtrx = nx.directed_modularity_matrix(adj_graph)

	np.savez(str(dirOut + 'modularity_ntwrkX_' + str(year) + '_' + str(num_countries) + 'countries.npz'),
		netwrk = mod_mtrx)

	return mod_mtrx


# This bit here makes it so you can call this as a function from the command line with the year as an input argument.
# Call this from the command line like:
# 	>> python3 Construct_WTnet_Adjacency.py 1964
# On cluster you may have to load python module first, like:
# 	>> module load python/anaconda3
if __name__ == "__main__":
	Construct_WT_ntwrkX_Modularity(sys.argv[1]) # this counts on the input being only 4 characters, a year - like '1984'.
	#Construct_WTnet_Modularity(sys.argv[1:4]) # this counts on the input being only 4 characters, a year - like '1984'.
