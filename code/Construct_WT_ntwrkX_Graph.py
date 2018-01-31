import numpy as np
import pandas as pd
import networkx as nx
import pickle
import utils.data_manipulation as dm
import utils.network_manipulation as nm

# This script Construct directed network Graph object that NetworkX can analyze from the Adjacency matrix
# loaded from an .npz file. It will save the graph object as a .gpickle file. It will also save output 
# figures of the Adjacency Matrix, and the network overlayed on a map.

#------------------------------------------------------------------------------------------------------------
## (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
dirPre = dm.set_dir_tree()

#------------------------------------------------------------------------------------------------------------
## (1). Load country names that align with 3 letter acronyms used in origin destination file
countries = dm.load_countries(dirPre)
num_countries = np.size(countries,0)

## (2) Load in names and codes for types of goods traded
#goods = dm.load_products(dirPre)

### Loop over each year
years = np.concatenate( (range(1962,1967), range(1969,2015) )  , axis=0)

a=num_countries          # number of countries (other 2 are 'world' and 'areas')

# For each year: Each trade network from years 1962 - 2014.
for y in years:
	print(str(y))

	# load in adjacency for a given year.
	dirIn = str( dirPre + 'adjacency_ntwrk_npz_files/' )
	trade_ntwrkA, imports, exports = dm.load_adjacency_npz_year(dirIn, y, num_countries)

	# Construct a networkX graph containing node and edge attributes.
	lat_lon_in = str(dirPre + 'MIT_WT_datafiles/cntry_lat_lon_combined_fin_UTF8.pickle') # must be a pickle file.
	dirOut_ntwrkX = str(dirPre + 'adjacency_ntwrkX_pickle_files/') # directory to save networkX pickle file of Trade Net 
	trade_ntwrkG = nm.construct_ntwrkX_Graph( lat_lon_in, trade_ntwrkA, imports, exports, dirOut_ntwrkX, y )
