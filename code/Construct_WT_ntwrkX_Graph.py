import numpy as np
import pandas as pd
import networkx as nx
import pickle
import utils.data_manipulation as dm
import utils.network_manipulation as nm

# This script Construct directed network Graph object that NetworkX can analyze from the Adjacency matrix
# loaded from an .npz file. It will save the graph object as a .gpickle file. It will also save output 
# figures of the Adjacency Matrix, and the network overlayed on a map.

sym = 'sym' # 'sym' if undirected or '' if directed.


#------------------------------------------------------------------------------------------------------------
## (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
dirPre = dm.set_dir_tree()

#------------------------------------------------------------------------------------------------------------
## (1). Load country names that align with 3 letter acronyms used in origin destination file
countries = dm.load_country_lat_lon_csv(dirPre)
num_countries = countries.shape[0]

### Loop over each year
years = range(1962,2015)
# For each year: Each trade network from years 1962 - 2014.
for year in years:
	print(str(year))

	# Construct a networkX graph containing node and edge attributes.
	trade_ntwrkG = nm.construct_ntwrkX_Graph( dirPre, year, flg_sym )	