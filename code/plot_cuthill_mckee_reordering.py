

# (0). Import packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import csv
import networkx as nx
import sys
from sklearn.cluster import KMeans
import scipy.sparse as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian



import utils.data_manipulation as dm
import utils.network_manipulation as nm
import utils.plot_functions as pf




# IF RUNNING AS A FUNCTION, I NEED TO INPUT.
#
# years, method {A, Ln, Q}, 
#
# What is happening inside?
#
# (1). Reorder Matrix with Cuthill-Mckee algorithm to get clusters.
# (2).
#


method = 'Adjacency'
#method = 'Normalized Laplacian'
#method = 'Modularity'
#method = 'Topographic Modularity'

print(method)


#------------------------------------------------------------------------------------------------------------
## (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
dirPre = dm.set_dir_tree()

#------------------------------------------------------------------------------------------------------------
## (1). Load country names that align with 3 letter acronyms used in origin destination file
countries = dm.load_countries(dirPre)
num_countries = np.size(countries,0)-2 # get rid of 'world' and 'areas'




## (2) Load in names and codes for types of goods traded
#goods = dm.load_products(dirPre)  




# (4). Loop through, load and plot all previously saved adjacency matrix files.  
years = np.concatenate( (range(1962,1967),range(1969,2015)), axis=0) # np.array([1962]) # years for which we have world trade data.
#a1=263 # number of 'countries' in the data (but has 2 extra)
#b=2 # get rid of 'world' and 'areas' from trade network

for y in years:
	print(y)

	dirIn = str( dirPre + 'adjacency_ntwrk_npz_files/' )
	trade_ntwrk, imports, exports = dm.load_adjacency_npz_year(dirIn, y, num_countries+2)

	# (4). Check that Imports are just the column sums and Exports are row sums
	assert np.any( np.sum(trade_ntwrk,axis=0) == imports) , 'Imports are Weird'
	assert np.any( np.sum(trade_ntwrk,axis=1) == exports) , 'Exports are Weird'
	#print('Everything checks out with sums, imports and exports.')


	# (5c). Compute Normalized Laplacian (For Asymmetric, use_out_degree can use imports or exports)
	trade_ntwrk = nm.construct_ntwrk_method(trade_ntwrk,method)

	# (5b). Find Cuthill-McKee reordering of Adjacency Matrix (requires sparse matrices and scipy's sparse library).
	# Q: Does it make sense to do this on Laplacian & Modularity? How do things change?
	if True:
		perm = nm.cuthill_mckee(trade_ntwrk)
		ind1 = np.broadcast_to(perm, (np.size(perm),np.size(perm)) )
		ind2 = ind1.transpose()

		# (5c). Plot Cuthill-Mckee reordering of Adjacency Matrix. 
		if True:
			fig = plt.figure(figsize=(15,15))
		    
		    # Plot Adjacency
			plt.subplot(2,2,1)
			plt.imshow( np.log10(trade_ntwrk), interpolation='none' )
			plt.title(method)
			#plt.colorbar(fraction=0.046, pad=0.04)
			sm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
			sm._A = []
			cbar = plt.colorbar(sm, ticks=[0, 1])
			cbar.ax.set_yticklabels([pf.order_mag(np.min(trade_ntwrk)), pf.order_mag(np.max(trade_ntwrk))]) 
		    
		    
		    # Plot reordered Adjacency
			plt.subplot(2,2,2)
			plt.imshow( np.log10(trade_ntwrk[(ind1,ind2)]), interpolation='none' )
			plt.title("Cuthill-Mckee Reordering")
			#plt.colorbar(fraction=0.046, pad=0.04)

			sm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
			sm._A = []
			cbar = plt.colorbar(sm, ticks=[0, 1])
			cbar.ax.set_yticklabels([pf.order_mag(np.min(trade_ntwrk)), pf.order_mag(np.max(trade_ntwrk))]) 
		    
			#print("Reordered Countries")    
			#print(countries.name[perm[0:40]])
		    
		    # Plot Imports and Exports reordered by Cuthill-Mckee algorithm
			plt.subplot(2,2,3)
			plt.plot(imports)
			plt.plot(exports)
			plt.title("Total Imports and Exports")
		    #
			plt.subplot(2,2,4)
			plt.plot(imports[perm])
			plt.plot(exports[perm])
			plt.legend( ('imports','exports') )

			fig.savefig(str( dirPre + 'out_figures/cuthill_mckee_reordering/' + method + '_' + str(y) + '_cuthill_mckee.png' ), bbox_inches='tight')
			#plt.show()
			plt.close(fig)
