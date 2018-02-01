import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import utils.plot_functions as pf 
import utils.data_manipulation as dm
import utils.network_manipulation as nm



"""

import pandas as pd

import time
import os
import csv

import sys
"""


#LOOK UP: from VS260C Rowland Taylor Lecture.
# anishenko 2010 j neurophys 
# baden Nature 2016 529. imaging retina


# This script Construct directed network Graph object that NetworkX can analyze from the Adjacency matrix
# loaded from an .npz file. It will save the graph object as a .gpickle file. It will also save output 
# figures of the Adjacency Matrix, and the network overlayed on a map.

#------------------------------------------------------------------------------------------------------------
## (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
dirPre = dm.set_dir_tree()


### Loop over each year - # For each year: Each trade network from years 1962 - 2014.
years =  np.concatenate( (range(1962,1967), range(1969,2015) ), axis=0) # np.array([1965]) #
for y in years:
	print(str(y))

	trade_ntwrkG = nx.read_gpickle( str( dirPre + 'adjacency_ntwrkX_pickle_files/trade_ntwrkX_'+ str(y) + '.gpickle' ) )

	# extrace node attributes for plotting
	LatLon = nx.get_node_attributes(trade_ntwrkG,'LatLon')            # these output dicts containing tuples of [(key, value)].
	continent = nx.get_node_attributes(trade_ntwrkG,'continent')
	countryId3 = nx.get_node_attributes(trade_ntwrkG,'countryId3')
	countryName = nx.get_node_attributes(trade_ntwrkG,'countryName')
	exports = nx.get_node_attributes(trade_ntwrkG,'exports')
	imports = nx.get_node_attributes(trade_ntwrkG,'imports')

	num_countries = len(trade_ntwrkG.nodes())

	# Construct an array to color nodes by their continent affiliation
	node_colors_by_continent = np.zeros(num_countries)
	conts = set(continent.values())
	cntr=0
	for each in conts:
		idx = [key for key, val in continent.items() if val==each] # list <- dict
		idx = np.array(idx).astype(int)                            # int array <- list
		node_colors_by_continent[idx] = cntr
		cntr=cntr+1

	# list comprehensions to extract node attributes from dicts.
	LatLon = [val for key, val in LatLon.items()] 
	exports = [val for key, val in exports.items()]
	imports = [val for key, val in imports.items()]
	#continent = [val for key, val in continent.items()] 
	#countryId3 = [val for key, val in countryId3.items()]
	#countryName = [val for key, val in countryName.items()]


	#------------------------------------------------------------------------------------------------------------			
	# (5). Plot and save figure of graph in networkX (with nodes in relative geographic location, 
	#      labels for names, importd & exports for size.
	#
	if(False):
		#%matplotlib inline
		figG = plt.figure(figsize=(15,9))

		nSize = exports     # could also be imports OR mean(imports,exports)
		nSize = 1000*(nSize / np.max(nSize))

		# Set up vector of log of relative size of edges for colormap and edge thickness in trade map plot
		nWidth = [(trade_ntwrkG[u][v]["weight"]) for u,v in trade_ntwrkG.edges()]
		nWidth = np.asarray(nWidth)
		nz = np.rot90(np.nonzero(nWidth)) # a tuple of x & y location of each nonzero value in Adjacency matrix
		low = np.min(nWidth[nz])
		high = np.max(nWidth)
		nWidth[nz] = np.log(nWidth[nz] - low + 1.1) # subtract off min value and
		nWidth = nWidth/np.max(nWidth)    # take log of normalized width of edges

		#
		edgeG = nx.draw_networkx_edges(trade_ntwrkG, pos=LatLon, arrows=False, font_weight='bold', alpha=0.2, width=3*nWidth,
			edge_color=nWidth, edge_cmap=plt.cm.jet) # edge_vmin=np.min(nWidth), edge_vmax=np.max(nWidth),
		#
		nodeG = nx.draw_networkx_nodes(trade_ntwrkG, pos=LatLon, node_size=nSize, cmap=plt.cm.Dark2, with_labels=True,
								font_weight='bold', node_color=node_colors_by_continent)
		#
		labelG = nx.draw_networkx_labels(trade_ntwrkG, pos=LatLon, labels=countryId3, font_size=12)
		#
		pf.draw_map() # using BaseMap package.
		#
		pf.plot_labels(figG, str( 'Global Trade Map : ' + str(y) ), 'Longitude', 'Latitude', 20) # title, xlabel, ylabel and plot ticks formatting
		#
		plt.xlim(-180, +180)
		plt.ylim(-90,+90)
		plt.grid(linewidth=2)
		#
		sm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
		sm._A = []
		cbar = plt.colorbar(sm, ticks=[0, 1])
		cbar.ax.set_yticklabels([pf.order_mag(low), pf.order_mag(high)]) 

		#plt.legend(trade_ntwrkG.nodes())

		figG.savefig(str( '../out_figures/trade_maps/' + str(y) + '_trade_map.png' ), bbox_inches='tight')

		#plt.show()
		plt.close(figG)




	#------------------------------------------------------------------------------------------------------------
	# (6). Imshow and save figure of Adjacency Matrix - another way to visualize changes in network
	#
	#
	if(False):

		trade_ntwrkA, imports, exports = dm.load_adjacency_npz_year(str( dirPre + 'adjacency_ntwrk_npz_files/'), y, num_countries+2)
		#trade_ntwrkA = out[0]


		figA = plt.figure(figsize=(15,15))
		plt.imshow(np.log10(trade_ntwrkA))
		plt.colorbar()

		pf.plot_labels(figA, str( 'Global Trade Adjacency : ' + str(y) ), 'Country', 'Country', 20) # title, xlabel, ylabel and plot ticks formatting

		figA.savefig(str( dirPre + 'out_figures/adjacency_mats/' + str(y) + '_adj_mat.png' ))

		#plt.show()
		plt.close(figA)
