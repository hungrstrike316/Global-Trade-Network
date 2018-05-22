import pickle
import numpy as np
import networkx as nx
from collections import Counter
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import community as c 					# python-louvain module
import utils.plot_functions as pf
import utils.data_manipulation as dm
import utils.network_manipulation as nm

# from ast import literal_eval # to evaluate what is in a string literally.

flg_plot_graph_on_map = True
flg_imshow_adjacency  = False
res = 1
getcontext().prec = 3  # later used for adjusting alpha in scatter plot

#LOOK UP: from VS260C Rowland Taylor Lecture.
# anishenko 2010 j neurophys
# baden Nature 2016 529. imaging retina


# This script Construct directed network Graph object that NetworkX can analyze from the Adjacency matrix
# loaded from an .npz file. It will save the graph object as a .gpickle file. It will also save output
# figures of the Adjacency Matrix, and the network overlayed on a map.

#------------------------------------------------------------------------------------------------------------
## (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
dirPre = dm.set_dir_tree()

for file_year in np.arange(1962, 1963):
	try:
	    year = np.array(file_year)
	    flg_sym = True
	    trade_ntwrkG = nm.construct_ntwrkX_Graph(dirPre=dirPre, year=year, flg_sym=flg_sym)
	    # giant = max(nx.connected_component_subgraphs(G), key=len)
	    # out_deg = len(list(G)) - len(giant)
	except:
	    pass


	part = c.best_partition(trade_ntwrkG, partition=None, weight='weight', resolution=res, randomize=False)
	res_counts = Counter(part.values())
	res_order = sorted(res_counts, key=lambda i: int(res_counts[i]))
	partition_sizes = Counter(res_counts.values())
	lg_clust = [key for key, val in res_counts.items() if val == max(list(res_counts.values()))][0]

	clust_node_dict = {str(res_order[-idx]):[key for key,val in part.items() if
						val == res_order[-idx]] for idx in np.arange(1,len(res_order))
						if res_counts[res_order[-idx]] > 1}

	ones_clusters = set([key for key,val in res_counts.items() if val == 1])
	ones_nodes = set([key for key,val in part.items() if val in ones_clusters])
	# extract node attributes for plotting
	LatLon = nx.get_node_attributes(trade_ntwrkG,'LatLon')            # these output dicts containing tuples of [(key, value)].
	# continent = nx.get_node_attributes(trade_ntwrkG,'continent')
	# countryId3 = nx.get_node_attributes(trade_ntwrkG,'countryId3')
	# countryName = nx.get_node_attributes(trade_ntwrkG,'countryName')
	# exports = nx.get_node_attributes(trade_ntwrkG,'exports')
	# imports = nx.get_node_attributes(trade_ntwrkG,'imports')

	num_clusters = len(partition_sizes)
	ones_LatLon = np.array([val for key, val in LatLon.items() if key in ones_nodes])
	clust_LatLon = {k:np.array([val for key, val in LatLon.items() if key in v]) for k,v in clust_node_dict.items()}

	inc = Decimal(0.2)
	alpha = Decimal(0.9)
	alphas = {}
	for idx in res_order[:-num_clusters:-1]:
		alphas[str(idx)] = float(alpha)
		alpha -= inc

# #------------------------------------------------------------------------------------------------------------
# # (5). Plot and save figure of graph in networkX (with nodes in relative geographic location,
# #      labels for names, importd & exports for size.
	figG = plt.figure(figsize=(15,9))
	map1 = Basemap(projection='cyl')
	pf.draw_map()

	for key,val in clust_LatLon.items():
		map1.scatter(val[:,0], val[:,1], marker= 'o', color='r', alpha=alphas[key])
	map1.scatter(ones_LatLon[:,0], ones_LatLon[:,1], marker='o', color='grey')
	plt.title("GTN Clustering Map " + str(file_year))
	plt.xlim(-180, +180)
	plt.ylim(-90,+90)
	plt.show()



	#
	figG.savefig(str( '../out_figures/GTN_clust_maps/' + str(file_year) + '_clust_map.png' ), bbox_inches='tight')

	plt.close(figG)
	#
	#
