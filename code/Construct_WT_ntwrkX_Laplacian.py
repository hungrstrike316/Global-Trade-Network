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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

normalized = False
directed = False
flg_plot_1st3_Eigs = True
sym = 'sym' # 'sym' if undirected or '' if directed.
weight = 'weight' # 'weight' if weighted graph or None if unweighted graph
norm = ''
# (-) Obtain accurate directory locations for both input and output files.
dirPre = dm.set_dir_tree()
dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
dirOut = str(dirPre + 'laplacian_ntwrkX_npz_files/')
file_name = 'laplacian_ntwrkX_'
title = 'Laplacian'
if normalized:
	file_name = 'normalized_laplacian_ntwrkX_'
	title = 'Undirected Normalied Laplacian'
	norm = 'norm'
	sym = 'sym'
if directed:
	file_name ='directed_normalized_laplacian_'
	title = 'Directed Laplacian'
	norm = 'norm'
	sym = ''


## (-) Load country names that align with 3 letter acronyms used in origin destination file
countriesLL = dm.load_country_lat_lon_csv(dirPre)
num_countries = countriesLL.shape[0]



# (-). Get array indicating continents for each country to colorcode nodes in graph.
continent=[]
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
for o in range(0,num_countries):
	reg = countriesLL.id[o]
	continent = np.append(continent,reg[0:2])
conts = set(continent) # this is a 'set object' (all the different countries)
conts = list(conts) # convert 'set object' to a list that I can iterate over.
conts = np.sort(conts)
node_colors_by_continent = np.zeros(len(continent))
for i in range(0,len(conts)):
	node_colors_by_continent[ np.array(continent == conts[i]) ] = i

# (-) Loop through different years and compute modularity using NetworkX
for year in range(1962,1963):

	## (-) First the adjacency matrix is loaded from the adj_npz.
	# Then the adj_matrix is converted into a NetworkX DiGraph object.
	# Finally the DiGraph is used to create a laplacian matrix, using the  NetworkX
	# laplacian_matrix functions, selected in our network manipulation suit.
	adj_npz, _, _ = dm.load_adjacency_npz_year(dirIn, year, num_countries, sym)
	adj_graph = nm.construct_ntwrkX_Graph(dirPre, year, sym)
	lap_ntwkX = nm.networkX_laplacian(adj_graph,sym,norm,weight)


	# (-) Uncomment these lines to check if Our implemented modularity matches results of NetworkX version.
	# dirLap = str(dirPre + 'laplacian_ntwrk_npz_files/')
	# lap_npz = dm.load_laplacian_npz_year(dirLap, year, num_countries, sym)
	# diff = lap_ntwkX - lap_npz
	# claim = (diff).any()
	# print('Claim differences in 2 laplacian calcs.', claim )
	# if claim:
	# 	plt.imshow(diff)
	# 	plt.colorbar()
	# 	plt.show()
	if False:
		np.savez(str(dirOut + file_name + str(year) + '_' + str(num_countries) + 'countries.npz'),
			netwrk = lap_ntwkX)




	## (-) Compute eigenvalues, Wi, and eignvectors, Vi, of the laplacian matrix.

	Wi, Vi = np.linalg.eig(lap_ntwkX)
	# if directed:
		# Vi = Vi.A
	#
	# if flg_plot_1st3_Eigs:
	#
	# 	fig=plt.figure(figsize=(8,8))
	# 	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	#
	# 	for i in range(0,7):
	# 		indx = node_colors_by_continent==i
	# 		ax.scatter(Vi[0][indx], Vi[1][indx], Vi[2][indx], s=150, edgecolor='k', c=colors[i], alpha=0.4)
	# 	# ax.scatter(Vi[0], Vi[1], Vi[2]) #s=150, edgecolor='k', c=colors[0])# alpha=0.4)
	#
	#
	# 	lims=1
	# 	ax.set_xlim(-lims,lims)
	# 	ax.set_ylim(-lims,lims)
	# 	ax.set_zlim(-lims,lims)
	#
	# 	# gt=0.1
	# 	# for i,cname in countriesLL.name.iteritems():
	# 	#  i = np.int(i)
	# 	 # if i<num_countries:
	# 		#  if np.any([np.abs(Vi[0][i])>gt, np.abs(Vi[1][i])>gt, np.abs(Vi[2][i])>gt]):
	# 		# 	 #print( str( cname + " " + str(Vi[0][i])  + " " +  str(Vi[1][i]) ) + " " +  str(Vi[2][i]) )
	# 		# 	 ax.text( Vi[0][i]+0.03, Vi[1][i]+0.03, Vi[2][i]+0.03, cname )
	#
	# 	szfont = 24
	# 	ax.set_xlabel( '1st Eig', fontsize=szfont )
	# 	ax.set_ylabel( '2nd Eig', fontsize=szfont  )
	# 	ax.set_zlabel( '3rd Eig', fontsize=szfont  )
	#
	# 	x = np.linspace(-lims,lims,2)
	#
	# 	ax.plot(x, 0*x, 0*x, color='k' )
	# 	ax.plot(0*x, x, 0*x, color='k' )
	# 	ax.plot(0*x, 0*x, x, color='k' )
	#
	# 	plt.title( title + " network in " + str(year), fontsize=szfont )
	# 	plt.show()
	# 	fig.savefig(str( dirPre + 'out_figures/laplacian_scatter_plots/' + title + str(year) + '_threeEigVecs.png' ),bbox_inches='tight')
	# 	# plt.show()
	# 	plt.close(fig)
