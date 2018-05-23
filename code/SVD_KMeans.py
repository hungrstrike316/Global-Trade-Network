

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
import sklearn.cluster as skc
import sklearn.metrics as skm
import scipy as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian



import utils.data_manipulation as dm
import utils.network_manipulation as nm
import utils.plot_functions as pf


flg_plot_1st3_SVDs			= True

flg_cluster_DBSCAN			= False
flg_cluster_agglomerative	= False
flg_cluster_kmeans			= False

flg_symmetric_network = True # Network is made undirected if true. Otherwise it is naturally directed or has asymmetric adjacency matrix.


# IF RUNNING AS A FUNCTION, I NEED TO INPUT.
#
# years, method {A, Ln, Q}, 
#
# What is happening inside?
#
# (1). Reorder Matrix with Cuthill-Mckee algorithm to get clusters.
# (2).
#




#method = 'Adjacency'
#method = 'NormLaplacian'
method = 'Modularity'
#method = 'TopoModularity'


#------------------------------------------------------------------------------------------------------------
## (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
dirPre = dm.set_dir_tree()

#------------------------------------------------------------------------------------------------------------
## (1). Load country names that align with 3 letter acronyms used in origin destination file
#countries = dm.load_countries(dirPre)

countriesLL = dm.load_country_lat_lon_csv(dirPre) # country names & ids and Lat,Lon information.
num_countries = countriesLL.shape[0]


## (2) Load in names and codes for types of goods traded
#goods = dm.load_products(dirPre)  


# (3). Get array indicating continents for each country to colorcode nodes in graph.
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



if flg_symmetric_network:
	sym = 'sym'
else:
	sym = ''	


# (4). Loop through, load and plot all previously saved adjacency matrix files.  
years =  range(1962,1963) # years for which we have world trade data. range(1962,2015)

for y in years:
	print(y)

	dirIn = str( dirPre + 'adjacency_ntwrk_npz_files/' )
	try:
		trade_ntwrk, imports, exports = dm.load_adjacency_npz_year(dirIn, y, num_countries, flg_symmetric_network)
	except:
		print('Can not find adjacency file')
		continue




	# # (3). Plot adjacency matrix and total imports / exports from each country.
	# if False:
	# 	fig = plt.figure(figsize=(15,15))
	# 	plt.subplot(2,1,1)
	# 	plt.imshow( np.log(trade_ntwrk), interpolation='none' )
	# 	plt.title( str('Global Trade Network in ' + str(y)) )
	# 	plt.colorbar(fraction=0.046, pad=0.04)
	# 	plt.xticks( range(0,num_countries), countriesLL.id_3char[0:num_countries] )
	# 	plt.yticks( range(0,num_countries), countriesLL.id_3char[0:num_countries] )
    
	# 	plt.subplot(2,2,3)
	# 	imprt = plt.plot( np.log(imports))#, label="import" )
	# 	exprt = plt.plot( np.log(exports))#, label="export" )
	# 	meanIE = plt.plot( np.log( np.mean([imports,exports],axis=0) ))
	# 	plt.title("Import & Export for Individual Countries")
	# 	plt.ylabel("Log of Trade Value")
	# 	plt.xlabel("Country")
	# 	plt.xticks( range(0,num_countries), countriesLL.id_3char[0:num_countries] )
	# 	plt.legend(["Imports","Exports","Avg of I&E"])
    
	# 	plt.subplot(2,2,4)
	# 	imprt = plt.plot( np.log( imports - exports) )
	# 	plt.title("Difference between Import & Export")
	# 	plt.ylabel("Log Diff Trade Value")
	# 	plt.xlabel("Country")
	# 	plt.xticks( range(0,num_countries), countriesLL.id_3char[0:num_countries] )

	# 	fig.savefig(str( dirPre + 'out_figures/adjacency_mats/' + str(y) + '_boring.png' ),bbox_inches='tight')
	# 	#plt.show()
	# 	plt.close(fig)


	# (4). Check that Imports are just the column sums and Exports are row sums
	assert np.any( np.sum(trade_ntwrk,axis=0) == imports) , 'Imports are Weird'
	assert np.any( np.sum(trade_ntwrk,axis=1) == exports) , 'Exports are Weird'
	print('Everything checks out with sums, imports and exports.')


	# (5c). Compute Normalized Laplacian (For Asymmetric, use_out_degree can use imports or exports)
	trade_ntwrk = nm.construct_ntwrk_method(trade_ntwrk,method)

	


	# (6). Compute Singular Value Decomposition on trade_ntwrkA (without anything on the diagonal)
	Ui, Si, Vi = np.linalg.svd( trade_ntwrk, full_matrices=True, compute_uv=True )


	# # (7). Plot first couple singular values (Ui = left side one, Vi = right side one)
	# if False:
	# 	fig = plt.figure(figsize=(15,15))

	# 	plt.subplot(4,1,1)
	# 	plt.plot( np.log10(Si) )
	# 	plt.title( str('Log of SVD on ' + method + ' (Analogy to Eigenvalues)') )

	# 	numSVDs = 3 # number of SVDs
	# 	for p in range(0,numSVDs):
	    
	# 		plt.subplot(4,2,(3+p*2) )
	# 		plt.plot( np.sort(UiL[p][:]) )
	# 		plt.plot( np.sort(UiL[:][p]) )
	# 		plt.title(str("U "+ str(p)) )  
	                  
	# 		plt.subplot(4,2,(4+p*2) )
	# 		plt.plot( np.sort(ViL[p][:]) )
	# 		plt.plot( np.sort(ViL[:][p]) )
	# 		plt.title(str("V "+ str(p)) )

	# 		fig.savefig(str( dirPre + 'out_figures/single_SVDs/' + method + '_' + str(y) + '.png' ),bbox_inches='tight')
	# 		plt.show()
	# 		plt.close(fig)



	# # (8). Plot whole U and V matrices. It looks like Ui = Vi' for all.
	# if False:   
	# 	fig = plt.figure(figsize=(15,15))
	# 	plt.set_cmap('bwr')
    
	# 	plt.subplot(2,2,1)
	# 	plt.imshow( Ui, interpolation='none' )
	# 	plt.title("SVD's U in trade network in " + str(y))
	# 	plt.colorbar(fraction=0.046, pad=0.04)
	# 	plt.xticks( range(0,a), countriesLL.id_3char[0:a] )
	# 	plt.yticks( range(0,a), countriesLL.id_3char[0:a] )

	# 	plt.subplot(2,2,2)
	# 	plt.imshow( Vi.transpose(), interpolation='none' )
	# 	plt.title("SVD's V' in trade network in " + str(y))
	# 	plt.colorbar(fraction=0.046, pad=0.04)
	# 	plt.xticks( range(0,a), countriesLL.id_3char[0:a] )
	# 	plt.yticks( range(0,a), countriesLL.id_3char[0:a] )    
    
	# 	plt.subplot(2,2,3)
	# 	plt.imshow( Ui.transpose(), interpolation='none' )
	# 	plt.title("SVD's U' in trade network in " + str(y))
	# 	plt.colorbar(fraction=0.046, pad=0.04)
	# 	plt.xticks( range(0,a), countriesLL.id_3char[0:a] )
	# 	plt.yticks( range(0,a), countriesLL.id_3char[0:a] )

	# 	plt.subplot(2,2,4)
	# 	plt.imshow( Vi, interpolation='none' )
	# 	plt.title("SVD's V in trade network in " + str(y))
	# 	plt.colorbar(fraction=0.046, pad=0.04)
	# 	plt.xticks( range(0,a), countriesLL.id_3char[0:a] )
	# 	plt.yticks( range(0,a), countriesLL.id_3char[0:a] ) 

	# 	fig.savefig(str( dirPre + 'out_figures/SVD_transposes/' + method + '_' + str(y) + '.png' ),bbox_inches='tight')
	# 	plt.show()
	# 	plt.close(fig)   



	# (9). Are Ui and Vi' the same thing?  Yes they are. Is this always true? I dont think it should be.
	assert np.any( Ui.transpose() == Vi) , 'Ui is NOT the transpose of Vi. Is this weird?'
	print('Ui is transpose of Vi')



	# (10). Look at clusters found with first couple Singular values individually.
	if False:
		c=5 # number of countries to show for each Singular Value.
		numSVDs = 2
	    
		for e in range(0,numSVDs):
			print(str('SVD Vec #' + str(e)))
			Y = Vi[e][:]      # array to be sorted
			x = np.sort(Y)    # array sorted from smallest to largest
			z = np.argsort(Y) # returns indexes of sorted array

			print('Largest Positive')
			print( countries.name[z[0:c]] )
			print( continent[z[0:c]] )

			print('Largest negative')
			print( countries.name[z[-1-c:-1]] )
			print( continent[z[-1-c:-1]] )



	# (11). Plot the total Import and Export of countries against their order in the individual SVDs.
	if False:
		fig = plt.figure(figsize=(15,15))
		numSVDs = 3
	    
		for e in range(0,numSVDs):
	        
			Y = Vi[e][:]      # array to be sorted
			x = np.sort(Y)    # array sorted from smallest to largest
			z = np.argsort(Y) # returns indexes of sorted array

			plt.subplot(numSVDs, 2, 2*e+1)    
			plt.plot( np.log10( imports[z]) )
			plt.plot( np.log10( exports[z]) )
			plt.legend(['Import','Export'])
			plt.ylabel( str('SVD Vec #' + str(e)) )

			if e==0:
				plt.title("Log 10 of Import and Export Quantity For Each Country")
				plt.xlabel( "Countries sorted by SVD vector value" )

		fig.savefig(str( dirPre + 'out_figures/indiv_SVD_imports_exports/' + sym + method + '_' + str(y) + '.png' ),bbox_inches='tight')
		plt.show()
		plt.close(fig)





	# (12). Plot countries where locations are values in each SVD vector.
	if flg_plot_1st3_SVDs:
		fig=plt.figure(figsize=(15,15))
		

		# Plot first couple SVD with colors indicating which continent the countries are on.    
		ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	    
		for i in range(0,7):
			indx = node_colors_by_continent==i
			ax.scatter(Vi[0][indx], Vi[1][indx], Vi[2][indx], s=150, edgecolor='k', c=colors[i], alpha=0.4)
	    
		lims=1    
		ax.set_xlim(-lims,lims)
		ax.set_ylim(-lims,lims)
		ax.set_zlim(-lims,lims)    

		gt=0.1
		for i,cname in countriesLL.name.iteritems():     
			i = np.int(i)  
			if i<num_countries:
				if np.any([np.abs(Vi[0][i])>gt, np.abs(Vi[1][i])>gt, np.abs(Vi[2][i])>gt]):   
					#print( str( cname + " " + str(Vi[0][i])  + " " +  str(Vi[1][i]) ) + " " +  str(Vi[2][i]) )
					ax.text( Vi[0][i]+0.03, Vi[1][i]+0.03, Vi[2][i]+0.03, cname ) 

		szfont = 24        
		ax.set_xlabel( 'SVD #0', fontsize=szfont )
		ax.set_ylabel( 'SVD #1', fontsize=szfont  )
		ax.set_zlabel( 'SVD #2', fontsize=szfont  )
	    
		x = np.linspace(-lims,lims,2)
	    
		ax.plot(x, 0*x, 0*x, color='k' ) 
		ax.plot(0*x, x, 0*x, color='k' ) 
		ax.plot(0*x, 0*x, x, color='k' ) 

		plt.title("SVD's V in trade network in " + str(y), fontsize=szfont )   

		fig.savefig(str( dirPre + 'out_figures/SVD_scatter_plots/' + sym + method + '_' + str(y) + '_threeSVDs.png' ),bbox_inches='tight')
		#plt.show()
		plt.close(fig)



	# (13). Check how many values in each individual SVD vector are larger than some value (gt)
	if False:
		print(np.where(np.abs(Vi[0][:])>gt))
		print(np.where(np.abs(Vi[1][:])>gt))
		print(np.where(np.abs(Vi[2][:])>gt))




	# (14). Do clustering (K-means, DBSCAN, Agglomerative Hierarchical)

	# Grid Search over different K values and #'s of SVDs used.
	numClust = [5] #[5,7,9,10,15,20,50]
	nDims	 = [3] #[3,5,7,10,50,100,200,num_countries]

	#CHS_gather 	= np.zeros( (len(numClust),len(nDims)) )
	silh_gather = np.zeros( (len(numClust),len(nDims)) )
	ClSz_var 	= np.zeros( (len(numClust),len(nDims)) )

	if flg_cluster_kmeans:

		for ki,k in enumerate(numClust):
			for di,d in enumerate(nDims):
		
				print(ki,di)
				print('Compute Kmeans on SVDs')	
				t = time.time()
				km = skc.KMeans(n_clusters=k, n_init=10, max_iter=300, tol=0.001, verbose=False).fit(Vi[0:d].T)
				print( 'Time = ', time.time()-t )
				#
				kmLabels = km.labels_
				kmCenters = km.cluster_centers_
				kmParams = km
				#np.savez(file_path, kmCenters, kmLabels, kmParams)
				

				# Compute cluster metrics
				t = time.time()
				#CHS = skm.cluster.calinski_harabaz_score(Vi[0:d].T, kmLabels)
				#silh_score = skm.cluster.silhouette_samples(Vi[0:d].T, kmLabels, metric='euclidean')
				silh_avg = skm.cluster.silhouette_score(Vi[0:d].T, kmLabels, metric='euclidean')
				print('Time = ', time.time() - t)

				#CHS_gather[ki][di] = CHS
				silh_gather[ki][di] = silh_avg

				# Distribution of cluster sizes.
				ClSz,_ = np.histogram(kmLabels,k)
				ClSz_var[ki][di] = ClSz.var()

				# Show country names that belong to each cluster
				xx=[]
				for i in range(k):
					print('Cluster #',i)
					inds = np.where(kmLabels == i)
					inds = inds[0]
					ninds = np.setxor1d(inds, range(num_countries) )


					#print(countriesLL.name[inds])
					tradesWithinCluster = trade_ntwrk[ np.ix_(inds,inds) ]
					tradesLeavingCluster = trade_ntwrk[ np.ix_(inds,ninds) ]
					tradesEnteringCluster = trade_ntwrk[ np.ix_(ninds,inds) ]
					tradesOutsideCluster = trade_ntwrk[ np.ix_(ninds,ninds) ]


					xx = [tradesWithinCluster[np.nonzero(tradesWithinCluster)].mean(),		
							tradesLeavingCluster[np.nonzero(tradesLeavingCluster)].mean(),	
							tradesEnteringCluster[np.nonzero(tradesEnteringCluster)].mean(),	
							tradesOutsideCluster[np.nonzero(tradesOutsideCluster)].mean(),
							trade_ntwrk[np.nonzero(trade_ntwrk)].mean(), inds.size ]

					print(xx)		


					# f,ax = plt.subplots(2,2)
					# ax[0][0].imshow( np.log10(tradesWithinCluster), interpolation='none' ) # trading within cluster.
					# ax[1][0].imshow( np.log10(tradesLeavingCluster), interpolation='none' ) # trading within cluster.
					# ax[0][1].imshow( np.log10(tradesEnteringCluster), interpolation='none' ) # trading within cluster.
					# ax[1][1].imshow( np.log10(tradesOutsideCluster), interpolation='none' ) # trading within cluster.
					# f.show()



		# Show silhouette score clustering performance for different params.
		ax = plt.subplot(1,1,1)
		plt.imshow(silh_gather)
		plt.colorbar()
		plt.title( str('Average Silhouette Score for K-means Clustering of Global Trade network in ' + str(y) + ' with ' + method) )
		ax.set_xticks( range(len(nDims)) )
		ax.set_xticklabels(nDims)
		plt.xlabel('# of SVDs used')
		ax.set_yticks(  range(len(numClust))  )
		ax.set_yticklabels(numClust)
		plt.ylabel('# of Clusters (k)')
		plt.show()			


		# Show Variance on Cluster Sizes for each combination of parameter settings.
		ax = plt.subplot(1,1,1)
		plt.imshow(ClSz_var)
		plt.colorbar()
		plt.title( str('Cluster Size Variance for K-means Clustering of Global Trade network in ' + str(y) + ' with ' + method) )
		ax.set_xticks( range(len(nDims)) )
		ax.set_xticklabels(nDims)
		plt.xlabel('# of SVDs used')
		ax.set_yticks(  range(len(numClust))  )
		ax.set_yticklabels(numClust)
		plt.ylabel('# of Clusters (k)')
		plt.show()







	# Do hierarchical agglomerative clustering on spike words.
	if flg_cluster_agglomerative:

		print('Compute Agglomerative Clustering directly on binary spike words')	
		t = time.time()
		ac = skc.AgglomerativeClustering(n_clusters=numClust, affinity='euclidean', linkage='ward').fit(Vi[0:nDims].T)
		print('Time = ',time.time()-t)


	# Do hierarchical agglomerative clustering on spike words.
	if flg_cluster_DBSCAN:

		print('Compute DBSCAN Clustering directly on binary spike words')	
		t = time.time()
		db = skc.DBSCAN(eps=0.5, min_samples=2).fit(Vi[0:nDims].T)
		print('Time = ',time.time()-t)	



