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
import community as c
import sys
import sklearn.cluster as skc
import sklearn.metrics as skm
import scipy as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian
import utils.data_manipulation as dm
import utils.network_manipulation as nm
import utils.plot_functions as pf


class Kmeans:
	dirPre = dm.set_dir_tree()
	dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
	countriesLL = dm.load_country_lat_lon_csv(dirPre) # country names & ids and Lat,Lon information.
	num_countries = countriesLL.shape[0]

	def __init__(self, year, method, quality_measure, numClust, nDims, flg_sym):
		self.year = year
		self.flg_sym = flg_sym
		self.trade_ntwrk, self.imports, self.exports = dm.load_adjacency_npz_year(self.dirIn, year, self.num_countries, flg_sym)
		assert np.any( np.sum(self.trade_ntwrk,axis=0) == self.imports) , 'Imports are Weird'
		assert np.any( np.sum(self.trade_ntwrk,axis=1) == self.exports) , 'Exports are Weird'
		self.trade_ntwrk = nm.construct_ntwrk_method(self.trade_ntwrk, method)
		self.numClust = numClust
		self.nDims = nDims
		self.kmLabels = None
		self.quality_measure = quality_measure
		self.method = method

	def kmeans(self):
		# Compute Singular Value Decomposition on trade_ntwrkA (without anything on the diagonal)
		Ui, Si, Vi = np.linalg.svd(self.trade_ntwrk, full_matrices=True, compute_uv=True)
		km = skc.KMeans(n_clusters=self.numClust, n_init=10, max_iter=300, tol=0.001, verbose=False).fit(Vi[0:self.nDims].T)
		kmLabels = km.labels_
		kmCenters = km.cluster_centers_
		kmParams = km
		self.kmLabels = kmLabels
		return kmLabels, kmCenters, kmParams

	def reformat_kmLabels_nx(self):
		"""
		reformat kmLabels to be used in nx quality function
		"""
		community = [set() for _ in range(self.numClust)]
		for i in range(len(self.kmLabels)):
			community[self.kmLabels[i]].add(i)
		return community

	def reformat_kmLabels_c(self):
		partition = {}
		for i in range(len(self.kmLabels)):
			partition[i] = self.kmLabels[i]
		return partition

	def kmeans_quality_measure(self):
		assert self.kmLabels is not None, "Run kmeans method before running quality measures in order to set the labels"
		G = nm.construct_ntwrkX_Graph(self.dirPre, self.year, self.flg_sym)
		if self.quality_measure is 'louvain_modularity':
			assert self.flg_sym is True, "louvain modularity does not accept asymmetrical graphs"
			reformed_kmLabels = self.reformat_kmLabels_c()
			return c.modularity(reformed_kmLabels, G)
		else:
			reformed_kmLabels = self.reformat_kmLabels_nx()
			if self.quality_measure is "modularity":
				return nx.algorithms.community.quality.modularity(G, reformed_kmLabels)
			if self.quality_measure is "coverage":
				return nx.algorithms.community.quality.coverage(G, reformed_kmLabels)
			if self.quality_measure is "performance":
				return nx.algorithms.community.quality.performance(G, reformed_kmLabels)

# ------------------------------------------------------------------------------------------------------------

def show_country_names_for_cluster(kmeans_object, numClust, kmLabels):
	xx=[]
	for i in range(numClust):
		print('Cluster #', i)
		inds = np.where(kmLabels == i)
		inds = inds[0]
		ninds = np.setxor1d(inds, range(kmeans_object.num_countries) )


		#print(countriesLL.name[inds])
		tradesWithinCluster = kmeans_object.trade_ntwrk[ np.ix_(inds,inds) ]
		tradesLeavingCluster = kmeans_object.trade_ntwrk[ np.ix_(inds,ninds) ]
		tradesEnteringCluster = kmeans_object.trade_ntwrk[ np.ix_(ninds,inds) ]
		tradesOutsideCluster = kmeans_object.trade_ntwrk[ np.ix_(ninds,ninds) ]


		xx = [tradesWithinCluster[np.nonzero(tradesWithinCluster)].mean(),		
				tradesLeavingCluster[np.nonzero(tradesLeavingCluster)].mean(),	
				tradesEnteringCluster[np.nonzero(tradesEnteringCluster)].mean(),	
				tradesOutsideCluster[np.nonzero(tradesOutsideCluster)].mean(),
				kmeans_object.trade_ntwrk[np.nonzero(kmeans_object.trade_ntwrk)].mean(), inds.size ]

		print(xx)

def show_quality_score(kmeans_object, quality_gather, nDimsList, numClustList):
	ax = plt.subplot(1,1,1)
	plt.imshow(quality_gather)
	plt.colorbar()
	plt.title( str('Average Silhouette Score for K-means Clustering of Global Trade network in ' + str(kmeans_object.year) + ' with ' + kmeans_object.method) )
	ax.set_xticks(range(len(nDimsList)))
	ax.set_xticklabels(nDimsList)
	plt.xlabel('# of SVDs used')
	ax.set_yticks(  range(len(numClustList)))
	ax.set_yticklabels(numClustList)
	plt.ylabel('# of Clusters (k)')
	plt.show()

def show_variance_cluster_size(kmeans_object, ClSz_var, nDimsList, numClustList):
	ax = plt.subplot(1,1,1)
	plt.imshow(ClSz_var)
	plt.colorbar()
	plt.title( str('Cluster Size Variance for K-means Clustering of Global Trade network in ' + str(kmeans_object.year) + ' with ' + kmeans_object.method) )
	ax.set_xticks( range(len(nDimsList)))
	ax.set_xticklabels(nDimsList)
	plt.xlabel('# of SVDs used')
	ax.set_yticks(  range(len(numClustList)))
	ax.set_yticklabels(numClustList)
	plt.ylabel('# of Clusters (k)')
	plt.show()

def kmeans_matrix(year, method, quality_measure, numClustList, nDimsList, flg_sym):
	quality_gather = np.zeros((len(numClustList), len(nDimsList)))
	ClSz_var = np.zeros((len(numClustList), len(nDimsList)))

	for ki, k in enumerate(numClustList):
		for di, d in enumerate(nDimsList):
			kmeans_object = Kmeans(year, method, quality_measure, k, d, flg_sym)
			# Perfrom kmeans algorithm on kmeans_object (the labels are assigned in the class)
			kmeans_object.kmeans()
			quality = kmeans_object.kmeans_quality_measure()
			quality_gather[ki][di] = quality
			ClSz, _ = np.histogram(kmeans_object.kmLabels, k)
			ClSz_var[ki][di] = ClSz.var()
			# Show country names that belong to each cluster
			show_country_names_for_cluster(kmeans_object, k, kmeans_object.kmLabels)

	# Show quality score clustering performance for different params.
	show_quality_score(kmeans_object, quality_gather, nDimsList, numClustList)

	# Show Variance on Cluster Sizes for each combination of parameter settings.
	show_variance_cluster_size(kmeans_object, ClSz_var, nDimsList, numClustList)


# -------------------------------------------------------------------------------------



