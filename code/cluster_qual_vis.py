# This script will use community detection algorithms from networkx.community module
# to find communities in the Global Trade Network data to then generate visualizations
# of the affect of partition resolution on clustering output.


# (0). Import packages and libraries
import numpy as np
from collections import Counter
import networkx as nx
import community as c 					# python-louvain module
import sklearn.cluster as skc
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import itertools
import utils.data_manipulation as dm
import utils.network_manipulation as nm
import utils.plot_functions as pf



#------------------------------------------------------------------------------------------------------------
# Load in a network for a specific year
dirPre = dm.set_dir_tree()

for file_year in np.arange(1962, 1972):
    try:
        year = np.array(file_year)
        flg_sym = True
        G = nm.construct_ntwrkX_Graph(dirPre=dirPre, year=year, flg_sym=flg_sym)
        giant = max(nx.connected_component_subgraphs(G), key=len)
        out_deg = len(list(G)) - len(giant)
    except:
        pass




#------------------------------------------------------------------------------------------------------------
# Explore 'community' module
# Compute best partition and dendrogram using Louvain algorithm in 'community' module

    res = [0.1,0.5,1,3,5,7,10] # different resolution values for partitioning algorithms

    q_bp		= np.zeros_like(res)
    q_dend		= np.zeros( (3,len(res)) )
    coverage 	= np.zeros_like(res)
    clust_in_res = np.zeros(len(res))
    res_count_y, res_x, max_y, totals = [], [], [], []

    for i,r in enumerate(res):
        print('Resolution is ',r)
        #
        # (1). compute partitions
        part = c.best_partition(G, partition=None, weight='weight', resolution=r, randomize=False)
        dend = c.generate_dendrogram(G, part_init=None, weight='weight', resolution=r, randomize=False)
        print('Tree depth is ',len(dend))

        # (2). parse data for plotting plot_functions

        res_counts = Counter(part.values())
        partition_sizes = Counter(res_counts.values())
        partitions = np.array([i for i,j in res_counts.items()])
        partition_counts =np.array([j for i,j in res_counts.items()])
        mem_in_part = np.array([i for i,j in partition_sizes.items()])
        num_with_mem = np.array([j for i,j in partition_sizes.items()])
        max_y.append(max(list(res_counts.values())))
        totals.append(str(len(partition_counts)) + 'Clusters')
        for value in res_counts.values():
            res_x.append(r)
            res_count_y.append(value)

        # (3). compute partition quality metrics
        # 		(a). 'modularity'
        q_bp[i] = c.modularity(part, G, weight='weight')
        q_dend[0,i] = c.modularity(dend[0], G, weight='weight')
        clust_in_res[i] = max(part.values())
        # # 		(b). 'coverage' - (note: have to turn partition into a list of sets.)
        try:
        	G2 = c.induced_graph(dend[0], G, weight='weight') 			# define graph turns clusters into nodes.
        	q_dend[1,i] = c.modularity(dend[1], G2, weight='weight')
        	pp = c.partition_at_level(dend,1) 							# express partition at a give layer in terms of all nodes.
        	q_dend[2,i] = c.modularity(pp, G, weight='weight')
        except:
        	continue

#------------------------------------------------------------------------------------------------------------------

## Plot findings
#  Creates two subplots. First is resolution v size of clusters at each resolution, with GCC.
#  Second is resolution v modularity to show cluster resolution.

    if True:
        plt.figure(figsize=(11,8))
        res_xy = np.stack((res_x, res_count_y), axis=1)
        ax0 = plt.subplot2grid((7, 1), (0, 0), rowspan=5)
        ax0.plot(res_xy[:,0], res_xy[:,1], marker='o', ls='None', alpha=0.5, color='g')
        ax0.plot(res, max_y, ms=8, marker='^', ls='None', alpha=0.7, color='r')
        for i, txt in enumerate(totals):
        	ax0.annotate(txt, (res[i]-0.5, max_y[i]+5))
        ax0.plot(res, [len(giant)]*len(res), 'k--', label='GCC', alpha=0.7)
        ax0.text(0.1, len(giant) - 10, s='GCC  Disconnected Nodes = ' + str(out_deg))
        pf.axis_labels(ax0, 'Clustering Resolution of GTN ' + str(file_year), 'Resolution',
                        'Size of Cluster', res, [str(i) for i in res], titfont=18, grid=True)

        ax1 = plt.subplot2grid((7, 1), (6, 0))
        ax1.plot(res,q_bp,'b')
        pf.axis_labels(ax1, 'Cluster Quality', 'Resolution', 'Modularity',
                        res, [str(i) for i in res], titfont=18, grid=True )

        # plt.show()
        plt.savefig(str( dirPre + 'out_figures/cluster_quality_figures/clust_res_GTN_' + str(file_year) + '.png' ))
