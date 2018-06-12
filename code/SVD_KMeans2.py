import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as c
import sklearn.cluster as skc
import utils.data_manipulation as dm
import utils.network_manipulation as nm
import qualities as cq


class Kmeans:
    dirPre = dm.set_dir_tree()
    dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
    countriesLL = dm.load_country_lat_lon_csv(
        dirPre)  # country names & ids and Lat,Lon information.
    num_countries = countriesLL.shape[0]

    def __init__(self, year, method, quality_measure, numClust, nDims, flg_sym):
        self.year = year
        if flg_sym:
            self.flg_sym = 'sym'
        else:
            self.flg_sym = ''
        self.G = nm.construct_ntwrkX_Graph(self.dirPre, self.year, self.flg_sym)
        self.trade_ntwrk_graph, self.imports, self.exports =\
            dm.load_adjacency_npz_year(self.dirIn, year, self.num_countries,
                                       self.flg_sym)
        assert np.any(np.sum(self.trade_ntwrk_graph,
                             axis=0) == self.imports), 'Imports are Weird'
        assert np.any(np.sum(self.trade_ntwrk_graph,
                             axis=1) == self.exports), 'Exports are Weird'
        self.trade_ntwrk = nm.construct_ntwrk_method(self.trade_ntwrk_graph,
                                                     method)
        self.numClust = numClust
        self.nDims = nDims
        self.kmLabels = None
        self.quality_measure = quality_measure
        self.method = method

    def kmeans(self):
        """# Compute Singular Value Decomposition on trade_ntwrkA
        (without anything on the diagonal)
        Returns:
            tuple
        """
        Ui, Si, Vi = np.linalg.svd(self.trade_ntwrk, full_matrices=True,
                                   compute_uv=True)
        km = skc.KMeans(n_clusters=self.numClust, n_init=10, max_iter=300,
                        tol=0.001, verbose=False).fit(Vi[0:self.nDims].T)
        kmLabels = km.labels_
        kmCenters = km.cluster_centers_
        kmParams = km
        self.kmLabels = kmLabels
        return kmLabels, kmCenters, kmParams

    def reformat_kmLabels_nx(self):
        """reformat kmLabels to be used in nx quality function

        Returns:
            list: list of sets. Each set contains the index of countries
        """
        assert self.kmLabels is not None, "Run kmeans method first " \
                                          "to get the labels."
        community = [set() for _ in range(self.numClust)]
        for i in range(len(self.kmLabels)):
            # Pass the clusters without any nodes
            community[self.kmLabels[i]].add(i)
        return community

    def reformat_kmLabels_c(self):
        """

        Returns:
            dict: a dictionary where keys are their nodes
            and values the clusters

        """
        assert self.kmLabels is not None, "Run kmeans method first to get" \
                                          "the labels."
        partition = {}
        for i in range(len(self.kmLabels)):
            partition[i] = self.kmLabels[i]
        return partition

    def kmeans_quality_measure(self):
        """

        Returns:
            float: the quality measure of the clustering

        """
        assert self.kmLabels is not None, "Run kmeans method before running" \
                                          "quality measures in order to set" \
                                          "the labels"
        if self.quality_measure is 'louvain_modularity':
            assert self.flg_sym is 'sym', "louvain modularity does not accept" \
                                          "asymmetrical graphs"
            labels = self.reformat_kmLabels_c()
            return c.modularity(labels, self.G)
        else:
            labels = self.reformat_kmLabels_nx()
            if self.quality_measure is "modularity":
                return nx.algorithms.community.quality.modularity(self.G,
                                                                  labels,
                                                                  'weight')
            if self.quality_measure is "coverage":
                return nx.algorithms.community.quality.coverage(self.G,
                                                                labels)
            if self.quality_measure is "performance":
                return nx.algorithms.community.quality.performance(self.G,
                                                                   labels)
            if self.quality_measure is "density":
                return cq.density(self.G, labels)
            if self.quality_measure is "conductance":
                return cq.conductance(self.G, labels)


# -----------------------------------------------------------------------------


def kmeans_quality_matrix(year, method, quality_measure, numClustList,
                          nDimsList, flg_sym):
    """

    Args:
        year (int): e.g 2006
        method (str): "Adjacency" or "Laplacian"
        quality_measure (str): "modularity" or "density" or "conductance"
        numClustList (list): e.g [2, 3, 5, 7, 10, 15]
        nDimsList (list): e.g [3, 5, 10, 20, 35, 50, 100, 150, 200]
        flg_sym (bool): True or False

    Returns:
        numpy.ndarray

    """
    quality_gather = np.zeros((len(numClustList), len(nDimsList)))

    for ki, k in enumerate(numClustList):
        for di, d in enumerate(nDimsList):
            kmeans_object = Kmeans(year, method, quality_measure, k, d, flg_sym)
            kmeans_object.kmeans()
            quality = kmeans_object.kmeans_quality_measure()
            quality_gather[ki][di] = quality
    return quality_gather


def quality_plot(quality_matrix, quality, numClustList, nDimList, year, name):
    """

    Args:
        quality_matrix (numpy.ndarray): Matrix of shape
                                        len(numClustList) X len(nDimList)
        quality (str): "modularity" or "density" or "conductance"
        numClustList (list): e.g [2, 3, 5, 7, 10, 15]
        nDimList (list): e.g [3, 5, 10, 20, 35, 50, 100, 150, 200]
        year (int): e.g 2006
        name: name of the image. For example, "density_2006.png"

    Returns:
        -

    """
    plt.xlabel("Dims")
    plt.ylabel("Quality Measure")
    plt.title(quality + " (weighted) Measure, Symmetric, Adjacency, Year " +
              str(year), y=1.04)
    for k in range(len(numClustList)):
        plt.plot(nDimList, quality_matrix[k, :],
                 label="K = " + str(numClustList[k]))
        plt.scatter(nDimList, quality_matrix[k, :])
    plt.legend(loc=1)
    plt.savefig("../out_figures/cluster_quality_measures/" + name)
    plt.clf()
