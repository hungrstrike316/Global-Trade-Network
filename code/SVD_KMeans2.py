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

    def __init__(self, year, method, flg_sym):
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
        self.kmLabels = None

    def svd(self):
        Ui, Si, Vi = np.linalg.svd(self.trade_ntwrk, full_matrices=True,
                                   compute_uv=True)
        return Ui, Si, Vi

    def kmeans(self, numClust, nDims,  Vi):
        """Compute Singular Value Decomposition on trade_ntwrkA
        (without anything on the diagonal)
        Returns:
            tuple
        """
        km = skc.KMeans(n_clusters=numClust, n_init=10, max_iter=300,
                        tol=0.001, verbose=False).fit(Vi[0:nDims].T)
        kmLabels = km.labels_
        kmCenters = km.cluster_centers_
        kmParams = km
        self.kmLabels = kmLabels
        return kmLabels, kmCenters, kmParams

    def reformat_kmLabels_nx(self, numClust):
        """reformat kmLabels to be used in nx quality function

        Returns:
            list: list of sets. Each set contains the index of countries
        """
        assert self.kmLabels is not None, "Run kmeans method first " \
                                          "to get the labels."
        community = [set() for _ in range(numClust)]
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

    def kmeans_quality_measure(self, quality_measure, labels):
        """

        Returns:
            float: the quality measure of the clustering

        """
        assert self.kmLabels is not None, "Run kmeans method before running" \
                                          "quality measures in order to set" \
                                          "the labels"
        if quality_measure is 'louvain_modularity':
            assert self.flg_sym is 'sym', "louvain modularity does not accept" \
                                          "asymmetrical graphs"
            # labels = self.reformat_kmLabels_c()
            return c.modularity(labels, self.G)
        else:
            # labels = self.reformat_kmLabels_nx()
            if quality_measure is "modularity":
                return nx.algorithms.community.quality.modularity(self.G,
                                                                  labels,
                                                                  'weight')
            if quality_measure is "coverage":
                return nx.algorithms.community.quality.coverage(self.G,
                                                                labels)
            if quality_measure is "performance":
                return nx.algorithms.community.quality.performance(self.G,
                                                                   labels)
            if quality_measure is "density":
                return cq.density(self.G, labels)
            if quality_measure is "conductance":
                return cq.conductance(self.G, labels)


# -----------------------------------------------------------------------------

def get_labels(kmeans_object, quality_measure, numClust):
    """

    Args:
        kmeans_object (Kmeans): Kmeans object
        quality_measure (str): "modularity" or "density" or "conductance"
        numClust (int): number of clusters

    Returns:

    """
    if quality_measure is 'louvain_modularity':
        return kmeans_object.reformat_kmLabels_c()
    else:
        return kmeans_object.reformat_kmLabels_nx(numClust)


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
    kmeans_object = Kmeans(year, method, flg_sym)
    Vi = kmeans_object.svd()[2]
    for ki, k in enumerate(numClustList):
        for di, d in enumerate(nDimsList):
            kmeans_object.kmeans(k, d, Vi)
            labels = get_labels(kmeans_object, quality_measure, k)
            quality = kmeans_object.kmeans_quality_measure(quality_measure,
                                                           labels)
            quality_gather[ki][di] = quality
    return quality_gather


def kmeans_multiple_quality_matrix(year, method, qualities_list,
                                   numClustList, nDimsList, flg_sym):
    """

    Args:
        year (int): e.g 2006
        method (str): "Adjacency" or "Laplacian"
        qualities_list (list): list of quality names, e.g ["modularity",
                                                            "density"]
        numClustList (list): e.g [2, 3, 5, 7, 10, 15]
        nDimsList (list): e.g [3, 5, 10, 20, 35, 50, 100, 150, 200]
        flg_sym (bool): True or False

    Returns:

    """
    quality_gathers = [np.zeros((len(numClustList), len(nDimsList))) for _ in
                       range(len(qualities_list))]
    kmeans_object = Kmeans(year, method, flg_sym)
    Vi = kmeans_object.svd()[2]
    for quality_index in range(len(qualities_list)):
        for ki, k in enumerate(numClustList):
            for di, d in enumerate(nDimsList):
                kmeans_object.kmeans(k, d, Vi)
                quality_measure = qualities_list[quality_index]
                labels = get_labels(kmeans_object, quality_measure, k)
                quality = kmeans_object.kmeans_quality_measure(quality_measure,
                                                               labels)
                quality_gathers[quality_index][ki][di] = quality
    return quality_gathers


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
    plt.grid()
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


def multiple_qualities_plot(quality_matrices, qualities_list, numClustList,
                            nDimList, year, name):
    for quality_index in range(len(qualities_list)):
        quality_plot(quality_matrices[quality_index],
                     qualities_list[quality_index], numClustList, nDimList,
                     year, name + "_" + qualities_list[quality_index] + ".png")
    