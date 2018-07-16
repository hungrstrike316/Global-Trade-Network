import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import networkx as nx
import community as c
import sklearn.cluster as skc
import utils.data_manipulation as dm
import utils.network_manipulation as nm
import qualities as cq


class Clustering:
    dirPre = dm.set_dir_tree()
    dirIn = str(dirPre + 'adjacency_ntwrk_npz_files/')
    countriesLL = dm.load_country_lat_lon_csv(
        dirPre)  # country names & ids and Lat,Lon information.
    num_countries = countriesLL.shape[0]

    def __init__(self, year, method, flg_sym, norm="norm"):
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
        if method is "Laplacian":
            self.trade_ntwrk = nm.networkX_laplacian(self.G, self.flg_sym,
                                                     norm)
        else:
            self.trade_ntwrk = nm.construct_ntwrk_method(self.trade_ntwrk_graph,
                                                         method)
        self.labels = None

    def svd(self):
        """Compute Singular Value Decomposition on trade_ntwrkA
        (without anything on the diagonal)

        Returns:
            tuple

        """
        Ui, Si, Vi = np.linalg.svd(self.trade_ntwrk, full_matrices=True,
                                   compute_uv=True)
        return Ui, Si, Vi

    def kmeans(self, numClust, nDims,  Vi):
        """Computes kmeans clustering based on svd

        Returns:
            tuple
        """
        km = skc.KMeans(n_clusters=numClust, n_init=10, max_iter=300,
                        tol=0.001, verbose=False).fit(Vi[0:nDims].T)
        kmLabels = km.labels_
        kmCenters = km.cluster_centers_
        kmParams = km
        self.labels = kmLabels
        return kmLabels, kmCenters, kmParams

    def best_partition(self):
        best_partitions = c.best_partition(self.G)
        formatted_label = np.empty((len(best_partitions)), dtype=np.int32)
        for country_index in best_partitions:
            formatted_label[country_index] = best_partitions[country_index]
        self.labels = list(formatted_label)
        return list(formatted_label)

    def reformat_kmLabels_nx(self, numClust):
        """reformat kmLabels to be used in nx quality function

        Returns:
            list: list of sets. Each set contains the index of countries
        """
        assert self.labels is not None, "Run kmeans method first " \
                                        "to get the labels."
        community = [set() for _ in range(numClust)]
        for i in range(len(self.labels)):
            # Pass the clusters without any nodes
            community[self.labels[i]].add(i)
        return community

    def reformat_kmLabels_c(self):
        """

        Returns:
            dict: a dictionary where keys are their nodes
            and values the clusters

        """
        assert self.labels is not None, "Run kmeans method first to get" \
                                        "the labels."
        partition = {}
        for i in range(len(self.labels)):
            partition[i] = self.labels[i]
        return partition

    def cluster_quality_measure(self, quality_measure, labels):
        """

        Returns:
            float: the quality measure of the clustering

        """
        assert self.labels is not None, "Run kmeans method before running" \
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
            else:
                raise ValueError("Quality measure is not found.")


# -----------------------------------------------------------------------------

def get_labels(cluster_object, quality_measure, numClust):
    """

    Args:
        cluster_object (Clustering): Clustering object
        quality_measure (str): "modularity" or "density" or "conductance"
        numClust (int): number of clusters

    Returns:

    """
    if quality_measure is 'louvain_modularity':
        return cluster_object.reformat_kmLabels_c()
    else:
        return cluster_object.reformat_kmLabels_nx(numClust)


def clusters_sizes(clusters):
    """

    Args:
        clusters (list): List of sets.

    Returns:
        list

    """
    n_c_list = []
    for cluster in clusters:
        n_c_list.append(len(cluster))
    return n_c_list


def count_item_frequency(list, item):
    count = 0
    for i in list:
        if i == item:
            count += 1
    return count



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
    clusters_gather = dict()
    cluster_object = Clustering(year, method, flg_sym)
    Vi = cluster_object.svd()[2]
    for ki, k in enumerate(numClustList):
        for di, d in enumerate(nDimsList):
            cluster_object.kmeans(k, d, Vi)
            labels = get_labels(cluster_object, quality_measure, k)

            quality = cluster_object.cluster_quality_measure(quality_measure,
                                                             labels)
            cluster_sizes = clusters_sizes(labels)
            clusters_gather[(ki, di)] = cluster_sizes
            quality_gather[ki][di] = quality
    return quality_gather, clusters_gather


def kmeans_multiple_quality_matrix(year, method, qualities_list,
                                   numClustList, nDimsList, flg_sym,
                                   best_partition=False):
    """

    Args:
        year (int): e.g 2006
        method (str): "Adjacency" or "Laplacian"
        qualities_list (list): list of quality names, e.g ["modularity",
                                                            "density"]
        numClustList (list): e.g [2, 3, 5, 7, 10, 15]
        nDimsList (list): e.g [3, 5, 10, 20, 35, 50, 100, 150, 200]
        flg_sym (bool): True or False
        best_partition (bool): Set to True if best_partition is desired.

    Returns:
        list
    """
    quality_gathers = [np.zeros((len(numClustList), len(nDimsList))) for _ in
                       range(len(qualities_list))]
    cluster_object = Clustering(year, method, flg_sym)
    Vi = cluster_object.svd()[2]
    for quality_index in range(len(qualities_list)):
        quality_measure = qualities_list[quality_index]
        for ki, k in enumerate(numClustList):
            for di, d in enumerate(nDimsList):
                cluster_object.kmeans(k, d, Vi)
                labels = get_labels(cluster_object, quality_measure, k)
                quality = cluster_object.cluster_quality_measure(quality_measure,
                                                                labels)
                quality_gathers[quality_index][ki][di] = quality
        if best_partition:
            cluster_object.best_partition()
            k = max(cluster_object.labels) + 1
            labels = get_labels(cluster_object, quality_measure, k)
            quality = cluster_object.cluster_quality_measure(quality_measure,
                                                             labels)
            best_partition_row = np.full((1, len(nDimsList)), quality)
            quality_gathers[quality_index] = np.concatenate((quality_gathers[quality_index],
                                              best_partition_row))
    return quality_gathers


def kmeans_cluster_size_visualization_2d_hist(quality_matrix, clusters_gather,
                                      method, quality, numClustList, nDimList,
                                      year, name):
    print(clusters_gather)
    for ki in range(len(numClustList)):
        x = []
        y = []
        for di in range(len(nDimList)):
            cluster_sizes = clusters_gather[(ki, di)]
            for i in range(len(cluster_sizes)):
                x.append(nDimList[di])
                y.append(cluster_sizes[i])

        plt.title("HELLO")
        f, axarr = plt.subplots(2, 1)
        h = axarr[0].hist2d(x, y, bins=20)
        axarr[0].grid()
        axarr[0].set_xticks(nDimList)
        f.colorbar(h[3], ax=axarr[0])

        axarr[1].grid()
        axarr[1].set_xlabel("Dims")
        axarr[1].set_ylabel("quality Measure")
        axarr[1].set_title(quality + " (weighted) Measure, Symmetric, " + method +
              ", Year " + str(year))

        axarr[1].plot(nDimList, quality_matrix[ki, :])
        plt.tight_layout()
        plt.savefig("../out_figures/cluster_quality_measures/" + name + str(ki) + ".png")
        plt.clf()


def cluster_visualizer(quality_matrix, clusters_gather,
                                      method, quality, numClustList, nDimList,
                                      year, name):
    for ki in range(len(numClustList)):
        f, axarr = plt.subplots(2, 1, figsize=(20, 10))
        f.suptitle(quality + " (weighted) Measure, Symmetric, " + method +
              ", Year " + str(year), fontsize=20)
        axarr[0].grid()
        axarr[0].set_xlabel("Dims")
        axarr[0].set_ylabel("Cluster Size")
        axarr[0].set_xticks(nDimList)
        for di in range(len(nDimList)):
            cluster_sizes = clusters_gather[(ki, di)]
            for size in cluster_sizes:
                size_frequency = count_item_frequency(cluster_sizes, size)
                cm = plt.cm.get_cmap('jet')
                axarr[0].scatter([nDimList[di]], size, s = [80],
                                 c=[size_frequency], vmin=0, vmax=10, cmap=cm)
        PCM = axarr[0].get_children()[2]
        # plt.colorbar(PCM, cax=axarr[0])
        cb_ticks = np.arange(11)
        f.colorbar(PCM, ax=axarr[0], orientation='horizontal', ticks=cb_ticks)
        # axarr[0].set_yticks(unique_sizes)
        # plt.setp(axarr[0].get_yticklabels(), rotation=45, horizontalalignment='right')
        axarr[1].grid()
        axarr[1].set_xlabel("Dims")
        axarr[1].set_ylabel("quality Measure")
        # axarr[1].set_title()

        axarr[1].plot(nDimList, quality_matrix[ki, :], linewidth=3.3)
        s = np.full((len(nDimList)), 80)
        axarr[1].scatter(nDimList, quality_matrix[ki, :], s=s)
        plt.tight_layout()
        f.subplots_adjust(top=0.93)
        plt.savefig("../out_figures/cluster_quality_measures/" + name + str(ki) + ".png")
        plt.clf()


def quality_plot(quality_matrix, method, quality, numClustList, nDimList, year,
                 name):
    """

    Args:
        quality_matrix (numpy.ndarray): Matrix of shape
                                        len(numClustList) X len(nDimList)
        method (str): "Adjacency" or "Laplacian"
        quality (str): "modularity" or "density" or "conductance"
        numClustList (list): e.g [2, 3, 5, 7, 10, 15, "best_partition]
        nDimList (list): e.g [3, 5, 10, 20, 35, 50, 100, 150, 200]
        year (int): e.g 2006
        name: name of the image. For example, "density_2006.png"

    Returns:
        -

    """

    plt.grid()
    plt.xlabel("Dims")
    plt.ylabel("Quality Measure")
    plt.title(quality + " (weighted) Measure, Symmetric, " + method +
              ", Year " + str(year), y=1.04)
    for k in range(len(numClustList)):
        plt.plot(nDimList, quality_matrix[k, :],
                 label="K = " + str(numClustList[k]))
        plt.scatter(nDimList, quality_matrix[k, :])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("../out_figures/cluster_quality_measures/" + name)
    plt.clf()


def quality_plot_with_clusters(quality_matrix, clusters_list, method, quality,
                               numClustList, nDimList, year, name):
    zeros = np.zeros_like(nDimList)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(0.0, 261.0)
    for k in range(len(numClustList)):
        ax.plot(nDimList,quality_matrix[k, :], zeros, label="K = " + str(numClustList[k]))
        ax.scatter(nDimList, quality_matrix[k, :], zeros)
    ax.legend(loc="upper left")
    plt.savefig("../out_figures/cluster_quality_measures/" + name)
    plt.clf()


def multiple_qualities_plot(quality_matrices, method, qualities_list,
                            numClustList, nDimList, year, name):
    for quality_index in range(len(qualities_list)):
        quality_plot(quality_matrices[quality_index], method,
                     qualities_list[quality_index], numClustList, nDimList,
                     year, name + "_" + qualities_list[quality_index] + ".png")
