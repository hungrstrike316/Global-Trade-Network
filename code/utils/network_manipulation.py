import numpy as np
import pandas as pd
import networkx as nx
import pickle
import scipy.sparse as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian

import utils.data_manipulation as dm


def construct_ntwrkX_Graph(fileIn_LatLon, networkAdj, imports, exports, dirOut_ntwrkX, year):
    # Make a NetworkX Graph Object from an Adjacency matrix. It is a directed network with edge weights
    # equal to the amount of goods sent from country i to country j (or vice versa). Nodes are tagged
    # information about country name, Lat & Lon, continent, total imports & total exports. The resulting
    # Graph object will be saved as a gpickle file.

    trade_ntwrkG = nx.DiGraph()  # create the weighted directed graph.

    # ----------------------------------------------------------------------------------------------------------
    # (1). Get latitude and longitude information from a pickle UTF8 file created from a csv and set them
    # construct nodes with attributes of name.
    lat_lon = dm.load_lat_lon_pickle(fileIn_LatLon)
    cntr = 0
    for row in lat_lon:  # num_countries):
        # print(row)
        try:
            xx = float(row[4])
            yy = float(row[3])
        except:
            xx = 0
            yy = -90
        trade_ntwrkG.add_node(cntr, LatLon=(xx, yy), countryId3=row[1], countryName=row[2],
                              continent=row[0][:2], imports=imports[cntr], exports=exports[cntr])
        cntr += 1
        # print([xx, yy])
        # print(trade_ntwrkG.row[1])
    a = cntr

    # (2). Set up weighted edges for World Trade NetworkAdj Graph.
    # nWidth = np.empty(0) # weight of edges in the graph object for nice visualization

    for o in range(0, a):  # num_countries):
        # print([ str(o) + " : " + countries.id_3char[o] ])

        for d in range(0, a):  # num_countries):
            if networkAdj[o, d] > 0:
                trade_ntwrkG.add_edge(o, d, weight=networkAdj[o, d])  # create the weighted directed graph.
                # nWidth = np.append(nWidth, networkAdj[o,d])

    # (3). Save a gpickle file containing the networkAdj constructed in
    nx.write_gpickle(trade_ntwrkG, str(dirOut_ntwrkX + 'trade_ntwrkX_' + str(year) + '.gpickle'))

    return trade_ntwrkG


def cuthill_mckee(trade_ntwrk):
    if np.all(trade_ntwrk == trade_ntwrk.transpose()):
        sym = True
    else:
        sym = False

    perm = sp.csgraph.reverse_cuthill_mckee(sp.csc_matrix(trade_ntwrk), symmetric_mode=sym)
    return perm


def construct_ntwrk_method(trade_ntwrk, method):
    if method == "Adjacency":
        print()
    elif method == "Normalized Laplacian":
        trade_ntwrk = sp.csgraph.laplacian(trade_ntwrk, normed=True, use_out_degree=False)
    elif method == "Modularity":
        print(str(method + ' not Implemented yet'))
    elif method == "Topographic Modularity":
        print(str(method + ' not Implemented yet'))
    else:
        print(str(method + ' does not match any method I know of.'))

    return trade_ntwrk


def modularity(adjacency, return_symmetric=False):
    """Returns modularity matrix of a directed graph. If A is the adjacency
    matrix of a directed graph, then the modularity matrix B is given by B_ij =
    A_ij - (in_degree_i * out_degree_j)/(total degree of network)

    See 'Leicht, E. A., & Newman, M. E. (2008). Community structure in directed networks.
    Physical review letters, 100(11), 118703'

    Parameters
    ----------
    adjacency : NumPy array_like
        Adjacency matrix of graph
    return_symmetric : boolean
        Boolean flag that specifies whether to return undirected modularity or
        symmetricized undirected modularity, defaults to False. See Returns.
    Returns
    -------
    ndarray
        If return_symmetric is True, returns B + B', otherwise returns B.

    """
    in_degree = np.sum(adjacency, axis=0)
    out_degree = np.sum(adjacency, axis=1)
    B = adjacency - (np.multiply.outer(in_degree, out_degree)) / np.sum(adjacency)
    if return_symmetric:
        return B + B.T
    return B
