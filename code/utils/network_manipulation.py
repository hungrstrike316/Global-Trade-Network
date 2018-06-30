import numpy as np
import networkx as nx
import scipy.sparse as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian
import utils.data_manipulation as dm


def construct_ntwrkX_Graph(dirPre, year, sym):
    ## (1). Make a NetworkX Graph Object from an Adjacency matrix. It is a directed network with edge weights
    # equal to the amount of goods sent from country i to country j (or vice versa). Nodes are tagged
    # information about country name, Lat & Lon, continent, total imports & total exports. The resulting
    # Graph object will be saved as a gpickle file.

    if (sym=='sym'):
        G = nx.Graph()  # create the weighted undirected (ie. symmetric) graph.
    else:
        G = nx.DiGraph()  # create the weighted directed graph.


    # ----------------------------------------------------------------------------------------------------------
    # (A). Get latitude and longitude information from a pickle UTF8 file created from a csv and set them
    # construct nodes with attributes of name.
    countriesLL = dm.load_country_lat_lon_csv(dirPre)
    num_countries = countriesLL.shape[0]
    country_indx = list(range(num_countries))

    continent =  [countriesLL['id'][row][:2] for row in range(num_countries)]
    countryId3 = [countriesLL['id_3char'][row] for row in range(num_countries)]
    countryName = [countriesLL['name'][row] for row in range(num_countries)]
    LonLat = [ (countriesLL['longitude'][row], countriesLL['latitude'][row]) for row in range(num_countries)]


    dirAdj = str( dirPre + 'adjacency_ntwrk_npz_files/' )
    try:
        # load in adjacency for a given year.
        A, I, E = dm.load_adjacency_npz_year(dirAdj, year, num_countries, sym)
    except:
        print('Adjacency File not found.')
        return


    for a in range(num_countries):
        G.add_nodes_from( [country_indx[a]], LatLon=LonLat[a], countryId3=countryId3[a], countryName=countryName[a],
                          continent=continent[a], imports=I[a], exports=E[a] )

        for b in range(num_countries):
            if A[a, b] > 0:
                G.add_weighted_edges_from( [(a, b, A[a, b])] )  # create the weighted directed graph.

    # # Note: To access data for each node or edge, do:
    # G.nodes.data('LatLon')[0]
    # G.nodes.data('countryId3')[0]
    # G.nodes.data('countryName')[0]
    # G.nodes.data('continent')[0]
    # G.nodes.data('imports')[0]
    # G.nodes.data('exports')[0]
    # G.edges.data('weight')


    # (C). Save a gpickle file containing the networkAdj constructed in
    nx.write_gpickle(G, str(dirPre + 'adjacency_ntwrkX_pickle_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gpickle'))

    # # (D). Save a gexf file containing the networkAdj to use with Gephi toolbox
    # nx.write_gexf( G, str(dirPre + 'adjacency_gexf_network_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gexf'), encoding='utf-8', prettyprint=True, version='1.2draft')
    #
    # # Note: Graph Data File Used for Gephi not working currently. Come back.

    return G




# -------------------------------- # -------------------------------- # --------------------------------
def load_ntwrkX_Graph(dirPre, year, sym=''):
    # Read in gpickle file containing the networkAdj constructed in
    trade_ntwrkG = nx.read_gpickle( str(dirPre + 'adjacency_ntwrkX_pickle_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gpickle') )

    # # Read in gexf file containing the networkAdj to use with Gephi toolbox
    # trade_ntwrkG = nx.read_gexf( str(dirPre + 'adjacency_gexf_network_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gexf') )

    return trade_ntwrkG


# -------------------------------- # -------------------------------- # --------------------------------





def cuthill_mckee(trade_ntwrk):
    ## (2). Compute the Cuthill-Mckee reordering of adjacency (or other) matrix.
    # Tries to make sparse matrix block diagonal. This is essentially a clustering.
    if np.any(trade_ntwrk != trade_ntwrk.transpose()):
        sym = False
    else:
        sym = True

    perm = sp.csgraph.reverse_cuthill_mckee(sp.csc_matrix(trade_ntwrk), symmetric_mode=sym)
    return perm



# -------------------------------- # -------------------------------- # --------------------------------



def construct_ntwrk_method(trade_ntwrk, method):
    ## (3). A helper function that you feed adjacency matrix and a string denoting the method and
    #  it will implement that method and print a message which method it is implementing.
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



# -------------------------------- # -------------------------------- # --------------------------------

def networkX_laplacian(adj_graph, sym, norm, weight='weight'):
    # Compute the laplacian matrix using NetworkX built in functions.
    #
    # Can take symmetric or directed graphs (sym='sym' or sym='').
    # Can take weighted or unweighted graphs (weight='weight' or weight=None).
    # Can create normalized or non-normalized matrices (norm='norm' or norm='').
    #
    if (sym=='sym') and (norm=='norm'):
        B = nx.normalized_laplacian_matrix(adj_graph,weight=weight).A # nx returns is numpy martrix, .A returns array.
    elif sym=='' and norm=='norm':
        B = nx.directed_laplacian_matrix(adj_graph,weight=weight).A  # nx returns is numpy martrix, .A returns array.
    else:
        B = nx.laplacian_matrix(adj_graph,weight=weight).toarray()  # nx returns scipy sparse, toarray() returns a np array.
                                                            # Ensures cohesion with nx normalized and nx directed modules,
                                                            # and supports np.array indexing conventions.

    return B

# -------------------------------- # -------------------------------- # --------------------------------



def modularity(adjacency, sym):
    ## (4). Returns modularity matrix of a directed graph. If A is the adjacency
    # matrix of a directed graph, then the modularity matrix B is given by B_ij =
    # A_ij - (in_degree_i * out_degree_j)/(total degree of network)
    #
    # See 'Leicht, E. A., & Newman, M. E. (2008). Community structure in directed networks.
    # Physical review letters, 100(11), 118703'
    #
    # Parameters
    # ----------
    # adjacency : NumPy array_like
    #    Adjacency matrix of graph
    # return_symmetric : string
    #    Boolean flag that specifies whether to return undirected modularity or
    #    symmetricized undirected modularity, defaults to False. See Returns.
    # Returns
    # -------
    # ndarray
    #    If sym=='sym' is True, (returns B + B')/2, otherwise returns B.

    in_degree = np.sum(adjacency, axis=0)
    out_degree = np.sum(adjacency, axis=1)
    B = adjacency - (np.multiply.outer(out_degree, in_degree) / adjacency.sum() ) #np.sum(adjacency)

    if (sym=='sym'):
        return (B + B.T)/2

    return B



def networkX_modularity(adj_graph,sym,weight='weight'):
    # Compute the modularity matrix using NetworkX built in function.
    #
    # Can take symmetric or directed graphs (sym='sym' or sym='').
    # Can take weighted or unweighted graphs (weight='weight' or weight=None).
    #
    if (sym=='sym'):
        B = nx.modularity_matrix(adj_graph,weight=weight)
    else:
        B = nx.directed_modularity_matrix(adj_graph,weight=weight)

    return B

# -------------------------------- # --------------------------------


def convert_adjacency_to_giant_component(G, adjacency):
    giant = max(nx.connected_components(G), key=len)
    giant = list(giant)
    selected_columns = adjacency[:, giant]
    selected_columns_and_rows = selected_columns.T[:, giant].T
    return selected_columns_and_rows

# -------------------------------- # --------------------------------


def adjacency_matrix_from_graph(g):
    """Returns the adjacency matrix of a networkx graph
    Args:
        g (networkx.classes.graph.Graph): NetworkX graph
    Returns:
        numpy.ndarray: The adjacency matrix of g
    """
    a = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()), weight='weight')
    return np.array(a.todense())


def xyz(adj):

    print(adj)

    # nx.community.LFR_benchmark_graph
    # nx.community.community_utils
    # nx.community.kclique
    # nx.community.modularity
    # nx.community.asyn_fluidc
    # nx.community.coverage
    # nx.community.kernighan_lin
    # nx.community.performance
    # nx.community.asyn_lpa_communities
    # nx.community.girvan_newman
    # nx.community.kernighan_lin_bisection
    # nx.community.quality
    # nx.community.centrality
    # nx.community.is_partition
    # nx.community.label_propagation
    # nx.community.community_generators
    # nx.community.k_clique_communities
    # nx.community.label_propagation_communities
