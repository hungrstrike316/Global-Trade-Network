import numpy as np
#import pandas as pd
import networkx as nx
#import pickle
import scipy.sparse as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian

import utils.data_manipulation as dm


def construct_ntwrkX_Graph( networkAdj, imports, exports, dirPre, year, flg_sym):
    ## (1). Make a NetworkX Graph Object from an Adjacency matrix. It is a directed network with edge weights
    # equal to the amount of goods sent from country i to country j (or vice versa). Nodes are tagged
    # information about country name, Lat & Lon, continent, total imports & total exports. The resulting
    # Graph object will be saved as a gpickle file.

    #networkAdj = networkAdj.astype(int)

    if flg_sym:
        trade_ntwrkG = nx.Graph()  # create the weighted undirected (ie. symmetric) graph.
        sym = 'sym'
    else:
        trade_ntwrkG = nx.DiGraph()  # create the weighted directed graph.
        sym = ''

    # ----------------------------------------------------------------------------------------------------------
    # (A). Get latitude and longitude information from a pickle UTF8 file created from a csv and set them
    # construct nodes with attributes of name.
    countriesLL = dm.load_country_lat_lon_csv(dirPre) # lat_lon = dm.load_lat_lon_pickle(fileIn_LatLon)
    num_countries = countriesLL.shape[0]

    for a in range(num_countries):
        sample = countriesLL.iloc[[a]]
        lon = sample['longitude'][0]
        lat = sample['latitude'][0]
        trade_ntwrkG.add_node( a, LatLon=str((lon, lat)), countryId3=sample['id_3char'][0], countryName=sample['name'][0],
                              continent=sample['id'][0][:2], imports=str(imports[a].astype(int)), exports=str(exports[a].astype(int)) )

    # (B). Set up weighted edges for World Trade NetworkAdj Graph.
    # nWidth = np.empty(0) # weight of edges in the graph object for nice visualization

    for o in range(num_countries):
        # print([ str(o) + " : " + countries.id_3char[o] ])
        for d in range(num_countries):
            if networkAdj[o, d] > 0:
                trade_ntwrkG.add_edge( o, d, weight=str(networkAdj[o, d].astype(int) ) )  # create the weighted directed graph.

    # (C). Save a gpickle file containing the networkAdj constructed in
    nx.write_gpickle(trade_ntwrkG, str(dirPre + 'adjacency_ntwrkX_pickle_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gpickle'))

    # (D). Save a gexf file containing the networkAdj to use with Gephi toolbox
    nx.write_gexf( trade_ntwrkG, str(dirPre + 'adjacency_gexf_network_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gexf'), encoding='utf-8', prettyprint=True, version='1.1draft')

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



def modularity(adjacency, return_symmetric=False):
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
    # return_symmetric : boolean
    #    Boolean flag that specifies whether to return undirected modularity or
    #    symmetricized undirected modularity, defaults to False. See Returns.
    # Returns
    # -------
    # ndarray
    #    If return_symmetric is True, returns B + B', otherwise returns B.

    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
    #%matplotlib inline

    adjacency = adjacency.astype(np.float64)
    
    in_degree = np.sum(adjacency, axis=0)
    out_degree = np.sum(adjacency, axis=1)
    B = adjacency - (np.multiply.outer(out_degree, in_degree) / np.sum(in_degree) ) #np.sum(adjacency)



    """


    print(adjacency)
    print(in_degree)

    # adjacency.nonzero()[0].size) #

    # THE PROBLEM IS THAT NETWORK X VERSION CONSIDERS UNWEIGHTED GRAPHS SO IT
    # DIVIDES BY NUMBER OF NONZERO ENTRIES AND NOT BY THEIR TOTAL WEIGHT !!!


    
    # NOTE X and Y ARE EXACTLY THE SAME.
    X = np.multiply.outer(out_degree, in_degree)

    indsp = csr_matrix(adjacency.sum(axis=0))
    oudsp = csr_matrix(adjacency.sum(axis=1))
    Y = oudsp.T * indsp

    plt.subplot(211)
    plt.imshow(X)
    plt.subplot(212)

    plt.imshow(Y)
    plt.show()
    print(X - Y)
    (X-Y).any()
    



    # Here, testing our modularity vs nx's with unweighted / binary adjacency matrix
    # (Do for both symmetric and non-symmetric)
    xx = np.random.binomial( 1, 0.1, (10,10) ) * np.random.randint(1, 1000, (10,10) )
    yy = xx # np.sign(xx + xx.T)
    yy[np.diag_indices_from(yy)] = 0
    adjacency = yy

    

    # Compute modularity using Baeo's way
    in_degree = np.sum(adjacency, axis=0)
    out_degree = np.sum(adjacency, axis=1)
    B = adjacency - (np.multiply.outer(out_degree, in_degree) / np.sum(in_degree) ) # np.sum(adjacency)

    # Two ways to convert to sparse csr_matrix's. They are the same.
    adj_sp = csr_matrix(adjacency) # DO I NEED TO HAVE ANY SPARSE MATRICES FOR NX? 
    G = nx.to_networkx_graph(adjacency,create_using=nx.DiGraph())
    A = nx.adjacency_matrix(G) # makes a csr_matrix of adjacency matrix of graph G


    # Compute Modularity using NetworkX implementation
    G = nx.to_networkx_graph(adjacency,create_using=nx.Graph())
    B2 = nx.modularitymatrix.modularity_matrix(G)
    np.any(B-B2)


    G = nx.to_networkx_graph(adjacency,create_using=nx.DiGraph())
    B2 = nx.modularitymatrix.directed_modularity_matrix(G)
    np.any(B-B2)




    nx.write_gexf(G, '../adjacency_gexf_network_files/testWt.gexf', encoding='utf-8', prettyprint=True, version='1.1draft')


    # Results.  
    #   (1). They are the same for undirected, unweighted adjacency.
    #   (2). They are the same for directed,   unweighted adjacency.
    #   (3). undirected, weighted ?? 
    #   (4). directed,   weighted ?? 
    







    # Also...
    laplacian_matrix
    directed_modularity_matrix

    """








    if return_symmetric:
        return (B + B.T)/2 

    return B
