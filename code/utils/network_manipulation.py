import numpy as np
#import pandas as pd
import networkx as nx
#import pickle
import scipy.sparse as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian

import utils.data_manipulation as dm


def construct_ntwrkX_Graph(dirPre, year, flg_sym):
    ## (1). Make a NetworkX Graph Object from an Adjacency matrix. It is a directed network with edge weights
    # equal to the amount of goods sent from country i to country j (or vice versa). Nodes are tagged
    # information about country name, Lat & Lon, continent, total imports & total exports. The resulting
    # Graph object will be saved as a gpickle file.

    if flg_sym:
        G = nx.Graph()  # create the weighted undirected (ie. symmetric) graph.
        #Gg = nx.Graph()
        sym = 'sym'
    else:
        G = nx.DiGraph()  # create the weighted directed graph.
        #Gg = nx.DiGraph()
        sym = ''

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
        A, I, E = dm.load_adjacency_npz_year(dirAdj, year, num_countries, flg_sym)
    except:
        print('Adjacency File not found.')
        return


    for a in range(num_countries):
        G.add_nodes_from( [country_indx[a]], LatLon=LonLat[a], countryId3=countryId3[a], countryName=countryName[a],
                          continent=continent[a], imports=I[a], exports=E[a] )
        #Gg.add_nodes_from( [country_indx[a]], LatLon=LonLat[a], countryId3=countryId3[a], countryName=countryName[a],
        #          continent=continent[a], imports=I[a], exports=E[a] )

        for b in range(num_countries):
            if A[a, b] > 0:
                G.add_weighted_edges_from( [(a, b, A[a, b])] )  # create the weighted directed graph.
                #Gg.add_edge( a, b, weight=str(A[a, b]) )

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



    # if flg_sym:
    #     trade_ntwrkG = nx.Graph()  # create the weighted undirected (ie. symmetric) graph.
    #     sym = 'sym'
    # else:
    #     trade_ntwrkG = nx.DiGraph()  # create the weighted directed graph.
    #     sym = ''

    # # ----------------------------------------------------------------------------------------------------------
    # # (A). Get latitude and longitude information from a pickle UTF8 file created from a csv and set them
    # # construct nodes with attributes of name.
    # countriesLL = dm.load_country_lat_lon_csv(dirPre) # lat_lon = dm.load_lat_lon_pickle(fileIn_LatLon)
    # num_countries = countriesLL.shape[0]

    # for a in range(num_countries):
    #     sample = countriesLL.iloc[[a]]
    #     lon = sample['longitude'][0]
    #     lat = sample['latitude'][0]
    #     trade_ntwrkG.add_node( a, LatLon=str((lon, lat)), countryId3=sample['id_3char'][0], countryName=sample['name'][0],
    #                           continent=sample['id'][0][:2], imports=str(imports[a].astype(int)), exports=str(exports[a].astype(int)) )

    # # (B). Set up weighted edges for World Trade NetworkAdj Graph.
    # # nWidth = np.empty(0) # weight of edges in the graph object for nice visualization

    # for o in range(num_countries):
    #     # print([ str(o) + " : " + countries.id_3char[o] ])
    #     for d in range(num_countries):
    #         if networkAdj[o, d] > 0:
    #             trade_ntwrkG.add_edge( o, d, weight=str(networkAdj[o, d].astype(int) ) )  # create the weighted directed graph.

    # # (C). Save a gpickle file containing the networkAdj constructed in
    # nx.write_gpickle(trade_ntwrkG, str(dirPre + 'adjacency_ntwrkX_pickle_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gpickle'))

    # # (D). Save a gexf file containing the networkAdj to use with Gephi toolbox
    # nx.write_gexf( trade_ntwrkG, str(dirPre + 'adjacency_gexf_network_files/' + sym + 'trade_ntwrkX_' + str(year) + '.gexf'), encoding='utf-8', prettyprint=True, version='1.2draft')

    # return trade_ntwrkG




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



# -------------------------------- # -------------------------------- # --------------------------------



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




