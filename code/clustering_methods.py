# This script will use community detection algorithms from networkx.community module
# (and maybe other modules too) to find communities in the Global Trade Network data.
# 


# (0). Import packages and libraries
import numpy as np
import networkx as nx
import community as c 					# python-louvain module
import sklearn.cluster as skc
import sklearn.metrics as skm
import matplotlib.pyplot as plt

import utils.data_manipulation as dm
import utils.network_manipulation as nm
import utils.plot_functions as pf


# import matplotlib.cm as cm
# import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
# import time
# import os
# import csv

# import sys

# import scipy as sp       # library to deal with sparse graphs for Cuthill-Mckee and Laplacian






#------------------------------------------------------------------------------------------------------------
# Load in a network for a specific year

dirPre = dm.set_dir_tree()
year = np.array(1962)
flg_sym = True
G = nm.construct_ntwrkX_Graph(dirPre=dirPre, year=year, flg_sym=flg_sym)





#------------------------------------------------------------------------------------------------------------
# Explore 'community' module
# Compute best partition and dendrogram using Louvain algorithm in 'community' module

res = [0.1,0.5,1,3,5,7,10] # different resolution values for partitioning algorithms

q_bp		= np.zeros_like(res)
q_dend		= np.zeros( (3,len(res)) )
coverage 	= np.zeros_like(res)

for i,r in enumerate(res):
	print('Resolution is ',r)
	#
	# (1). compute partitions
	part = c.best_partition(G, partition=None, weight='weight', resolution=r, randomize=False)
	dend = c.generate_dendrogram(G, part_init=None, weight='weight', resolution=r, randomize=False)
	print('Tree depth is ',len(dend))
	#
	# (2). compute partition quality metrics
	# 		(a). 'modularity'
	q_bp[i] = c.modularity(part, G, weight='weight')
	q_dend[0,i] = c.modularity(dend[0], G, weight='weight')
	#
	# # 		(b). 'coverage' - (note: have to turn partition into a list of sets.)
	# # ???? NOT WORKING AND NOT SURE WHY.
	# partsList = []
	# numParts = part.get( max(part,key=part.get) )
	# for p in range( numParts ):
	# 	partsList.append( set([i for i,j in part.items() if j == p]) )
	# coverage[i] = nx.community.coverage(G, partsList)
	#
	#
	#
	#
	# Looking further into dendrogram.  Makes sense.
	try:
		G2 = c.induced_graph(dend[0], G, weight='weight') 			# define graph turns clusters into nodes.
		q_dend[1,i] = c.modularity(dend[1], G2, weight='weight')
		pp = c.partition_at_level(dend,1) 							# express partition at a give layer in terms of all nodes.
		q_dend[2,i] = c.modularity(pp, G, weight='weight')
	except:
		continue

# Plot modularity metric for different partitions at different resolution parameters.
if False:
	plt.plot(res,q_bp,'b')
	plt.plot(res,q_dend[0],'r')
	plt.plot(res,q_dend[1],'g')
	plt.plot(res,q_dend[2],'k')
	plt.xlabel('Resolution')
	plt.ylabel('modularity')
	plt.legend(['best partition','dendrogram0','dendrogram1a','dendrogram1b'])
	plt.show()


# Plot coverage metric
if True:
	plt.plot(res,coverage,'b')
	plt.xlabel('Resolution')
	plt.ylabel('Coverage')
	plt.legend(['best partition'])
	plt.show()














#------------------------------------------------------------------------------------------------------------
# Explore networkx's community partitioning 

# # # # # # CONSTRUCTING GRAPHS WITH COMMUNITY STRUCTURE TO TEST ALGORITHMS
#
# (1). LFR_benchmark_graph - construct graph for testing community detection algorithms. Define degree distributions, 
# 						community size distributions, strength of edges in graph and in communities.
#
# (2). nx.community.community_generators. - whole library of generators for graphs with community structure
#
#
#
#
# # # # # # CLUSTERING ALGORITHMS
#
# (1). kernighan_lin_bisection - algorithm iteratively swaps pairs of nodes to reduce the edge cut between the two sets.
#
# (2). asyn_fluidc - asynchronous fluid communities algorithm (k = # communities to find) 
#					Note: have to find Giant Connected Component first.
#		part = nx.community.asyn_fluidc(G, k=5, max_iter=100)
#
# (3). girvan_newman - Finds communities in a graph using the Girvan–Newman method.
# 		from operator import itemgetter
# 		G = nx.path_graph(10)
# 		edges = G.edges()
# 		nx.set_edge_attributes(G, {(u, v): v for u, v in edges}, 'weight')
# 		def heaviest(G):
# 			u, v, w = max(G.edges(data='weight'), key=itemgetter(2))
# 			return (u, v)
#	
# 		comp = girvan_newman(G, most_valuable_edge=heaviest)
# 		tuple(sorted(c) for c in next(comp))
#
# (4). asyn_lpa_communities - asynchronous label propagation community detection algorithm
#		nx.community.asyn_lpa_communities(G, weight=None)
#
# (5). label_propagation_communities - Generates community sets determined by label propagation
#		part = nx.community.label_propagation_communities(G)
#
# (6). k_clique_communities - Find k-clique communities in graph using the percolation method.
#		union of all cliques of size k that can be reached through adjacent (sharing k-1 nodes) k-cliques.
#		nx.community.k_clique_communities(G, k, cliques=None)
#
#
#
#
# # # # # # CLUSTER QUALITY METRICS
#
# (1). coverage - quality measure of a partition: ratio of intra-community edges to the total edges in the graph.
#
# (2). performance - ratio of intra-community edges plus inter-community non-edges with the total number of potential edges.
#		perf = nx.community.performance(G, partition)
#
# (3). modularity - Returns the modularity of the given partition of the graph.
#		mod = nx.community.modularity(G, communities, weight='weight')
#
# (4).	nx.community.quality.intra_community_edges
#
# (5).	nx.community.quality.inter_community_edges
#
# (6).	nx.community.quality.inter_community_non_edges

      


# # OTHER NETWORKX STUFF THAT WILL BE USEFUL 
# nx.fruchterman_reingold_layout - nice density based graph visualization.


# nx.AmbiguousSolution                                nx.edge_disjoint_paths                              nx.multiline_adjlist
# nx.DiGraph                                          nx.edge_expansion                                   nx.mycielski
# nx.ExceededMaxIterations                            nx.edge_load_centrality                             nx.mycielski_graph
# nx.Graph                                            nx.edge_subgraph                                    nx.mycielskian
# nx.GraphMLReader                                    nx.edgedfs                                          nx.navigable_small_world_graph
# nx.GraphMLWriter                                    nx.edgelist                                         nx.negative_edge_cycle
# nx.HasACycle                                        nx.edges                                            nx.neighbor_degree
# nx.LCF_graph                                        nx.effective_size                                   nx.neighbors
# nx.MultiDiGraph                                     nx.efficiency                                       nx.network_simplex
# nx.MultiGraph                                       nx.ego                                              nx.networkx
# nx.NetworkXAlgorithmError                           nx.ego_graph                                        nx.newman_watts_strogatz_graph
# nx.NetworkXError                                    nx.eigenvector                                      nx.node_attribute_xy
# nx.NetworkXException                                nx.eigenvector_centrality                           nx.node_boundary
# nx.NetworkXNoCycle                                  nx.eigenvector_centrality_numpy                     nx.node_clique_number
# nx.NetworkXNoPath                                   nx.empty_graph                                      nx.node_connected_component
# nx.NetworkXNotImplemented                           nx.enumerate_all_cliques                            nx.node_connectivity
# nx.NetworkXPointlessConcept                         nx.erdos_renyi_graph                                nx.node_degree_xy
# nx.NetworkXTreewidthBoundExceeded                   nx.estrada_index                                    nx.node_disjoint_paths
# nx.NetworkXUnbounded                                nx.euler                                            nx.node_expansion
# nx.NetworkXUnfeasible                               nx.eulerian_circuit                                 nx.node_link
# nx.NodeNotFound                                     nx.exception                                        nx.node_link_data
# nx.NotATree                                         nx.expanders                                        nx.node_link_graph
# nx.OrderedDiGraph                                   nx.expected_degree_graph                            nx.nodes
# nx.OrderedGraph                                     nx.extended_barabasi_albert_graph                   nx.nodes_with_selfloops
# nx.OrderedMultiDiGraph                              nx.extrema_bounding                                 nx.non_edges
# nx.OrderedMultiGraph                                nx.fast_could_be_isomorphic                         nx.non_neighbors
# nx.PowerIterationFailedConvergence                  nx.fast_gnp_random_graph                            nx.nonisomorphic_trees
# nx.absolute_import                                  nx.faster_could_be_isomorphic                       nx.normalized_cut_size
# nx.adamic_adar_index                                nx.fiedler_vector                                   nx.normalized_laplacian_matrix
# nx.add_cycle                                        nx.filters                                          nx.not_implemented_for
# nx.add_path                                         nx.find_cliques                                     nx.null_graph
# nx.add_star                                         nx.find_cliques_recursive                           nx.number_attracting_components
# nx.adj_matrix                                       nx.find_cores                                       nx.number_connected_components
# nx.adjacency                                        nx.find_cycle                                       nx.number_of_cliques
# nx.adjacency_data                                   nx.find_induced_nodes                               nx.number_of_edges
# nx.adjacency_graph                                  nx.florentine_families_graph                        nx.number_of_isolates
# nx.adjacency_matrix                                 nx.flow                                             nx.number_of_nodes
# nx.adjacency_spectrum                               nx.flow_hierarchy                                   nx.number_of_nonisomorphic_trees
# nx.adjlist                                          nx.flow_matrix                                      nx.number_of_selfloops
# nx.algebraic_connectivity                           nx.floyd_warshall                                   nx.number_strongly_connected_components
# nx.algebraicconnectivity                            nx.floyd_warshall_numpy                             nx.number_weakly_connected_components
# nx.algorithms                                       nx.floyd_warshall_predecessor_and_distance          nx.numeric_assortativity_coefficient
# nx.all                                              nx.freeze                                           nx.numeric_mixing_matrix
# nx.all_neighbors                                    nx.from_dict_of_dicts                               nx.nx
# nx.all_node_cuts                                    nx.from_dict_of_lists                               nx.nx_agraph
# nx.all_pairs_bellman_ford_path                      nx.from_edgelist                                    nx.nx_pydot
# nx.all_pairs_bellman_ford_path_length               nx.from_graph6_bytes                                nx.nx_pylab
# nx.all_pairs_dijkstra                               nx.from_nested_tuple                                nx.nx_shp
# nx.all_pairs_dijkstra_path                          nx.from_numpy_array                                 nx.nx_yaml
# nx.all_pairs_dijkstra_path_length                   nx.from_numpy_matrix                                nx.octahedral_graph
# nx.all_pairs_lowest_common_ancestor                 nx.from_pandas_adjacency                            nx.operators
# nx.all_pairs_node_connectivity                      nx.from_pandas_edgelist                             nx.optimal_edit_paths
# nx.all_pairs_shortest_path                          nx.from_prufer_sequence                             nx.optimize_edit_paths
# nx.all_pairs_shortest_path_length                   nx.from_scipy_sparse_matrix                         nx.optimize_graph_edit_distance
# nx.all_shortest_paths                               nx.from_sparse6_bytes                               nx.ordered
# nx.all_simple_paths                                 nx.frucht_graph                                     nx.out_degree_centrality
# nx.ancestors                                        nx.fruchterman_reingold_layout                      nx.overall_reciprocity
# nx.antichains                                       nx.full_rary_tree                                   nx.pagerank
# nx.approximate_current_flow_betweenness_centrality  nx.function                                         nx.pagerank_alg
# nx.articulation_points                              nx.gaussian_random_partition_graph                  nx.pagerank_numpy
# nx.assortativity                                    nx.general_random_intersection_graph                nx.pagerank_scipy
# nx.astar                                            nx.generalized_degree                               nx.pairs
# nx.astar_path                                       nx.generate_adjlist                                 nx.pajek
# nx.astar_path_length                                nx.generate_edgelist                                nx.pappus_graph
# nx.atlas                                            nx.generate_gexf                                    nx.parse_adjlist
# nx.attr_matrix                                      nx.generate_gml                                     nx.parse_edgelist
# nx.attr_sparse_matrix                               nx.generate_graphml                                 nx.parse_gml
# nx.attracting                                       nx.generate_multiline_adjlist                       nx.parse_graphml
# nx.attracting_component_subgraphs                   nx.generate_pajek                                   nx.parse_leda
# nx.attracting_components                            nx.generators                                       nx.parse_multiline_adjlist
# nx.attribute_assortativity_coefficient              nx.generic                                          nx.parse_pajek
# nx.attribute_mixing_dict                            nx.geographical_threshold_graph                     nx.partial_duplication_graph
# nx.attribute_mixing_matrix                          nx.geometric                                        nx.path_graph
# nx.attrmatrix                                       nx.get_edge_attributes                              nx.periphery
# nx.authority_matrix                                 nx.get_node_attributes                              nx.petersen_graph
# nx.average_clustering                               nx.gexf                                             nx.planted_partition_graph
# nx.average_degree_connectivity                      nx.global_efficiency                                nx.power
# nx.average_neighbor_degree                          nx.global_parameters                                nx.powerlaw_cluster_graph
# nx.average_node_connectivity                        nx.global_reaching_centrality                       nx.predecessor
# nx.average_shortest_path_length                     nx.gml                                              nx.preferential_attachment
# nx.balanced_tree                                    nx.gn_graph                                         nx.prefix_tree
# nx.barabasi_albert_graph                            nx.gnc_graph                                        nx.product
# nx.barbell_graph                                    nx.gnm_random_graph                                 nx.project
# nx.beamsearch                                       nx.gnp_random_graph                                 nx.projected_graph
# nx.bellman_ford_path                                nx.gnr_graph                                        nx.quotient_graph
# nx.bellman_ford_path_length                         nx.goldberg_radzik                                  nx.ra_index_soundarajan_hopcroft
# nx.bellman_ford_predecessor_and_distance            nx.gomory_hu_tree                                   nx.radius
# nx.betweenness                                      nx.google_matrix                                    nx.random_clustered
# nx.betweenness_centrality                           nx.gpickle                                          nx.random_clustered_graph
# nx.betweenness_centrality_source                    nx.graph                                            nx.random_degree_sequence_graph
# nx.betweenness_centrality_subset                    nx.graph6                                           nx.random_geometric_graph
# nx.betweenness_subset                               nx.graph_atlas                                      nx.random_graphs
# nx.bfs_beam_edges                                   nx.graph_atlas_g                                    nx.random_k_out_graph
# nx.bfs_edges                                        nx.graph_clique_number                              nx.random_kernel_graph
# nx.bfs_predecessors                                 nx.graph_edit_distance                              nx.random_layout
# nx.bfs_successors                                   nx.graph_number_of_cliques                          nx.random_lobster
# nx.bfs_tree                                         nx.graphical                                        nx.random_partition_graph
# nx.biconnected                                      nx.graphmatrix                                      nx.random_powerlaw_tree
# nx.biconnected_component_edges                      nx.graphml                                          nx.random_powerlaw_tree_sequence
# nx.biconnected_component_subgraphs                  nx.graphviews                                       nx.random_regular_graph
# nx.biconnected_components                           nx.greedy_color                                     nx.random_shell_graph
# nx.bidirectional_dijkstra                           nx.grid_2d_graph                                    nx.random_tree
# nx.bidirectional_shortest_path                      nx.grid_graph                                       nx.reaching
# nx.binary                                           nx.harmonic                                         nx.read_adjlist
# nx.binomial_graph                                   nx.harmonic_centrality                              nx.read_edgelist
# nx.bipartite                                        nx.has_bridges                                      nx.read_gexf
# nx.boundary                                         nx.has_path                                         nx.read_gml
# nx.boundary_expansion                               nx.havel_hakimi_graph                               nx.read_gpickle
# nx.breadth_first_search                             nx.heawood_graph                                    nx.read_graph6
# nx.bridges                                          nx.hexagonal_lattice_graph                          nx.read_graphml
# nx.bull_graph                                       nx.hierarchy                                        nx.read_leda
# nx.capacity_scaling                                 nx.hits                                             nx.read_multiline_adjlist
# nx.cartesian_product                                nx.hits_alg                                         nx.read_pajek
# nx.caveman_graph                                    nx.hits_numpy                                       nx.read_shp
# nx.center                                           nx.hits_scipy                                       nx.read_sparse6
# nx.centrality                                       nx.hoffman_singleton_graph                          nx.read_weighted_edgelist
# nx.chain_decomposition                              nx.house_graph                                      nx.read_yaml
# nx.chains                                           nx.house_x_graph                                    nx.readwrite
# nx.chordal                                          nx.hub_matrix                                       nx.reciprocity
# nx.chordal_cycle_graph                              nx.hybrid                                           nx.recursive_simple_cycles
# nx.chordal_graph_cliques                            nx.hypercube_graph                                  nx.relabel
# nx.chordal_graph_treewidth                          nx.icosahedral_graph                                nx.relabel_gexf_graph
# nx.chvatal_graph                                    nx.identified_nodes                                 nx.relabel_nodes
# nx.circulant_graph                                  nx.immediate_dominators                             nx.relaxed_caveman_graph
# nx.circular_ladder_graph                            nx.in_degree_centrality                             nx.release
# nx.circular_layout                                  nx.incidence_matrix                                 nx.reportviews
# nx.classes                                          nx.induced_subgraph                                 nx.rescale_layout
# nx.classic                                          nx.info                                             nx.resource_allocation_index
# nx.clique                                           nx.information_centrality                           nx.restricted_view
# nx.cliques_containing_node                          nx.intersection                                     nx.reverse
# nx.closeness                                        nx.intersection_all                                 nx.reverse_view
# nx.closeness_centrality                             nx.intersection_array                               nx.rich_club_coefficient
# nx.closeness_vitality                               nx.inverse_line_graph                               nx.richclub
# nx.cluster                                          nx.is_aperiodic                                     nx.ring_of_cliques
# nx.clustering                                       nx.is_arborescence                                  nx.rooted_product
# nx.cn_soundarajan_hopcroft                          nx.is_attracting_component                          nx.s_metric
# nx.coloring                                         nx.is_biconnected                                   nx.scale_free_graph
# nx.common_neighbors                                 nx.is_bipartite                                     nx.sedgewick_maze_graph
# nx.communicability                                  nx.is_branching                                     nx.selfloop_edges
# nx.communicability_alg                              nx.is_chordal                                       nx.semiconnected
# nx.communicability_betweenness_centrality           nx.is_connected                                     nx.set_edge_attributes
# nx.communicability_exp                              nx.is_digraphical                                   nx.set_node_attributes
# nx.community                                        nx.is_directed                                      nx.shell_layout
# nx.complement                                       nx.is_directed_acyclic_graph                        nx.shortest_path
# nx.complete_bipartite_graph                         nx.is_distance_regular                              nx.shortest_path_length
# nx.complete_graph                                   nx.is_dominating_set                                nx.shortest_paths
# nx.complete_multipartite_graph                      nx.is_edge_cover                                    nx.shortest_simple_paths
# nx.components                                       nx.is_empty                                         nx.similarity
# nx.compose                                          nx.is_eulerian                                      nx.simple_cycles
# nx.compose_all                                      nx.is_forest                                        nx.simple_paths
# nx.condensation                                     nx.is_frozen                                        nx.single_source_bellman_ford
# nx.conductance                                      nx.is_graphical                                     nx.single_source_bellman_ford_path
# nx.configuration_model                              nx.is_isolate                                       nx.single_source_bellman_ford_path_length
# nx.connected                                        nx.is_isomorphic                                    nx.single_source_dijkstra
# nx.connected_caveman_graph                          nx.is_k_edge_connected                              nx.single_source_dijkstra_path
# nx.connected_component_subgraphs                    nx.is_kl_connected                                  nx.single_source_dijkstra_path_length
# nx.connected_components                             nx.is_matching                                      nx.single_source_shortest_path
# nx.connected_double_edge_swap                       nx.is_maximal_matching                              nx.single_source_shortest_path_length
# nx.connected_watts_strogatz_graph                   nx.is_multigraphical                                nx.single_target_shortest_path
# nx.connectivity                                     nx.is_negatively_weighted                           nx.single_target_shortest_path_length
# nx.constraint                                       nx.is_pseudographical                               nx.small
# nx.contracted_edge                                  nx.is_semiconnected                                 nx.smetric
# nx.contracted_nodes                                 nx.is_simple_path                                   nx.social
# nx.convert                                          nx.is_strongly_connected                            nx.soft_random_geometric_graph
# nx.convert_matrix                                   nx.is_strongly_regular                              nx.sparse6
# nx.convert_node_labels_to_integers                  nx.is_tree                                          nx.spectral_layout
# nx.core                                             nx.is_valid_degree_sequence_erdos_gallai            nx.spectral_ordering
# nx.core_number                                      nx.is_valid_degree_sequence_havel_hakimi            nx.spectrum
# nx.coreviews                                        nx.is_valid_joint_degree                            nx.spring_layout
# nx.correlation                                      nx.is_weakly_connected                              nx.square_clustering
# nx.cost_of_flow                                     nx.is_weighted                                      nx.star_graph
# nx.could_be_isomorphic                              nx.isolate                                          nx.stochastic
# nx.covering                                         nx.isolates                                         nx.stochastic_graph
# nx.create_empty_copy                                nx.isomorphism                                      nx.stoer_wagner
# nx.cubical_graph                                    nx.jaccard_coefficient                              nx.strong_product
# nx.current_flow_betweenness                         nx.jit                                              nx.strongly_connected
# nx.current_flow_betweenness_centrality              nx.jit_data                                         nx.strongly_connected_component_subgraphs
# nx.current_flow_betweenness_centrality_subset       nx.jit_graph                                        nx.strongly_connected_components
# nx.current_flow_betweenness_subset                  nx.johnson                                          nx.strongly_connected_components_recursive
# nx.current_flow_closeness                           nx.join                                             nx.structuralholes
# nx.current_flow_closeness_centrality                nx.joint_degree_graph                               nx.subgraph
# nx.cut_size                                         nx.joint_degree_seq                                 nx.subgraph_alg
# nx.cuts                                             nx.json_graph                                       nx.subgraph_centrality
# nx.cycle_basis                                      nx.k_components                                     nx.subgraph_centrality_exp
# nx.cycle_graph                                      nx.k_core                                           nx.swap
# nx.cycles                                           nx.k_corona                                         nx.symmetric_difference
# nx.cytoscape                                        nx.k_crust                                          nx.tensor_product
# nx.cytoscape_data                                   nx.k_edge_augmentation                              nx.test
# nx.cytoscape_graph                                  nx.k_edge_components                                nx.tests
# nx.dag                                              nx.k_edge_subgraphs                                 nx.tetrahedral_graph
# nx.dag_longest_path                                 nx.k_nearest_neighbors                              nx.thresholded_random_geometric_graph
# nx.dag_longest_path_length                          nx.k_random_intersection_graph                      nx.to_dict_of_dicts
# nx.dag_to_branching                                 nx.k_shell                                          nx.to_dict_of_lists
# nx.davis_southern_women_graph                       nx.kamada_kawai_layout                              nx.to_directed
# nx.degree                                           nx.karate_club_graph                                nx.to_edgelist
# nx.degree_alg                                       nx.katz                                             nx.to_graph6_bytes
# nx.degree_assortativity_coefficient                 nx.katz_centrality                                  nx.to_nested_tuple
# nx.degree_centrality                                nx.katz_centrality_numpy                            nx.to_networkx_graph
# nx.degree_histogram                                 nx.kl_connected_subgraph                            nx.to_numpy_array
# nx.degree_mixing_dict                               nx.kosaraju_strongly_connected_components           nx.to_numpy_matrix
# nx.degree_mixing_matrix                             nx.krackhardt_kite_graph                            nx.to_numpy_recarray
# nx.degree_pearson_correlation_coefficient           nx.ladder_graph                                     nx.to_pandas_adjacency
# nx.degree_seq                                       nx.laplacian_matrix                                 nx.to_pandas_edgelist
# nx.degree_sequence_tree                             nx.laplacian_spectrum                               nx.to_prufer_sequence
# nx.dense                                            nx.laplacianmatrix                                  nx.to_scipy_sparse_matrix
# nx.dense_gnm_random_graph                           nx.lattice                                          nx.to_sparse6_bytes
# nx.density                                          nx.layout                                           nx.to_undirected
# nx.depth_first_search                               nx.leda                                             nx.topological_sort
# nx.desargues_graph                                  nx.lexicographic_product                            nx.tournament
# nx.descendants                                      nx.lexicographical_topological_sort                 nx.transitive_closure
# nx.dfs_edges                                        nx.linalg                                           nx.transitive_reduction
# nx.dfs_labeled_edges                                nx.line                                             nx.transitivity
# nx.dfs_postorder_nodes                              nx.line_graph                                       nx.traversal
# nx.dfs_predecessors                                 nx.link_analysis                                    nx.tree
# nx.dfs_preorder_nodes                               nx.link_prediction                                  nx.tree_all_pairs_lowest_common_ancestor
# nx.dfs_successors                                   nx.load                                             nx.tree_data
# nx.dfs_tree                                         nx.load_centrality                                  nx.tree_graph
# nx.diameter                                         nx.local_bridges                                    nx.trees
# nx.diamond_graph                                    nx.local_constraint                                 nx.triad_graph
# nx.difference                                       nx.local_efficiency                                 nx.triadic_census
# nx.digraph                                          nx.local_reaching_centrality                        nx.triads
# nx.dijkstra_path                                    nx.lollipop_graph                                   nx.triangles
# nx.dijkstra_path_length                             nx.lowest_common_ancestor                           nx.triangular_lattice_graph
# nx.dijkstra_predecessor_and_distance                nx.lowest_common_ancestors                          nx.trivial_graph
# nx.directed                                         nx.make_clique_bipartite                            nx.truncated_cube_graph
# nx.directed_configuration_model                     nx.make_max_clique_graph                            nx.truncated_tetrahedron_graph
# nx.directed_havel_hakimi_graph                      nx.make_small_graph                                 nx.turan_graph
# nx.directed_laplacian_matrix                        nx.margulis_gabber_galil_graph                      nx.tutte_graph
# nx.directed_modularity_matrix                       nx.matching                                         nx.unary
# nx.disjoint_union                                   nx.max_flow_min_cost                                nx.uniform_random_intersection_graph
# nx.disjoint_union_all                               nx.max_weight_matching                              nx.union
# nx.dispersion                                       nx.maximal_independent_set                          nx.union_all
# nx.distance_measures                                nx.maximal_matching                                 nx.unweighted
# nx.distance_regular                                 nx.maximum_branching                                nx.utils
# nx.dodecahedral_graph                               nx.maximum_flow                                     nx.vitality
# nx.dominance                                        nx.maximum_flow_value                               nx.volume
# nx.dominance_frontiers                              nx.maximum_spanning_arborescence                    nx.voronoi
# nx.dominating                                       nx.maximum_spanning_edges                           nx.voronoi_cells
# nx.dominating_set                                   nx.maximum_spanning_tree                            nx.watts_strogatz_graph
# nx.dorogovtsev_goltsev_mendes_graph                 nx.min_cost_flow                                    nx.waxman_graph
# nx.double_edge_swap                                 nx.min_cost_flow_cost                               nx.weakly_connected
# nx.draw                                             nx.min_edge_cover                                   nx.weakly_connected_component_subgraphs
# nx.draw_circular                                    nx.minimum_branching                                nx.weakly_connected_components
# nx.draw_kamada_kawai                                nx.minimum_cut                                      nx.weighted
# nx.draw_networkx                                    nx.minimum_cut_value                                nx.wheel_graph
# nx.draw_networkx_edge_labels                        nx.minimum_cycle_basis                              nx.wiener
# nx.draw_networkx_edges                              nx.minimum_edge_cut                                 nx.wiener_index
# nx.draw_networkx_labels                             nx.minimum_node_cut                                 nx.windmill_graph
# nx.draw_networkx_nodes                              nx.minimum_spanning_arborescence                    nx.within_inter_cluster
# nx.draw_random                                      nx.minimum_spanning_edges                           nx.write_adjlist
# nx.draw_shell                                       nx.minimum_spanning_tree                            nx.write_edgelist
# nx.draw_spectral                                    nx.minors                                           nx.write_gexf
# nx.draw_spring                                      nx.mis                                              nx.write_gml
# nx.drawing                                          nx.mixing                                           nx.write_gpickle
# nx.duplication                                      nx.mixing_dict                                      nx.write_graph6
# nx.duplication_divergence_graph                     nx.mixing_expansion                                 nx.write_graphml
# nx.eccentricity                                     nx.modularity_matrix                                nx.write_graphml_lxml
# nx.edge_betweenness                                 nx.modularity_spectrum                              nx.write_graphml_xml
# nx.edge_betweenness_centrality                      nx.modularitymatrix                                 nx.write_multiline_adjlist
# nx.edge_betweenness_centrality_subset               nx.moebius_kantor_graph                             nx.write_pajek
# nx.edge_boundary                                    nx.multi_source_dijkstra                            nx.write_shp
# nx.edge_connectivity                                nx.multi_source_dijkstra_path                       nx.write_sparse6
# nx.edge_current_flow_betweenness_centrality         nx.multi_source_dijkstra_path_length                nx.write_weighted_edgelist
# nx.edge_current_flow_betweenness_centrality_subset  nx.multidigraph                                     nx.write_yaml
# nx.edge_dfs                                         nx.multigraph                                       




#------------------------------------------------------------------------------------------------------------
# Explore scikit-learn clustering module 


# skc.AffinityPropagation      skc.MeanShift                skc.affinity_propagation_    skc.get_bin_seeds            skc.mean_shift_
# skc.AgglomerativeClustering  skc.MiniBatchKMeans          skc.bicluster                skc.hierarchical             skc.spectral
# skc.Birch                    skc.SpectralBiclustering     skc.birch                    skc.k_means                  skc.spectral_clustering
# skc.DBSCAN                   skc.SpectralClustering       skc.dbscan                   skc.k_means_                 skc.ward_tree
# skc.FeatureAgglomeration     skc.SpectralCoclustering     skc.dbscan_                  skc.linkage_tree             
# skc.KMeans                   skc.affinity_propagation     skc.estimate_bandwidth       skc.mean_shift   



#------------------------------------------------------------------------------------------------------------
# Explore scikit-learn clustering metrics module 


# skm.SCORERS                                skm.fbeta_score                            skm.pairwise_distances
# skm.accuracy_score                         skm.fowlkes_mallows_score                  skm.pairwise_distances_argmin
# skm.adjusted_mutual_info_score             skm.get_scorer                             skm.pairwise_distances_argmin_min
# skm.adjusted_rand_score                    skm.hamming_loss                           skm.pairwise_fast
# skm.auc                                    skm.hinge_loss                             skm.pairwise_kernels
# skm.average_precision_score                skm.homogeneity_completeness_v_measure     skm.precision_recall_curve
# skm.base                                   skm.homogeneity_score                      skm.precision_recall_fscore_support
# skm.brier_score_loss                       skm.jaccard_similarity_score               skm.precision_score
# skm.calinski_harabaz_score                 skm.label_ranking_average_precision_score  skm.r2_score
# skm.classification                         skm.label_ranking_loss                     skm.ranking
# skm.classification_report                  skm.log_loss                               skm.recall_score
# skm.cluster                                skm.make_scorer                            skm.regression
# skm.cohen_kappa_score                      skm.matthews_corrcoef                      skm.roc_auc_score
# skm.completeness_score                     skm.mean_absolute_error                    skm.roc_curve
# skm.confusion_matrix                       skm.mean_squared_error                     skm.scorer
# skm.consensus_score                        skm.mean_squared_log_error                 skm.silhouette_samples
# skm.coverage_error                         skm.median_absolute_error                  skm.silhouette_score
# skm.euclidean_distances                    skm.mutual_info_score                      skm.v_measure_score
# skm.explained_variance_score               skm.normalized_mutual_info_score           skm.zero_one_loss
# skm.f1_score                               skm.pairwise                               







#------------------------------------------------------------------------------------------------------------

# def cluster_metrics():
#     # Show country names that belong to each cluster
#     for i in range(numClust):
#         print('Cluster #',i)
#         print(countries.name[kmLabels==i])

#     # Compute cluster metrics
#     t = time.time()
#     CHS = skm.cluster.calinski_harabaz_score(Vi[0:nDims].T, kmLabels)
#     silh_score = skm.cluster.silhouette_samples(Vi[0:nDims].T, kmLabels, metric='euclidean')
#     silh_avg = skm.cluster.silhouette_score(Vi[0:nDims].T, kmLabels, metric='euclidean')
#     print('Time = ', time.time() - t)

#     print('CHS = ',CHS,' Silhouette Avg. = ',silh_avg)
#     print( 'Note: CHS & Silhouette seem to conflict eachother' )

#     # histogram of Silhouette Scores.
#     plt.hist(silh_score)
#     plt.show()