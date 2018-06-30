import numpy as np


def is_symmetric(adjacency):
    """

    Args:
        adjacency (numpy.ndarray): An adjacency matrix

    Returns:
        bool

    """
    """
    Returns True if an adjacency matrix is symmetric, False otherwise.
    """
    return (adjacency.T == adjacency).all()


def averaged_unsymmetric_adjacency(adjacency):
    """

    Args:
        adjacency (numpy.ndarray):

    Returns:
        numpy.ndarray

    """
    adjacency = adjacency.astype(np.float32)
    n = adjacency.shape[0]  # Number of rows
    # for j in range(n):
    #     for i in range(j + 1, n):
    #         adjacency[i][j] = (adjacency[i][j] + adjacency[j][i]) / 2
    for j in range(n):
        for i in range(n):
            adjacency[i][j] = (adjacency[i][j] + adjacency[j][i]) / 2
            adjacency[j][i] = adjacency[i][j]
    return adjacency


# -----------------------------------------------------------------------------

def modularity(g, clusters):
    """
    This is the same as NetworkX Modularity; however, the docstring of
    modularity function mistakingly says that it is using "2 * m" as the 
    denominator of the equation, when in fact that term is the sum of all the
    weights in the adjacency matrix.

    """
    summed = 0
    edges = g.edges
    # two_m = 2 * len(edges) ***this would be wrong***
    two_m = g.size('weight')
    for cluster in clusters:
        for edge in edges:
            if edge[0] in cluster and edge[1] in cluster:
                summed += g.get_edge_data(edge[0], edge[1])['weight'] - \
                          ((g.degree(edge[0], 'weight') * g.degree(edge[1],
                                                                   'weight')) /
                           two_m)
    return (1 / two_m) * summed

# -----------------------------------------------------------------------------


def conductance(g, clusters):
    """Calculates the conductance quality measure.

    Args:
        g (networkx.classes.graph.Graph): NetworkX graph
        clusters (list): List of sets

    Returns:
        float

    """
    measure = 0
    edges = g.edges
    for cluster in clusters:
        cut_size = 0
        total_degree_0 = 0
        total_degree_1 = 0
        flag = False  # Checks to see if we find any connections or not
        for edge in edges:
            node_0, node_1 = edge[0], edge[1]
            if node_0 in cluster and node_1 not in cluster:
                flag = True
                cut_size += g.get_edge_data(node_0, node_1)['weight']
                total_degree_0 += g.degree(node_0, 'weight')
                total_degree_1 += g.degree(node_1, 'weight')
        if not flag:
            continue
        min_total_degrees = min(total_degree_0, total_degree_1)
        measure += cut_size / min_total_degrees
    return measure

# -----------------------------------------------------------------------------


def density(g, clusters):
    """Calculates the density quality measure.

    Args:
        g (networkx.classes.graph.Graph): NetworkX graph
        clusters (list): List of sets.

    Returns:
        float

    """
    measure = 0
    edges = g.edges
    n = g.number_of_nodes()

    # vectors for testing
    n_c_vec = [] #
    weighted_internal_vec = [] #
    weighted_external_vec = [] #
    density_internal_vec = [] #
    density_external_vec = [] #
    for cluster in clusters:
        weighted_internal = 0
        weighted_external = 0
        n_c = len(cluster)
        n_c_vec.append(n_c) #
        if n == n_c:
            return 0
        for edge in edges:
            node_0, node_1 = edge[0], edge[1]
            if node_0 in cluster and node_1 in cluster:
                weighted_internal_vec.append(weighted_internal) #
                weighted_internal += g.get_edge_data(node_0, node_1)['weight']
            if node_0 in cluster and node_1 not in cluster:
                weighted_external_vec.append(weighted_external) #
                weighted_external += g.get_edge_data(node_0, node_1)['weight']
        density_external = weighted_external / (n_c * (n - n_c))
        density_external_vec.append(density_external) #
        if n_c is 1:
            return -density_external
        density_internal = weighted_internal / ((n_c * (n_c - 1)) / 2)
        density_internal_vec.append(density_internal) #
        measure += density_internal - density_external
    return measure
