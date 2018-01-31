class Cluster_Quality:
        from scipy.sparse import csr_matrix
        import numpy as np
        import math
        
        # instantiate class
        def __init__(self, adj, cluster):
            import numpy as np
            self.adj = adj
            self.cluster = cluster
            # total edges         
            self.m = adj.nnz
            # internal edges
            self.ein = 0
            # external edges 
            self.eout = 0
            # total cluster edges
            for i in cluster:
                for k in adj[i].nonzero()[1]:
                    if k in cluster:
                        self.ein += 1
                    else:
                        self.eout += 1

            self.ein = self.ein/2
            self.eout = self.eout
            self.etot = self.ein + self.eout
            #cluster size
            self.nc = len(cluster)
            # number of nodes
            self.n = adj.shape[0]
        
        # return density
        def density(self):
            import math
            #avoid division by zero
            if self.nc == 0 or self.nc == 1:
                dint = 0
            else:
                dint = self.ein/(self.nc * (self.nc - 1) / 2)        
            
            # avoid division by zero
            if self.nc == 0 or (self.n - self.nc) == 0:
                dext = math.inf
            else:
                dext = self.eout/(self.nc * (self.n - self.nc))
            
            return (dint, dext)




        # calculates overlap (assumes graph neighborhoods are closed)
        def overlap(self, sample_size = math.inf):
            from itertools import combinations
            import random
        
            if (sample_size < len(cluster)):
                sample = random.sample(cluster, sample_size)
            else:
                sample = cluster

            olap = list()
            for pair in combinations(sample, 2):
                intersect_size = len(np.intersect1d(adj[pair[0]].nonzero()[1], adj[pair[1]].nonzero()[1]))
                if adj[pair[0],pairs[1]] == 1:
                    intersect_size += 2
                mag1 = len(adj[pair[0]].nonzero()[1]) + 1
                mag2 = len(adj[pair[1]].nonzero()[1]) + 1
                olap.append(intersect_size / (mag1 + mag2 - intersect_size))
            return olap

        # calculates modularity
        def modularity (self):
            
            # sum of degrees of vertices in cluster
            dc = 2 * (self.ein + self.eout)

            return self.ein / self.m - dc / (2*self.m) * dc / (2*self.m)

        # calculates conductance of cluster
        def conductance (self):
            import math
            if self.ein == 0:
                return math.inf
            return self.eout/(self.ein * 2)
