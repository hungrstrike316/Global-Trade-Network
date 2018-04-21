# MEH FOR NOW...


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