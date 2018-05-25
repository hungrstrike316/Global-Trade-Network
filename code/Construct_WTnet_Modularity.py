import numpy as np
import utils.data_manipulation as dm
import utils.network_manipulation as nm

sym = 'sym' # 'sym' if undirected or '' if directed.


dirPre = dm.set_dir_tree()
dirIn = str( dirPre + 'adjacency_ntwrk_npz_files/' )
num_countries  = 261


for year in range(1962,2015):
	try:
		adj, _, _ = dm.load_adjacency_npz_year(dirIn, year, num_countries, sym)
		B = nm.modularity(adj, sym)
		np.savez(dirPre + "/modularity_ntwrk_npz_files/" + sym + "modularity_ntwrk_" + str(year) + "_" + str(num_countries) + "countries.npz", B)
	except FileNotFoundError:
		print('File Not Found for year',year)
		pass
