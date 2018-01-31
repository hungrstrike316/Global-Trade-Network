import sys
import numpy as np
import pandas as pd
import pickle

#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from mpl_toolkits import axes_grid1



def set_dir_tree():
	# (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
	if sys.platform == 'darwin':
		print('on Mac OS - assuming Chris laptop')
		dirPre = '../'
	elif sys.platform == 'linux' or sys.platform == 'linux2':
		print('on Linux - assuming Cortex Cluster')
		dirPre = '../'
	return dirPre



def load_countries(dirPre):
	# (1). Load in country names & ids from a tsv file on MIT dataset.
	fcountries = str(dirPre + 'MIT_WT_datafiles/country_names.tsv')	
	countries = pd.read_table(fcountries, sep='\t')
	return countries




def load_products(dirPre):
	# (2). Load in product list from a tsv file on MIT dataset.
	fgoods = str(dirPre + 'MIT_WT_datafiles/products_sitc_rev2.tsv')	
	goods = pd.read_table(fgoods, sep='\t')
	return goods



def extract_years_from_origin_destination_csv(dirIn, dirOut, file):
	## (3) Load data file with quantities of goods traded between pairs of countries and Chop the big tsv file (~4GB and 700M rows) into
	# smaller ones for each year so I can handle them more efficiently. Python goes into big memory swap when using the whole thing.
	#
	# Dont need to run this every time. Only once in fact.
		fname = str(dirIn + file)
		trade = pd.read_table(fname, sep='\t')
		years = np.unique(trade.year)
		for y in range(0, len(years)):
			iY = trade['year'] == years[y]
			fout_name = str(dirOut + '/yr' + str(years[y]) + file[4:-3] +'.csv')
			df = trade[iY]
			df.to_csv(fout_name, sep='\t')


def construct_adjacency_from_year_origin_destination_csv(dirIn, dirOut, fileTag, year, num_countries):
## (4) Construct directed network (in an Adjacency matrix form) that shows goods shipped to and from each pair of
	# countries. There are two possible networks we can build in the data. This section convinces me they are equivalent.
	# (a). Exports from Origin to Destination. (trade_ntwrkExp) - transpose of import matrix (not calculating anymore.)
	# (b). Imports from Origin to Destination. (trade_ntwrkImp)
	#
	# While this technically works, it is very slow. How to speed it up?
	# Dont need to run this every time. Only once in fact.
	#
	# (CAN I DO THIS MORE QUICKLY / EFFICIENTLY IF I CONVERT CSV INTO A PICKLE FILE FIRST?)

	trade_ntwrkImp = np.zeros( (num_countries, num_countries) )
	#trade_ntwrkExp = np.zeros( (num_countries, num_countries) )
	
	tradeCountry_import = np.zeros( num_countries )
	tradeCountry_export = np.zeros( num_countries )

	ftradeYr = str(dirIn + 'yr' + str(year) + fileTag)
	tradeYr = pd.read_table(ftradeYr, sep='\t')
	print( 'year = ' + str(year) )
			
	for o in range(0, num_countries):
		iO = tradeYr['origin'].str.strip() == countries.id_3char[o]
		tradeCountry_import[o] = ( np.sum(tradeYr[iO].import_val) )
		tradeCountry_export[o] = ( np.sum(tradeYr[iO].export_val) )
		print([ str(o) + " / " + str(a) + " : " + countries.id_3char[o] ])
				
		for d in range(0, num_countries):
			iD = tradeYr['dest'].str.strip() == countries.id_3char[d]
			#trade_ntwrkExp[o,d] = ( np.sum(tradeYr[iO & iD].export_val) )
			trade_ntwrkImp[o,d] = ( np.sum(tradeYr[iO & iD].import_val) )
			
	# Save a file with the adjacency matrix (trade_ntwrkImp), the total import (tradeCountry_import) and export of each country (tradeCountry_export):
	np.savez(str(dirOut + 'adjacency_ntwrk_' + str(years) + '_' + str(num_countries) + 'countries.npz'), 
		netwrk = trade_ntwrkImp, imprt = tradeCountry_import, exprt = tradeCountry_export)





def load_adjacency_npz_year(dirIn, year, num_countries):
	# (5). Load a previously saved adjacency matrix files:
	loaded = np.load(str(dirIn + 'adjacency_ntwrk_' + str(year) + '_' + str(num_countries) + 'countries.npz'))
	trade_ntwrkA = loaded['netwrk']
	tradeCountry_import = loaded['imprt']
	tradeCountry_export = loaded['exprt']
	loaded.close()
	#
	trade_ntwrkA = trade_ntwrkA[:-2,:-2]            # get rid of worlds & areas.
	tradeCountry_import = tradeCountry_import[:-2]
	tradeCountry_export = tradeCountry_export[:-2]
	#
	return trade_ntwrkA, tradeCountry_import, tradeCountry_export



def load_lat_lon_pickle(file):
	with open( file, 'rb') as handle:
		lat_lon = pickle.load(handle)
	lat_lon.pop()
	lat_lon.pop() # pop twice to get rid of world & areas	
	return lat_lon