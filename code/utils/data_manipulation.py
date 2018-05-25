import sys
import numpy as np
import pandas as pd
import pickle



def set_dir_tree():
	# (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
	if sys.platform == 'darwin':
		print('on Mac OS - assuming Chris laptop')
		dirPre = '../'
	elif sys.platform == 'linux' or sys.platform == 'linux2':
		print('on Linux - assuming Cortex Cluster')
		dirPre = '../'
	return dirPre




# -------------------------------- # -------------------------------- # --------------------------------



def load_country_lat_lon_csv(dirPre):
	## (7). Load in csv/tsv with country names, lat & lon.
	fLL = str(dirPre + 'MIT_WT_datafiles/cntry_lat_lon_combined_fin_UTF8.csv')
	countriesLL = pd.read_table(fLL,sep=',')
	countriesLL = countriesLL.drop(262)
	countriesLL = countriesLL.drop(261) # get rid of 'world' and 'areas'
	countriesLL = countriesLL.rename(index=str, columns={'\ufeffid':'id'}  ) # reset this weird key name as just id.

	return countriesLL

# -------------------------------- # -------------------------------- # --------------------------------



def load_products(dirPre):
	## (2). Load in product list from a tsv file on MIT dataset.
	fgoods = str(dirPre + 'MIT_WT_datafiles/products_sitc_rev2.tsv')
	goods = pd.read_table(fgoods, sep='\t')
	return goods


# -------------------------------- # -------------------------------- # --------------------------------



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


# -------------------------------- # -------------------------------- # --------------------------------




def construct_adjacency_from_year_origin_destination_csv(dirIn, dirOut, fileTag, year, countries):
	## (4) Construct directed network (in an Adjacency matrix form) that shows goods shipped to and from each pair of
	# countries. There are two possible networks we can build in the data. This section convinces me they are equivalent.
	# (a). Exports from Origin to Destination. (trade_ntwrkExp) - transpose of import matrix (not calculating anymore.)
	# (b). Imports from Origin to Destination. (trade_ntwrkImp)
	#
	# While this technically works, it is very slow. How to speed it up?
	# Dont need to run this every time. Only once in fact.
	#
	# (CAN I DO THIS MORE QUICKLY / EFFICIENTLY IF I CONVERT CSV INTO A PICKLE FILE FIRST?)

	num_countries = countries.shape[0]
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
		print([ str(o) + " / " + str(num_countries) + " : " + countries.id_3char[o] ])

		for d in range(0, num_countries):
			iD = tradeYr['dest'].str.strip() == countries.id_3char[d]
			#trade_ntwrkExp[o,d] = ( np.sum(tradeYr[iO & iD].export_val) )
			trade_ntwrkImp[o,d] = ( np.sum(tradeYr[iO & iD].import_val) )

	# Save a file with the adjacency matrix (trade_ntwrkImp), the total import (tradeCountry_import) and export of each country (tradeCountry_export):
	np.savez(str(dirOut + 'adjacency_ntwrk_' + str(year) + '_' + str(num_countries) + 'countries.npz'),
		netwrk = trade_ntwrkImp, imprt = tradeCountry_import, exprt = tradeCountry_export)



# -------------------------------- # -------------------------------- # --------------------------------



def load_adjacency_npz_year(dirIn, year, num_countries, sym):
	## (5). Load a previously saved adjacency matrix files:
	loaded = np.load(str(dirIn + 'adjacency_ntwrk_' + str(year) + '_' + str(num_countries) + 'countries.npz'))
	trade_ntwrk = loaded['netwrk']
	imports = loaded['imprt']
	exports = loaded['exprt']
	loaded.close()
	#
	# Explicitly set any non-zero on the diagonal of Adjacency to zero. 
	# They do not make sense (trade between a country and itself).
	# And they are actally in the csv data file.
	self_weights = trade_ntwrk.diagonal().nonzero()[0]
	imports[self_weights] = imports[self_weights] - trade_ntwrk.diagonal()[self_weights] 
	exports[self_weights] = exports[self_weights] - trade_ntwrk.diagonal()[self_weights] 
	trade_ntwrk[ (self_weights,self_weights) ] = 0
	#
	if (sym=='sym'):
		trade_ntwrk = (trade_ntwrk + trade_ntwrk.T) / 2
		IE_avg  = (imports + exports) / 2
		imports = IE_avg
		exports = IE_avg
	#
	# output results as integers
	trade_ntwrk = trade_ntwrk.astype(int)
	imports =  imports.astype(int)
	exports =  exports.astype(int)
	#
	return trade_ntwrk, imports, exports


# -------------------------------- # -------------------------------- # --------------------------------


def load_modularity_npz_year(dirIn, year, num_countries, sym):
	## (6). Load a previously saved modularity matrix files:
	loaded = np.load(str(dirIn + sym + 'modularity_ntwrk_' + str(year) + '_' + str(num_countries) + 'countries.npz'))
	trade_ntwrkA = loaded['arr_0'] # loaded['netwrk']
	loaded.close()

	return trade_ntwrkA



# -------------------------------- # -------------------------------- # --------------------------------


# # Doing this in 'load_country_lat_lon_csv' too.
# def load_lat_lon_pickle(file):
# 	## (7). Load in pickle file that we saved from csv/tsv with country names, lat & lon.
# 	with open( file, 'rb') as handle:
# 		lat_lon = pickle.load(handle)
# 	lat_lon.pop()
# 	lat_lon.pop() # pop twice to get rid of world & areas
# 	return lat_lon

# -------------------------------- # -------------------------------- # --------------------------------

# Doing this now in 'load_country_lat_lon_csv' above.
# def load_countries(dirPre):
# 	## (1). Load in country names & ids from a tsv file on MIT dataset.
# 	fcountries = str(dirPre + 'MIT_WT_datafiles/country_names.tsv')
# 	countries = pd.read_table(fcountries, sep='\t')
# 	countries = countries.drop(262)
# 	countries = countries.drop(261) # like pop to get rid of 'world' and 'areas'
# 	return countries



