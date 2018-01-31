
# This function takes data files provided online (see link below) and turn it into Adjacency matrices,
# one for each year.
#
# syntax: >> python Construct_WTnet_Adjacency.py 1972 <- (argument is some year between 1962 and 2014)
#
# Data files about world trade statistics from https://atlas.media.mit.edu/de/resources/data/
#  (1). year_origin_destination_sitc_rev2.tsv - contains the value of trade between all country pairs,
#       for all products and all year between 1962 and 2014:
#       						{ year	origin	dest	sitc	export_val	import_val }
#  (2). country_names.tsv - a glossary for all countries:   { id	id_3char	name }
#  (3). products_sitc_rev2.tsv - glossary for all products: { id	sitc	name }



# Import a bunch of python packages that I use below.
import numpy as np
import pandas as pd
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions	


# Define the function to construct a trade network adjacency matrix for single year. Run this in a for loop below
def Construct_WTnet_Adjacency(year):	

	dirPre = dm.set_dir_tree()

	## (1) Load country names that align with 3 letter acronyms used in origin destination file
	countries = dm.load_countries(dirPre)
	
	## (2) Load in names and codes for types of goods traded
	# goods = dm.load_products(dirPre)


	## (3) Load data file with quantities of goods traded between pairs of countries and Chop the big tsv file (~4GB and 700M rows) into
	# smaller ones for each year so I can handle them more efficiently. Python goes into big memory swap when using the whole thing.
	#
	# Dont need to run this every time. Only once in fact.
	if False:
		dirIn = str(dirPre + 'MIT_WT_datafiles/')
		dirOut = str(dirPre + 'origin_destination_csvs_byYear/')
		file = 'year_origin_destination_sitc_rev2.tsv'
		#
		extract_year_from_origin_destination_csv(dirIn, dirOut, file)


	## (4) Construct directed network (in an Adjacency matrix form) that shows goods shipped to and from each pair of
	# countries. There are two possible networks we can build in the data. This section convinces me they are equivalent.
	# (a). Exports from Origin to Destination. (trade_ntwrkExp)
	# (b). Imports from Origin to Destination. (trade_ntwrkImp)
	#
	# While this technically works, it is very slow. How to speed it up?
	# Dont need to run this every time. Only once in fact.
	if True:
		dirIn = str(dirPre + 'origin_destination_csvs_byYear/')		
		dirOut = str(dirPre + 'adjacency_ntwrk_npz_files/')
		fileTag = '_origin_destination_sitc_rev2.csv'
		#year = {This Variable Passed into Construct_WTnet_Adjacency function}
		#year = range(1962,2014) # this is input into function now as sys.argv[0] !
		#
		try:
			num_countries = np.size(countries,0)
		except:
			num_countries = 263 # hard coded if countries vector has not been loaded in.
		#
		construct_adjacency_matrix_from_year_origin_destination_csv(dirIn, dirOut, fileTag, year, num_countries)


# This bit here makes it so you can call this as a function from the command line with the year as an input argument.
# Call this from the command line like:
# 	>> python3 Construct_WTnet_Adjacency.py 1964
# On cluster you may have to load python module first, like:
# 	>> module load python/anaconda3
if __name__ == "__main__":
	Construct_WTnet_Adjacency(sys.argv[1:4]) # this counts on the input being only 4 characters, a year - like '1984'.

