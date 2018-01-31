import csv
import collections # iterator and counter libraries


with open("../MIT_WT_datafiles/country_names.csv", 'r') as cntry, open("../MIT_WT_datafiles/country_lat_lon_from_google.csv", 'r') as ll, open("../MIT_WT_datafiles/cntry_lat_lon_combined.csv", 'w') as output:
    
	reader = csv.reader(cntry) #,delimiter='\t') #... was a tsv file
	llread = csv.reader(ll)
	writer = csv.writer(output)
	next(reader)
	next(llread)

	writer.writerow(["id", "id_3char","name","latitude","longitude"])
	count = 0
	latlon = dict()


	for row in llread:
		print(row)
		latlon[row[3].casefold()]=(row[1],row[2]) # make a dictionary with country name as key - row[3].
												  # casefold makes all letters lowercase.

	for row in reader:
		if row[2].casefold() in set(latlon.keys()):
			#country_count[row[4]] += 1
			writer.writerow([ row[0].casefold(), row[1].casefold(), row[2].casefold(), latlon[row[2].casefold()][0], latlon[row[2].casefold()][1]] )
		else:
			writer.writerow([ row[0].casefold(), row[1].casefold(), row[2].casefold() ])
