#========================================================
# Create a pickle from a csv file.
#========================================================

import csv
import collections
import pickle


with open("../MIT_WT_datafiles/cntry_lat_lon_combined_fin_UTF8.csv" , 'r') as f:
    reader = csv.reader(f)
    next(reader)
    cll_info = [(i[0], i[1], i[2], i[3], i[4]) for i in reader]

with open("../MIT_WT_datafiles/cntry_lat_lon_combined_fin_UTF8.pickle", 'wb') as handle:
    pickle.dump(cll_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

