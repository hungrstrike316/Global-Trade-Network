import numpy as np
import pandas as pd
import sys
import networkx as nx
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions

import matplotlib.pyplot as plt


def modularity_comparison(year):
    ntwrkX_mod_mtrx = np.load("../modularity_ntwrkX_npz_files/modularity_ntwrkX_" + str(year) +
                  "_261countries.npz")
    our_mod_mtrx = np.load( "../modularity_ntwrk_npz_files/modularity_ntwrk_" + str(year) +
                  "_261countries.npz")
    mtrx_A = ntwrkX_mod_mtrx['netwrk'] # ORDERS OF MAGNITUDE LARGER !!
    mtrx_B = our_mod_mtrx['arr_0']
    diff_mtrx =  np.divide(mtrx_A, mtrx_B) # np.subtract(mtrx_A, mtrx_B) #
    np.savez(("../modularity_difference_files/modularity_diff_" + str(year) +"_"
             "261countries.npz"), difference=diff_mtrx)

    print(diff_mtrx)
    print( diff_mtrx.nonzero()[0].shape )
    print( mtrx_B.nonzero()[0].shape )

    plt.imshow(diff_mtrx) # 
    plt.colorbar()
    plt.show()
    #plt.ylabel(ylab, fontsize=szfont)
    #plt.xlabel(xlab, fontsize=szfont)
    #plt.title(tit, fontsize=szfont)
    #plt.xticks(fontsize=szfont)
    #plt.yticks(fontsize=szfont)

    return diff_mtrx

for i in range(1962, 1963): # range(1962, 2015):
    try:
        modularity_comparison(i)

    except FileNotFoundError:
        print('uuh')
        pass


