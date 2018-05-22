from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def draw_map():
    ##  (1). Basemap figure, draws costlines of continents under network.
    #  Call map1.draw__ before plt.save or plt.draw to draw network on map.
    map1 = Basemap(projection='cyl')
    map1.drawmapboundary(fill_color='white')
    # map1.fillcontinents(color='coral', alpha=0.3, lake_color='aqua')
    map1.drawcoastlines()
    map1.drawcountries()



# -------------------------------- # -------------------------------- # --------------------------------



def plot_labels(H, tit, xlab, ylab, szfont):
    ## (2). Add title, axis labels, and adjust font size for figure in standard-ish format.
    plt.ylabel(ylab, fontsize=szfont)
    plt.xlabel(xlab, fontsize=szfont)
    plt.title(tit, fontsize=szfont)
    plt.xticks(fontsize=szfont)
    plt.yticks(fontsize=szfont)
    return H



# -------------------------------- # -------------------------------- # --------------------------------



def order_mag(x):
	## (3). Replace long numbers with many zeros with shorthand notation - for axis or colorbar labeling.
    tag = ['', 'k', 'M', 'B', 'T']  # thousand, million, billion, trillion
    cntr = 0
    while x/1000 > 1:
        cntr = cntr+1
        x = x//1000
    return str('$' + str(x) + ' ' + tag[cntr])

# -------------------------------- # -------------------------------- # --------------------------------



def axis_labels(axis, tit, xlab, ylab, xtick=None, xticklab=None, ytick=None, yticklab=None, szfont=12, titfont=12, grid=False):
    ## (4). Add title, axis labels and ticks for axis objects.
    # title and axis lable size separable parameters, set to matplotlib default; grid defualt to False
    axis.set_xlabel(xlab, fontsize=szfont)
    axis.set_ylabel(ylab, fontsize=szfont)
    axis.set_title(tit, fontsize=titfont)
    if xticklab and xtick:
        axis.set_xticks(xtick)
        axis.set_xticklabels(xticklab)
    if yticklab and xtick:
        axis.set_yticks(ytick)
        axis.set_yticklabels(yticklab)
    if grid:
        axis.grid()
