from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def draw_map():
    #  Basemap figure, draws costlines of continents under network.
    #  Call map1.draw__ before plt.save or plt.draw to draw network on map.
    map1 = Basemap(projection='cyl')
    map1.drawmapboundary(fill_color='white')
    # map1.fillcontinents(color='coral', alpha=0.3, lake_color='aqua')
    map1.drawcoastlines()
    map1.drawcountries()


def plot_labels(H, tit, xlab, ylab, szfont):
    # Add title, axis labels, and adjust font size for figure.
    plt.ylabel(ylab, fontsize=szfont)
    plt.xlabel(xlab, fontsize=szfont)
    plt.title(tit, fontsize=szfont)
    plt.xticks(fontsize=szfont)
    plt.yticks(fontsize=szfont)
    return H


def order_mag(x):
    tag = ['', 'k', 'M', 'B', 'T']  # thousand, million, billion, trillion
    cntr = 0
    while x/1000 > 1:
        cntr = cntr+1
        x = x//1000
    return str('$' + str(x) + ' ' + tag[cntr])
