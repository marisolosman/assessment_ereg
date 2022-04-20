import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import glob
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.patches as patches
plt.ioff()


def plot_skill_uncal(lat, lon, hss, rpss, rel_comp, roc_comp, titulo, output):
    #funcion para graficar scores (ergo barra entre -1 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (9.7, 9.7), 300)
    #figure HSS
    ax = plt.subplot(2, 2, 1, projection=mapproj)
    #projection and map limits
    ax.set_extent(limits, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5)
    # set desired contour levels.
    clevs = np.arange(0, 120, 20)
    barra = plt.cm.get_cmap('OrRd', 5) #colorbar

    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    CS1 = ax.pcolor(dx, dy, hss, cmap = barra, vmin = 0, vmax = 100,
                    transform=ccrs.PlateCarree())
    ax.add_patch(patches.Rectangle(xy=[298, -22], height=-15, width=15, linewidth=1, edgecolor='black',
                 facecolor='none', transform=ccrs.PlateCarree()))
    ax.add_patch(patches.Rectangle(xy=[295, 10], height=-12, width=35, linewidth=1, edgecolor='black',
                 facecolor='none', transform=ccrs.PlateCarree()))
    ax.plot([275, 330], [-20, -20], color='black', linewidth=1, linestyle='--', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(CS1,orientation='vertical', ticks = clevs)
    cbar.ax.tick_params(labelsize = 8)
    plt.title('Heidke Skill Score', fontsize=10, loc='left')
    #figure RPSS
    ax = plt.subplot(2, 2, 2, projection=mapproj)
    #projection and map limits
    ax.set_extent(limits, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    # set desired contour levels.
    clevs = np.linspace(-1.1, 1.1 , 13)
    barra = plt.cm.get_cmap('coolwarm',11) #colorbar

    CS1 = ax.pcolor(dx, dy, rpss, cmap = barra, vmin = -1.1, vmax = 1.1,
                    transform=ccrs.PlateCarree())
    ax.add_patch(patches.Rectangle(xy=[298, -22], height=-15, width=15, linewidth=1, edgecolor='black',
                 facecolor='none', transform=ccrs.PlateCarree()))
    ax.add_patch(patches.Rectangle(xy=[295, 10], height=-12, width=35, linewidth=1, edgecolor='black',
                 facecolor='none', transform=ccrs.PlateCarree()))
    ax.plot([275, 330], [-20, -20], color='black', linewidth=1, linestyle='--', transform=ccrs.PlateCarree())

    cbar = fig.colorbar(CS1, orientation='vertical', ticks = np.arange(-0.9, 1.1, 0.2))
    cbar.ax.tick_params(labelsize = 8)
    plt.title('RPSS', fontsize=10, loc='left')
    #Reliability diagram
    ax = plt.subplot(2, 2, 3)
    ax.plot(np.arange(0.05, 1.05, 0.1), rel_comp['hrrd_above'], color='b', marker='o')
    ax.plot(np.arange(0.05, 1.05, 0.1), rel_comp['hrrd_below'], color='r', marker='o')
    ax.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12), color='k')
    ax.axis([0, 1, 0, 1])
    ax.set_xlabel('Mean Forecast Probability', fontsize=9)
    ax.set_ylabel( 'Observed Relative Frequency', fontsize=9)
    plt.title('Reliability diagram', fontsize=10, loc='left')
    ##histograms 
    ax2 = ax.twinx()
    ax2.plot(np.arange(0.05, 1.05, 0.1), rel_comp['frd_above'], color='b', linestyle=':')
    ax2.axis([0, 1, 0, 0.50])
    ax2.set_yticks(np.arange(0, 0.6, 0.1))
    ax2.set_ylabel('Forecast relative frequency')
    ax2.plot(np.arange(0.05, 1.05, 0.1), rel_comp['frd_below'], color = 'r', linestyle=':')
    #ROC Diagram
    ax = plt.subplot(2, 2, 4)
    ax.plot(roc_comp['farr_above'], roc_comp['hrr_above'], color = 'b',
            marker = 'o')
    ax.plot(roc_comp['farr_below'], roc_comp['hrr_below'], color = 'r',
            marker = 'o')
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for j in np.arange(1,np.shape(roc_comp['farr_above'])[0]):
        ax.text(roc_comp['farr_above'][j] + 0.01, roc_comp['hrr_above'][j] + 0.01,
                str(bins[j]), color = 'k', fontsize=7)
        ax.text(roc_comp['farr_below'][j] + 0.01, roc_comp['hrr_below'][j] + 0.01,
                str(bins[j]), color = 'k', fontsize=7)
        ax.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12), color = 'k')
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('False Alarm Rate', fontsize=9)
        ax.set_ylabel('Hit Rate', fontsize=9)
        ley1 = 'above: '+ '{:.3f}'.format(float(roc_comp['roc_above'])/2 + 0.5)
        ley2 = 'below: '+ '{:.3f}'.format(float(roc_comp['roc_below'])/2 + 0.5)
        ax.legend([ley1, ley2], loc='lower right')
        plt.title('ROC Diagram', fontsize=10, loc='left')
    plt.suptitle(titulo, fontsize=13, x=0.51, y=0.9)
    fig.subplots_adjust(top=0.84, hspace=0.1, wspace=0.4)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.clf()
    plt.cla()
    plt.close('all')
    return


def plot_scores(lat, lon, var, titulo, output):
    #funcion para graficar scores (ergo barra entre -1 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    # set desired contour levels.
    clevs = np.linspace(-1.1,1.1 , 13)
    barra = plt.cm.get_cmap('coolwarm',11) #colorbar
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (9.7, 9.7), 300)
    keys = sorted(var.keys())
    index = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1, projection=mapproj)
        #projection and map limits
        ax.set_extent(limits, crs=ccrs.PlateCarree())
        ax.coastlines(alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        CS1 = ax.pcolor(dx, dy, var[keys[i]], cmap = barra, vmin = -1.1, vmax = 1.1,
                        transform=ccrs.PlateCarree())
        # ax.add_patch(patches.Rectangle(xy=[298, -22], height=-15, width=15, linewidth=1, edgecolor='black',
        #         facecolor='none', transform=ccrs.PlateCarree()))
        # ax.add_patch(patches.Rectangle(xy=[295, 10], height=-12, width=35, linewidth=1, edgecolor='black',
        #                               facecolor='none', transform=ccrs.PlateCarree()))
        # ax.plot([275, 330], [-20, -20], color='black', linewidth=1, linestyle='--',
        #        transform=ccrs.PlateCarree())
        plt.title(index[i] + ' ' + keys[i].upper(), fontsize=8, loc='left')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(bottom=0.17, top=0.82, hspace=0.1)
    cbar_ax = fig.add_axes([0.19, 0.1, 0.65, 0.02])
    cbar = fig.colorbar(CS1, cax=cbar_ax, orientation='horizontal',
                        ticks = np.linspace(-0.9, 0.9, 10))
    cbar.ax.tick_params(labelsize = 8)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.clf()
    plt.cla()
    plt.close('all')
    return
def plot_bs(lat, lon, var, titulo, output):
    #funcion para graficar scores (ergo barra entre -1 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    # set desired contour levels.
    clevs = np.arange(0, 0.25, 0.05)
    barra = plt.cm.get_cmap('OrRd_r', 4) #colorbar
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (9.7, 9.7), 300)
    keys = sorted(var.keys())
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1, projection=mapproj)
        #projection and map limits
        ax.set_extent(limits, crs=ccrs.PlateCarree())
        ax.coastlines(alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        CS1 = ax.pcolor(dx, dy, var[keys[i]], cmap = barra, vmin = 0, vmax = 0.20,
                        transform=ccrs.PlateCarree())
        plt.title(keys[i], fontsize=8)
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(bottom=0.17, top=0.82, hspace=0.1)
    cbar_ax = fig.add_axes([0.19, 0.1, 0.65, 0.02])
    cbar = fig.colorbar(CS1, cax=cbar_ax, orientation='horizontal',
                        ticks = np.arange(0, 0.20, 0.05))
    cbar.ax.tick_params(labelsize = 8)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.clf()
    plt.cla()
    plt.close('all')
    return
def plot_hss(lat, lon, var, titulo, output):
    #funcion para graficar scores (ergo barra entre -1 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    # set desired contour levels.
    clevs = np.arange(0, 120, 20)
    barra = plt.cm.get_cmap('OrRd', 5) #colorbar
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (9.7, 9.7), 300)
    keys = sorted(var.keys())
    index = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1, projection=mapproj)
        #projection and map limits
        ax.set_extent(limits, crs=ccrs.PlateCarree())
        ax.coastlines(alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        CS1 = ax.pcolor(dx, dy, var[keys[i]], cmap = barra, vmin = 0, vmax = 100,
                        transform=ccrs.PlateCarree())
        ax.add_patch(patches.Rectangle(xy=[298, -22], height=-15, width=15, linewidth=1, edgecolor='black',
                 facecolor='none', transform=ccrs.PlateCarree()))
        ax.add_patch(patches.Rectangle(xy=[295, 10], height=-12, width=35, linewidth=1, edgecolor='black',
                                       facecolor='none', transform=ccrs.PlateCarree()))
        ax.plot([275, 330], [-20, -20], color='black', linewidth=1, linestyle='--',
                transform=ccrs.PlateCarree())
        plt.title(index[i] + ' ' + keys[i].upper(), fontsize=8, loc='left')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(bottom=0.17, top=0.82, hspace=0.1)
    cbar_ax = fig.add_axes([0.19, 0.1, 0.65, 0.02])
    cbar = fig.colorbar(CS1, cax=cbar_ax, orientation='horizontal',
                        ticks = np.arange(0, 120, 20))
    cbar.ax.tick_params(labelsize = 8)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.clf()
    plt.cla()
    plt.close()
    return

def plot_bs_dec(lat, lon, var, titulo, output):
    #funcion para graficar scores (ergo barra entre -1 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    # set desired contour levels.
    clevs = np.arange(0, 0.15, 0.025)
    barra = plt.cm.get_cmap('OrRd_r', 5) #colorbar
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (9.7, 9.7), 300)
    keys = sorted(var.keys())
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1, projection=mapproj)
        #project1on and map limits
        ax.set_extent(limits, crs=ccrs.PlateCarree())
        ax.coastlines(alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        CS1 = ax.pcolor(dx, dy, var[keys[i]], cmap = barra, vmin = 0, vmax = 0.125,
                        transform=ccrs.PlateCarree())
        plt.title(keys[i], fontsize=8)
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(bottom=0.17, top=0.82, hspace=0.1)
    cbar_ax = fig.add_axes([0.19, 0.1, 0.65, 0.02])
    cbar = fig.colorbar(CS1, cax=cbar_ax, orientation='horizontal',
                        ticks = np.arange(0, 0.125, 0.025))
    cbar.ax.tick_params(labelsize = 8)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.clf()
    plt.cla()
    plt.close()
    return

def plot_roc(farr_above,hrr_above, farr_below, hrr_below, roc_above, roc_below, bins,
             titulo, output):
    fig = plt.figure(1, (15.7, 9.7), 600)
    keys = sorted(farr_above.keys())
    index = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(farr_above[keys[i]], hrr_above[keys[i]], color = 'b',
                marker = 'o')
        ax.plot(farr_below[keys[i]], hrr_below[keys[i]], color = 'r',
                marker = 'o')
        for j in np.arange(1,np.shape(farr_above[keys[i]])[0]):
            ax.text(farr_above[keys[i]][j] + 0.01,hrr_above[keys[i]][j] + 0.01,
                     str(bins[j]), color = 'k', fontsize=7)
            ax.text(farr_below[keys[i]][j] + 0.01,hrr_below[keys[i]][j] + 0.01,
                     str(bins[j]), color = 'k', fontsize=7)
        ax.plot(np.linspace(0,1.1,12), np.linspace(0,1.1,12), color = 'k')
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('False Alarm Rate', fontsize=9)
        ax.set_ylabel('Hit Rate', fontsize=9)
        ley1 = 'above: '+ '{:.3f}'.format(float(roc_above[keys[i]]) / 2 + 0.5)
        ley2 = 'below: '+ '{:.3f}'.format(float(roc_below[keys[i]]) / 2 + 0.5)
        ax.legend([ley1, ley2], loc='lower right')
        plt.title(index[i] + ' ' + keys[i].upper(), fontsize=10, loc='left')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(top=0.85, hspace=0.25)
    plt.savefig(output, dpi = 600, bbox_inches = 'tight', papertype = 'A4')
    plt.clf()
    plt.cla()
    plt.close('all')
    return

def plot_roc_above_leadtimes(farr_above, hrr_above, farr_below, hrr_below, roc_above,
                       roc_below, bins, titulo, output):
    fig = plt.figure(1, (15.7, 9.7), 300)
    keys = sorted(farr_above.keys())
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(farr_above[keys[i]], hrr_above[keys[i]], color = 'darkblue',
                marker = 'o')
        ax.plot(farr_above[keys[i]], hrr_above[keys[i + 12]], color = 'royalblue',
                marker = 'o')
        ax.plot(farr_above[keys[i]], hrr_above[keys[i + 6]], color = 'deepskyblue',
                marker = 'o')
        #ax.plot(farr_below[keys[i]], hrr_below[keys[i]], color = 'darkred',
        #        marker = 'o')
        #ax.plot(farr_below[keys[i]], hrr_below[keys[i + 6]], color = 'tomato',
        #        marker = 'o')
        #ax.plot(farr_below[keys[i]], hrr_below[keys[i + 12]], color = 'coral',
        #        marker = 'o')
        for j in np.arange(1,np.shape(farr_above[keys[i]])[0]):
            ax.text(farr_above[keys[i]][j] + 0.01,hrr_above[keys[i]][j] + 0.01,
                     str(bins[j]), color = 'k', fontsize=6)
           # ax.text(farr_below[keys[i]][j] + 0.01,hrr_below[keys[i]][j] + 0.01,
           #          str(bins[j]), color = 'k', fontsize=6)
            ax.text(farr_above[keys[i + 6]][j] + 0.01,hrr_above[keys[i + 6]][j] + 0.01,
                     str(bins[j]), color = 'k', fontsize=6)
           # ax.text(farr_below[keys[i + 6]][j] + 0.01,hrr_below[keys[i + 6]][j] + 0.01,
           #          str(bins[j]), color = 'k', fontsize=6)
            ax.text(farr_above[keys[i + 12]][j] + 0.01,hrr_above[keys[i + 12]][j] + 0.01,
                     str(bins[j]), color = 'k', fontsize=6)
           # ax.text(farr_below[keys[i + 12]][j] + 0.01,hrr_below[keys[i + 12]][j] + 0.01,
           #          str(bins[j]), color = 'k', fontsize=6)
        ax.plot(np.linspace(0,1.1,12), np.linspace(0,1.1,12), color = 'k')
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('False Alarm Rate', fontsize=9)
        ax.set_ylabel('Hit Rate', fontsize=9)
        plt.title(keys[i][3:], fontsize=10, loc='left')

    fig.subplots_adjust(top=0.85, bottom=0.3, hspace=0.32)
    ley1 = 'lead 1 month'
    #ley2 = 'below: lead 1'
    ley3 = 'lead 4 months'
    #ley4 = 'below: lead 4'
    ley5 = 'lead 7 months'
    #ley6 = 'below: lead 7'
    ax.legend([ley1, ley3, ley5], bbox_to_anchor=(0, -0.35),
               loc='center right', ncol=3)
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    plt.savefig(output, dpi = 300, bbox_inches = 'tight', papertype = 'A4')
    plt.close()
    plt.clf()
    plt.cla()

def plot_roc_below_leadtimes(farr_above, hrr_above, farr_below, hrr_below, roc_above,
                             roc_below, bins, titulo, output):
    fig = plt.figure(1, (15.7, 9.7), 300)
    keys = sorted(farr_above.keys())
    print(keys)
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(farr_below[keys[i]], hrr_below[keys[i]], color = 'darkred',
                marker = 'o')
        ax.plot(farr_below[keys[i]], hrr_below[keys[i + 12]], color = 'orangered',
                marker = 'o')
        ax.plot(farr_below[keys[i]], hrr_below[keys[i + 6]], color = 'salmon',
                marker = 'o')
        for j in np.arange(1,np.shape(farr_above[keys[i]])[0]):
            ax.text(farr_below[keys[i]][j] + 0.01,hrr_below[keys[i]][j] + 0.01,
                    str(bins[j]), color = 'k', fontsize=6)
            ax.text(farr_below[keys[i + 6]][j] + 0.01,hrr_below[keys[i + 6]][j] + 0.01,
                    str(bins[j]), color = 'k', fontsize=6)
            ax.text(farr_below[keys[i + 12]][j] + 0.01,hrr_below[keys[i + 12]][j] + 0.01,
                    str(bins[j]), color = 'k', fontsize=6)
        ax.plot(np.linspace(0,1.1,12), np.linspace(0,1.1,12), color = 'k')
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('False Alarm Rate', fontsize=9)
        ax.set_ylabel('Hit Rate', fontsize=9)
        plt.title(keys[i][3:], fontsize=10, loc='left')
    fig.subplots_adjust(top=0.85, bottom=0.3, hspace=0.32)
    ley2 = 'lead 1 month'
    ley4 = 'lead 4 months'
    ley6 = 'lead 7 monts'
    ax.legend([ley2, ley4, ley6], bbox_to_anchor=(0, -0.35),
               loc='center right', ncol=3)
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    plt.savefig(output, dpi = 300, bbox_inches = 'tight', papertype = 'A4')
    plt.close()
    plt.clf()
    plt.cla()

def plot_rel(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output):
    fig = plt.figure(1, (17, 9.7), 300)
    keys = sorted(hrrd_above.keys())
    index = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(np.arange(0.05, 1.05, 0.1), hrrd_above[keys[i]], color='b', marker='o')
        ax.plot(np.arange(0.05, 1.05, 0.1), hrrd_below[keys[i]], color='r', marker='o')
        ax.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12), color='k')
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('Mean Forecast Probability', fontsize=9)
        ax.set_ylabel( 'Observed Relative Frequency', fontsize=9)
        plt.title(index[i] + ' ' + keys[i].upper(), fontsize=10, loc='left')
        ##histograms 
        ax2 = ax.twinx()
        ax2.plot(np.arange(0.05, 1.05, 0.1), frd_above[keys[i]], color='b', linestyle=':')
        ax2.axis([0, 1, 0, 0.65])
        ax2.set_yticks(np.arange(0, 0.7, 0.1))
        ax2.set_ylabel('Forecast relative frequency')
        ax2.plot(np.arange(0.05, 1.05, 0.1), frd_below[keys[i]], color = 'r', linestyle=':')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(top=0.85, hspace=0.25, wspace=0.4)
    plt.savefig(output, dpi=600, bbox_inches = 'tight', papertype = 'A4')
    plt.cla()
    plt.clf()
    plt.close('all')
def plot_rel_above_leadtimes(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output):
    fig = plt.figure(1, (17, 9.7), 300)
    keys = sorted(hrrd_above.keys())
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(np.arange(0.05, 1.05, 0.1), hrrd_above[keys[i]],
                color='darkblue', marker='o')
        ax.plot(np.arange(0.05, 1.05, 0.1), hrrd_above[keys[i + 12]],
                color='royalblue', marker='o')
        ax.plot(np.arange(0.05, 1.05, 0.1), hrrd_above[keys[i + 6]],
                color='deepskyblue', marker='o')
        ax.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12), color='k')
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('Mean Forecast Probability', fontsize=9)
        ax.set_ylabel( 'Observed Relative Frequency', fontsize=9)
        plt.title(keys[i], fontsize=10)
        ##histograms 
        ax2 = ax.twinx()
        ax2.plot(np.arange(0.05, 1.05, 0.1), frd_above[keys[i]], color='darkblue',
                 linestyle=':')
        ax2.plot(np.arange(0.05, 1.05, 0.1), frd_above[keys[i + 12]], color='royalblue',
                 linestyle=':')
        ax2.plot(np.arange(0.05, 1.05, 0.1), frd_above[keys[i + 6]], color='deepskyblue',
                 linestyle=':')
        ax2.axis([0, 1, 0, 0.50])
        ax2.set_yticks(np.arange(0, 0.6, 0.1))
        ax2.set_ylabel('Forecast/Observation Pairs')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(top=0.85, hspace=0.35, wspace=0.4, bottom=0.30)
    ley2 = 'lead 1 month'
    ley4 = 'lead 4 months'
    ley6 = 'lead 7 monts'
    ax.legend([ley2, ley4, ley6], bbox_to_anchor=(-0.1, -0.38),
               loc='center right', ncol=3)
    plt.savefig(output, dpi=600, bbox_inches = 'tight', papertype = 'A4')
    plt.close()
    plt.cla()
    plt.clf()

def plot_rel_below_uncal_leadtimes(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output):
    fig = plt.figure(1, (17, 9.7), 300)
    keys = sorted(hrrd_above.keys())
    plt.plot(np.arange(0.05, 1.05, 0.1), hrrd_below[keys[1]],
             color='darkred', marker='o')
    plt.plot(np.arange(0.05, 1.05, 0.1), hrrd_below[keys[0]],
             color='red', marker='o')
    plt.plot(np.arange(0.05, 1.05, 0.1), hrrd_below[keys[6]],
             color='orangered', marker='o')
    plt.plot(np.arange(0.05, 1.05, 0.1), hrrd_below[keys[4]],
            color='salmon', marker='o')
    plt.plot(np.arange(0.05, 1.05, 0.1), hrrd_below[keys[2]],
            color='coral', marker='o')
    plt.plot(np.linspace(0, 1.1, 12), np.linspace(0, 1.1, 12), color='k')
    plt.axis([0, 1, 0, 1])
    plt.savefig(output, dpi=600, bbox_inches = 'tight', papertype = 'A4')
    plt.close()
    plt.cla()
    plt.clf()

def plot_evolution_scores(hss, rpss, ninio34, titulo, output):
    fig = plt.figure(1, (13, 7.7), 300)
    keys = sorted(hss.keys())
    ax = plt.subplot(2, 1, 1)
    color = iter(cm.rainbow(np.linspace(0, 1, 7)))
    for i in range(7):
        if keys[i] == 'MME': ax.plot(hss[keys[i]], color='k', marker='o', label=keys[i].upper())
        else:
            ax.plot(hss[keys[i]], color=next(color), marker='o', label=keys[i].upper())
    ax2 = ax.twinx()
    ax2.plot(ninio34, ':r', linewidth=2)
    ax.set_ylabel( 'HSS', fontsize=9)
    ax2.set_ylabel('Ninio 34')
    ax2.set_ylim(-5, 5)
    ax.set_ylim(0, 70)
    ax.set_xlim(-0.3, 28.3)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_xticks(np.arange(0, 29, 5))
    ax.set_xticklabels(('1982', '1987', '1992', '1997', '2002', '2007'))
    ax = plt.subplot(2, 1, 2)
    color = iter(cm.rainbow(np.linspace(0, 1, 7)))
    for i in range(7):
        if keys[i] == 'MME': ax.plot(rpss[keys[i]], color='k', marker='o', label=keys[i].upper())
        else:
            ax.plot(rpss[keys[i]], color=next(color), marker='o', label=keys[i].upper())
    ax.hlines(0, -0.3, 28.3, colors='k', linestyles='dotted')
    ax.axis([-0.3, 28.3, -0.2, 0.45])
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel( 'RPSS', fontsize=10)
    ax.set_xticks(np.arange(0, 29, 5))
    ax.set_xticklabels(('1982', '1987', '1992', '1997', '2002', '2007'))
    plt.legend(bbox_to_anchor=(1.03, 1.15), loc='center left')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(top=0.85, hspace=0.25)
    plt.savefig(output, dpi=600, bbox_inches = 'tight', papertype = 'A4')
    plt.cla()
    plt.clf()
    plt.close('all')

#================================================================
#
#===============================================================
#path to scores
RUTA = '/datos/osman/nmme_results/verif_scores/'
#RUTA = '/home/osman/'
ctech = ['wpdf', 'wsereg']
wtech = ['same', 'pdf_int', 'mean_cor']

#=======================================
#RPSS * 6
#=======================================
# open RPSS for each calibrated forecast and generate dictionary with scores
#rpss = {}
##
#for i in ctech:
#    if i == 'wpdf':
#        II = 'APDFs'
#    else:
#        II = 'MMEREG'
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        rpss[II + '-' + j] = np.load(RUTA + 'rpss_' + file_end)['rpss']
#lats = np.load(RUTA + 'rpss_' + file_end)['lat']
#lons = np.load(RUTA + 'rpss_' + file_end)['lon']
#
#titulo = 'RPSS DJF Precipitation Forecast IC Nov'
#output = './figures_paper/figura2.eps'
#plot_scores(lats, lons, rpss, titulo, output)
#=======================================
#RPSS * 6 IC Aug
#=======================================
# open RPSS for each calibrated forecast and generate dictionary with scores
#rpss = {}
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_8_DJF_' + j + '_' + i + '.npz'
#        rpss[i + '-' + j] = np.load(RUTA + 'rpss_' + file_end)['rpss']
#lats = np.load(RUTA + 'rpss_' + file_end)['lat']
#lons = np.load(RUTA + 'rpss_' + file_end)['lon']
#titulo = 'RPSS DJF Precipitation Forecast IC Aug'
#output = 'figura9.png'
#plot_scores(lats, lons, rpss, titulo, output)
#=======================================
#RPSS * 6 IC Jun
#=======================================
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_6_DJF_' + j + '_' + i + '.npz'
#        rpss[i + '-' + j] = np.load(RUTA + 'rpss_' + file_end)['rpss']
#lats = np.load(RUTA + 'rpss_' + file_end)['lat']
#lons = np.load(RUTA + 'rpss_' + file_end)['lon']
#
#titulo = 'RPSS DJF Precipitation Forecast IC Jun'
#output = 'figura11.png'
#plot_scores(lats, lons, rpss, titulo, output)
#=======================================
#ROC * 6
#=======================================
farr_above = {}
hrr_above = {}
farr_below = {}
hrr_below = {}
roc_above = {}
roc_below = {}
for i in ctech:
    if i == 'wpdf':
        II = 'APDFs'
    else:
        II = 'MMEREG'
    for j in wtech:
        file_end = 'rel_roc_all_prec_prob_forecast_11_DJF_' + j + '_' + i + '_paper.npz'
        farr_above[II + '-' + j] = np.load(RUTA + file_end)['farr_above']
        farr_below[II + '-' + j] = np.load(RUTA + file_end)['farr_below']
        hrr_above[II + '-' + j] = np.load(RUTA + file_end)['hrr_above']
        hrr_below[II + '-' + j] = np.load(RUTA + file_end)['hrr_below']
        roc_above[II + '-' + j] = np.load(RUTA + file_end)['roc_above']
        roc_below[II + '-' + j] = np.load(RUTA + file_end)['roc_below']
bins = np.arange(0, 1.1, 0.1)
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
titulo = 'ROC Diagram DJF Precipitation Forecast IC Nov'
output = './figures_paper/figura4_new.eps'
plot_roc(farr_above, hrr_above, farr_below, hrr_below, roc_above, roc_below, bins,
             titulo, output)

##=======================================
##Reliability * 6
##=======================================
frd_above = {}
frd_below = {}
hrrd_below = {}
hrrd_above = {}
for i in ctech:
    if i == 'wpdf':
        II = 'APDFs'
    else:
        II = 'MMEREG'
    for j in wtech:
        file_end = 'rel_roc_all_prec_prob_forecast_11_DJF_' + j + '_' + i + '_paper.npz'
        frd_above[II + '-' + j] = np.load(RUTA + file_end)['frd_above']
        frd_below[II + '-' + j] = np.load(RUTA + file_end)['frd_below']
        hrrd_above[II + '-' + j] = np.load(RUTA + file_end)['hrrd_above']
        hrrd_below[II + '-' + j] = np.load(RUTA + file_end)['hrrd_below']
#
titulo = 'Reliability Diagram DJF Precipitation Forecast IC Nov'
output = './figures_paper/figura3_new.eps'
plot_rel(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output)
#=======================================
#Uncalibrated Forecast
#=======================================
#figure 1: HSS. figure 2 RPSS figure 3: reliability, figure 4: roc
RUTA = '/datos/osman/nmme_results/verif_scores/'
file_end = 'prec_prob_forecast_11_DJF_same_count.npz'
rpss = np.load(RUTA + 'rpss_' + file_end)['rpss']
hss = np.load(RUTA + 'hss_' + file_end)['hss']
lats = np.load(RUTA + 'rpss_' + file_end)['lat']
lons = np.load(RUTA + 'rpss_' + file_end)['lon']
#
files = glob.glob('/datos/osman/nmme_results/cal_forecasts/correlation_*.npz')
#
correlation = np.empty([len(files), rpss.shape[0], rpss.shape[1]])
ii = 0
for i in files:
    correlation[ii, :, :] = np.nanmean(np.load(i)['correlation'], axis=0)
    ii +=1

correlation = np.nanmean(correlation, axis=0)
#RUTA = '/home/osman/'
file_end = 'rel_roc_all_prec_prob_forecast_11_DJF_same_count_paper.npz'
roc_comp = {}
roc_comp['farr_above'] = np.load(RUTA + file_end)['farr_above']
roc_comp['farr_below'] = np.load(RUTA + file_end)['farr_below']
roc_comp['hrr_above'] = np.load(RUTA + file_end)['hrr_above']
roc_comp['hrr_below'] = np.load(RUTA + file_end)['hrr_below']
roc_comp['roc_above'] = np.load(RUTA + file_end)['roc_above']
roc_comp['roc_below'] = np.load(RUTA + file_end)['roc_below']

rel_comp = {}
rel_comp['frd_above'] = np.load(RUTA + file_end)['frd_above']
rel_comp['frd_below'] = np.load(RUTA + file_end)['frd_below']
rel_comp['hrrd_above'] = np.load(RUTA + file_end)['hrrd_above']
rel_comp['hrrd_below'] = np.load(RUTA + file_end)['hrrd_below']
titulo = 'NMME Performance - DJF Precipitation Forecast IC Nov'
output = './figures_paper/figura1_new.eps'
plot_skill_uncal(lats, lons, hss, rpss, rel_comp, roc_comp, titulo, output)
#=======================================
#ROC diagram vs leadtimes
#=======================================
#farr_above = {}
#hrr_above = {}
#farr_below={}
#hrr_below = {}
#roc_above = {}
#roc_below = {}
#for k in [11, 8 , 5]:
#    for i in ctech:
#        for j in wtech:
#            file_end = 'rel_roc_all_prec_prob_forecast_' + str(k) + '_DJF_'  + \
#                    j + '_' + i + '.npz'
#            farr_above[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['farr_above']
#            farr_below[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['farr_below']
#            hrr_above[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['hrr_above']
#            hrr_below[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['hrr_below']
#            roc_above[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['roc_above']
#            roc_below[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['roc_below']
#
#bins = np.arange(0, 1.1, 0.1)
#
#titulo = 'ROC Diagram DJF Precipitation Forecast - Above normal category'
#output = 'figura5.png'
##plot_roc_above_leadtimes(farr_above, hrr_above, farr_below, hrr_below, roc_above, roc_below, bins,
##             titulo, output)
#
#titulo = 'ROC Diagram DJF Precipitation Forecast - Below normal category'
#output = 'figura6.png'
#plot_roc_below_leadtimes(farr_above, hrr_above, farr_below, hrr_below, roc_above, roc_below, bins,
#                         titulo, output)
#=======================================
#Reliability vs leadimes
#=======================================
#frd_above = {}
#frd_below = {}
#hrrd_below = {}
#hrrd_above = {}
#
#for k in [11, 8 , 5]:
#    for i in ctech:
#        for j in wtech:
#            file_end = 'rel_roc_all_prec_prob_forecast_' + str(k) + '_DJF_' +\
#                    j + '_' + i + '.npz'
#            frd_above[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['frd_above']
#            frd_below[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['frd_below']
#            hrrd_above[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['hrrd_above']
#            hrrd_below[str(k) + '-' + i + '-' + j] = np.load(RUTA + file_end)['hrrd_below']
#
#titulo = 'Reliability Diagram DJF Precipitation Forecast - Above normal caterogry'
#output = 'figura7.png'
##plot_rel_above_leadtimes(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output)
#titulo = 'Reliability Diagram DJF Precipitation Forecast - Above normal caterogry'
#output = 'figura8.png'
##plot_rel_below_leadtimes(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output)
#frd_above = {}
#frd_below = {}
#hrrd_below = {}
#hrrd_above = {}
#
#for k in [11, 10, 9, 8 , 7, 6, 5]:
#    file_end = 'rel_roc_all_prec_prob_forecast_' + str(k) + '_DJF_' +\
#                    'same_count.npz'
#    frd_above[str(k)] = np.load(RUTA + file_end)['frd_above']
#    frd_below[str(k)] = np.load(RUTA + file_end)['frd_below']
#    hrrd_above[str(k)] = np.load(RUTA + file_end)['hrrd_above']
#    hrrd_below[str(k)] = np.load(RUTA + file_end)['hrrd_below']
#
#titulo = 'Reliability Diagram DJF Precipitation Forecast - Above normal caterogry'
#output = 'figura_rel_above_uncal.png'
##plot_rel_above_leadtimes(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output)
#
#
#titulo = 'Reliability Diagram DJF Precipitation Forecast - Below normal caterogry'
#output = 'figura_rel_below_uncal.png'
#plot_rel_below_uncal_leadtimes(hrrd_above, hrrd_below, frd_above, frd_below, titulo, output)
#=======================================
#BSS * 6
#=======================================
# open BSS for each calibrated forecast and generate dictionary with scores
#bss = {}
#
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        bss[i + '-' + j] = np.load(RUTA + 'bss_above_' + file_end)['bss_above'][0]
#lats = np.load(RUTA + 'bss_above_' + file_end)['lat']
#lons = np.load(RUTA + 'bss_above_' + file_end)['lon']
#
#titulo = 'BS DJF Precipitation Forecast IC Nov'
#output = './figures_paper/figura12.png'
#plot_bs(lats, lons, bss, titulo, output)
#
#bss = {}
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
#        bss[i + '-' + j] = data[3, :, :] - data[4, :, :] - data[5, :, :]
#lats = np.load(RUTA + 'bss_above_' + file_end)['lat']
#lons = np.load(RUTA + 'bss_above_' + file_end)['lon']
#
#titulo = 'Rel BS DJF Precipitation Forecast IC Nov'
#output = './figures_paper/figura13.png'
#plot_bs_dec(lats, lons, bss, titulo, output)
#
#bss = {}
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
#        bss[i + '-' + j] =data[1, :, :] - data[2, :, :]
#lats = np.load(RUTA + 'bss_above_' + file_end)['lat']
#lons = np.load(RUTA + 'bss_above_' + file_end)['lon']
#
#titulo = 'Unc - Res BS DJF Precipitation Forecast IC Nov'
#output = './figures_paper/figura14.png'
#plot_bs_dec(lats, lons, bss, titulo, output)
#
## open BSS for each calibrated forecast and generate dictionary with scores
#bss = {}
#
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        bss[i + '-' + j] = np.load(RUTA + 'bss_below_' + file_end)['bss_below'][0]
#lats = np.load(RUTA + 'bss_below_' + file_end)['lat']
#lons = np.load(RUTA + 'bss_below_' + file_end)['lon']
#
#titulo = 'BS DJF Below Normal Precipitation Forecast IC Nov'
#output = './figures_paper/figura15.png'
#plot_bs(lats, lons, bss, titulo, output)
#
#bss = {}
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
#        bss[i + '-' + j] = data[3, :, :] - data[4, :, :] - data[5, :, :]
#lats = np.load(RUTA + 'bss_below_' + file_end)['lat']
#lons = np.load(RUTA + 'bss_below_' + file_end)['lon']
#
#titulo = 'Rel BS DJF Below Normal Precipitation Forecast IC Nov'
#output = './figures_paper/figura16.png'
#plot_bs_dec(lats, lons, bss, titulo, output)
#
#bss = {}
#for i in ctech:
#    for j in wtech:
#        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
#        data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
#        bss[i + '-' + j] =data[1, :, :] - data[2, :, :]
#lats = np.load(RUTA + 'bss_below_' + file_end)['lat']
#lons = np.load(RUTA + 'bss_below_' + file_end)['lon']
#
#titulo = 'Unc - Res BS Below Normal DJF Precipitation Forecast IC Nov'
#output = './figures_paper/figura17.png'
#plot_bs_dec(lats, lons, bss, titulo, output)
#=======================================
#HSS * 6
#=======================================
hss = {}

for i in ctech:
    if i == 'wpdf':
        II = 'APDFs'
    else:
        II = 'MMEREG'
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
        hss[II + '-' + j] = np.load(RUTA + 'hss_' + file_end)['hss']
lats = np.load(RUTA + 'hss_' + file_end)['lat']
lons = np.load(RUTA + 'hss_' + file_end)['lon']
titulo = 'Heidke Skill Score DJF Precipitation Forecast IC Nov'
output = './figures_paper/figura18.eps'
#hss = ma.masked_array(hss, mask= np.logical_not(land))
plot_hss(lats, lons, hss, titulo, output)
bss = {}
for i in ctech:
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '_all_paper.npz'
        data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
        bss[i + '-' + j] = data
print("All domain - Below Normal")
for i in bss.keys():
    print(i, (1 - bss[i][0]/ (0.33 * (1-0.33))) * 100 )
bss = {}
for i in ctech:
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '_all_paper.npz'
        data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
        bss[i + '-' + j] = data
print("All domain - Above Normal")
for i in bss.keys():
    print(i, (1 - bss[i][0]/ (0.33 * (1-0.33))) * 100)
bss = {}
for i in ctech:
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '_trop_SA_paper.npz'
        data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
        bss[i + '-' + j] = data
print("Tropics - Below Normal")
for i in bss.keys():
    print(i, (1 - bss[i][0]/ (0.33 * (1-0.33))) * 100 )
bss = {}
for i in ctech:
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '_trop_SA_paper.npz'
        data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
        bss[i + '-' + j] = data
print("Tropics - Above Normal")
for i in bss.keys():
    print(i, (1 - bss[i][0]/ (0.33 * (1-0.33))) * 100 )
bss = {}
for i in ctech:
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '_extratrop_SA_paper.npz'
        data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
        bss[i + '-' + j] = data
print("Extratropics - Below Normal")
for i in bss.keys():
    print(i, (1 - bss[i][0]/ (0.33 * (1-0.33))) * 100 )
bss = {}
for i in ctech:
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '_extratrop_SA_paper.npz'
        data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
        bss[i + '-' + j] = data
print("Extratropics - Above Normal")
for i in bss.keys():
    print(i, (1 - bss[i][0]/ (0.33 * (1-0.33))) * 100 )


file_end = 'prec_prob_forecast_11_DJF_same_count_all_paper.npz'
data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
print('Above - CE')
print('All', (1 - data[0]/ (0.33 * (1-0.33))) * 100)


file_end = 'prec_prob_forecast_11_DJF_same_count_extratrop_SA_paper.npz'
data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
print('Above - CE')
print('Extratropics', (1 - data[0]/ (0.33 * (1-0.33))) * 100)
file_end = 'prec_prob_forecast_11_DJF_same_count_trop_SA_paper.npz'
data = np.load(RUTA + 'bss_above_' + file_end)['bss_above']
print('Above - CE')
print('Tropics', (1 - data[0]/ (0.33 * (1-0.33))) * 100)

file_end = 'prec_prob_forecast_11_DJF_same_count_all_paper.npz'
data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
print('Below - CE')
print('All', (1 - data[0]/ (0.33 * (1-0.33))) * 100)


file_end = 'prec_prob_forecast_11_DJF_same_count_extratrop_SA_paper.npz'
data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
print('Below - CE')
print('Extratropics', (1 - data[0]/ (0.33 * (1-0.33))) * 100)
file_end = 'prec_prob_forecast_11_DJF_same_count_trop_SA_paper.npz'
data = np.load(RUTA + 'bss_below_' + file_end)['bss_below']
print('Below - CE')
print('Tropics', (1 - data[0]/ (0.33 * (1-0.33))) * 100)

hss = {}
rpss = {}
for i in ctech:
    if i == 'wpdf':
        II = 'APDFs'
    else:
        II = 'MMEREG'
    for j in wtech:
        file_end = 'prec_prob_forecast_11_DJF_' + j + '_' + i + '.npz'
        data = np.load(RUTA + 'temporal_rpss_hss_' + file_end)
        hss[II + '-' + j] = data['hss']
        rpss[II + '-' + j] = data['rpss']

file_end = 'prec_prob_forecast_11_DJF_same_count.npz'
data = np.load(RUTA + 'temporal_rpss_hss_' + file_end)
hss['MME'] = data['hss']
rpss['MME'] = data['rpss']
titulo = 'DJF Prec probabilistic forecast IC Nov'
output = './figures_paper/figura19.eps'

ninio34 = xr.open_dataset('ninio34_djf.nc4')
plot_evolution_scores(hss, rpss, ninio34['__xarray_dataarray_variable__'].values,
                      titulo, output)

#
#
#
#
#
#
#





