"""smoth calibrated probabilities using a gaussian filter and plot forecast"""
import argparse #parse command line options
import time #test time consummed
import calendar
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
plt.ioff()

def asignar_categoria(for_terciles):
    """determines most likely category"""
    most_likely_cat = np.argmax(for_terciles, axis=2)
    [nlats, nlons] = for_terciles.shape[0:2]
    for_cat = np.zeros([nlats, nlons], dtype=int)
    for_cat.fill(np.nan)
    for ii in np.arange(nlats):
        for jj in np.arange(nlons):
            if (most_likely_cat[ii, jj] == 2):
                if for_terciles[ii, jj, 2] >= 0.7:
                    for_cat[ii, jj] = 12
                elif for_terciles[ii, jj, 2] >= 0.6:
                    for_cat[ii, jj] = 11
                elif for_terciles[ii, jj, 2] >= 0.5:
                    for_cat[ii, jj] = 10
                elif for_terciles[ii, jj, 2] >= 0.4:
                    for_cat[ii, jj] = 9
            elif (most_likely_cat[ii, jj] == 0):
                if for_terciles[ii, jj, 0] >= 0.7:
                    for_cat[ii, jj] = 1
                elif for_terciles[ii, jj, 0] >= 0.6:
                    for_cat[ii, jj] = 2
                elif for_terciles[ii, jj, 0] >= 0.5:
                    for_cat[ii, jj] = 3
                elif for_terciles[ii, jj, 0] >= 0.4:
                    for_cat[ii, jj] = 4
            elif (most_likely_cat[ii, jj] == 1):
                if for_terciles[ii, jj, 1] >= 0.7:
                    for_cat[ii, jj] = 8
                elif for_terciles[ii, jj, 1] >= 0.6:
                    for_cat[ii, jj] = 7
                elif for_terciles[ii, jj, 1] >= 0.5:
                    for_cat[ii, jj] = 6
                elif for_terciles[ii, jj, 1] >= 0.4:
                    for_cat[ii, jj] = 5

            mascara = for_cat < 1
            for_mask = np.ma.masked_array(for_cat, mascara)
    return for_mask

def plot_pronosticos(IC, YEAR, SSS, leadtime, year_verif, ccmap, colores, titulo, salida):
    archivo = '/datos/osman/nmme_results/prec_obs_category_' + str(year_verif) + '.nc4'
    data = xr.open_dataset(archivo)
    lat = data.lat.values
    lon = data.lon.values
    obs_terciles = data.observed_terciles.sel(season=SSS, year=YEAR + 1).values
    obs = np.zeros_like(obs_terciles[0, :, :])
    obs[obs_terciles[1, :, :] == 1] = 1
    obs[obs_terciles[2, :, :] == 1] = 2
    obs[obs_terciles[0, :, :] == 1] = 0
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    cmap = mpl.colors.ListedColormap(np.array([[217, 95, 14], [189, 189, 189],
                                               [44, 162, 95]]) / 256)
    fig = plt.figure(1, (9.7, 5.2), 300)
    #Obs
    ax = plt.subplot(2, 4, 8, projection=mapproj)
    ax.set_extent(limits, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)

    CS1 = ax.pcolor(dx, dy, obs, cmap=cmap, alpha=0.6,
                         vmin=-0.5, vmax=2.5, transform=ccrs.PlateCarree())
    plt.title('h) Observed Category', fontsize=8, loc='left')
    ax1 = fig.add_axes([0.751, 0.05, 0.13, 0.03])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,
                                   boundaries=bounds, ticks=[0, 1, 2],
                                   spacing='uniform',
                                   orientation='horizontal', alpha=0.6)
    cb.set_ticklabels(['Below', 'Normal', 'Above'])
    cb.ax.tick_params(labelsize=7)
    forecast_path = '/datos/osman/nmme_results/'
    #CE
    ax = plt.subplot(2, 4, 4, projection=mapproj)
    file_end = 'prec_prob_forecast_same_count'
    data = xr.open_dataset(forecast_path + file_end + '.nc4')
    prob_terc_comb = data.forecast_terciles.sel(leadtime=leadtime).isel(time=np.logical_and(data['time.month']==IC, data['time.year']==YEAR)).values.squeeze()
    for_mask = asignar_categoria(prob_terc_comb)
    #projection and map limits
    ax.set_extent(limits, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    CS1 = ax.pcolor(dx, dy, for_mask, cmap = ccmap, vmin = 0.5, vmax = 12.5,
                            transform=ccrs.PlateCarree())
    plt.title('d) MME', fontsize=8, loc='left')
    ii = 1
    jj = 1
    wtech = ['pdf_int', 'mean_cor', 'same']
    ctech = ['wpdf', 'wsereg']
    index = ['a)', 'b)', 'c)', 'e)', 'f)', 'g)']
    for i in ctech:
        if i == 'wpdf':
            II = 'APDFs'
        else:
            II = 'MMEREG'
        for j in wtech:
            file_end = 'prec_prob_forecast_' + j + '_' + i
            data = xr.open_dataset(forecast_path + file_end + '.nc4')
            prob_terc_comb = data.forecast_terciles.sel(leadtime=leadtime).isel(time=np.logical_and(data['time.month']==IC, data['time.year']==YEAR)).values.squeeze()
            for_mask = asignar_categoria(prob_terc_comb)
            ax = plt.subplot(2, 4, ii, projection=mapproj)
            #projection and map limits
            ax.set_extent(limits, crs=ccrs.PlateCarree())
            ax.coastlines(alpha=0.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
            CS1 = ax.pcolor(dx, dy, for_mask, cmap = ccmap, vmin = 0.5, vmax = 12.5,
                            transform=ccrs.PlateCarree())
            plt.title(index[jj - 1 ] + ' ' + II + '-' + j.upper(), fontsize=8, loc='left')
            ii += 1
            jj += 1
        ii = 5
    plt.suptitle(titulo)
    ax1 = fig.add_axes([0.175, 0.05, 0.15, 0.03])
    cmap1 = mpl.colors.ListedColormap(colores[0:4, :])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap1, norm=norm, boundaries=bounds,
                                    ticks=[1, 2, 3, 4], spacing='uniform',
                                    orientation='horizontal')
    cb1.set_ticklabels(['+70%', '65%', '55%', '45%'])
    cb1.ax.tick_params(labelsize=7)
    cb1.set_label('Below')
    ax2 = fig.add_axes([0.335, 0.05, 0.15, 0.03])
    cmap2 = mpl.colors.ListedColormap(colores[4:8, :])
    bounds = [4.5, 5.5, 6.5, 7.5, 8.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap2, norm=norm, boundaries=bounds,
                                    ticks=[5, 6, 7, 8], spacing='uniform',
                                    orientation='horizontal')
    cb2.set_ticklabels(['45%', '55%', '65%', '+70%'])
    cb2.ax.tick_params(labelsize=7)
    cb2.set_label('Normal')
    ax3 = fig.add_axes([0.495, 0.05, 0.15, 0.03])
    cmap3 = mpl.colors.ListedColormap(colores[8:, :])
    bounds = [8.5, 9.5, 10.5, 11.5, 12.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap3, norm=norm, boundaries=bounds,
                                    ticks=[9, 10, 11, 12], spacing='uniform',
                                    orientation='horizontal')
    cb3.set_ticklabels(['45%', '55%', '65%', '+70%'])
    cb3.ax.tick_params(labelsize=7)
    cb3.set_label('Above')
    plt.savefig(salida, dpi=600, bbox_inches='tight', papertype='A4')
    plt.close()
    return

def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Verify combined forecast')
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable to verify (prec or temp)')
    parser.add_argument('IC', type=int, nargs=1,\
            help='Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('year', type=int, nargs=1, \
                        help='Year of IC (1982 to 2010) ')
    parser.add_argument('leadtime', type=int, nargs=1,\
            help='Forecast leatime (in months, from 1 to 7)')

    args=parser.parse_args()
    #defino ref dataset y target season
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    #custom colorbar
    colores = np.array([[166., 54., 3.], [230., 85., 13.], [253., 141., 60.],
                        [253., 190., 133.], [227., 227., 227.], [204., 204.,
                                                                 204.],
                        [150., 150., 150.], [82., 82., 82.], [186., 228.,
                                                              179.],
                        [116., 196., 118.], [49., 163., 84.], [0., 109.,
                                                               44.]]) / 255
    cmap = mpl.colors.ListedColormap(colores)
    RUTA = '/datos/osman/nmme_output/comb_forecast/'
    output = './figures_paper/figura21.eps'
    plot_pronosticos(args.IC[0], args.year[0], SSS, args.leadtime[0], year_verif,
                     cmap, colores, args.variable[0].upper() + ' DJF ' + str(args.year[0]) \
                     + '/' + str(args.year[0] + 1) + ' forecast and observation', output)
#===================================================================================================
start = time.time()
main()
end = time.time()
print(end - start)


