"""
this code plots model weights for each IC and leadtime
"""
#!/usr/bin/env python

import argparse #parse command line options
import time #test time consummed
import numpy as np
import glob 
from pathlib import Path 
import calendar
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from matplotlib.colors import from_levels_and_colors
def plot_weights(lat, lon, var, titulo, output):
    #funcion para graficar pesos (ergo barra entre 0 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    [dx,dy] = np.meshgrid (lon,lat)
    clevs = np.array([-0.3, -0.1, 0.1, 0.3, 0.5])
    num_levels = 5
    vmin, vmax = -0.3, 0.5
    midpoint = 0
    levels = np.linspace(vmin, vmax, num_levels)
    midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
    vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
    colors = plt.cm.PRGn(vals)
    cmap, norm = from_levels_and_colors(levels, colors)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (13, 9.7), 300)
    keys = sorted(var.keys())
    const = 1/len(keys)
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1, projection=mapproj)
        #projection and map limits
        ax.set_extent(limits, crs=ccrs.PlateCarree())
        ax.coastlines(alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        CS1 = ax.pcolor(dx, dy, var[keys[i]] - const, cmap=cmap, norm=norm,
                       vmin=-0.3, vmax=0.5, transform=ccrs.PlateCarree())
        plt.title(keys[i], fontsize=8)
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(bottom=0.17, top=0.82, hspace=0.1)
    cbar_ax = fig.add_axes([0.19, 0.1, 0.65, 0.02])
    cbar = fig.colorbar(CS1, cax=cbar_ax, orientation='horizontal',
                       ticks=clevs)
    cbar.ax.tick_params(labelsize = 8)
    plt.savefig(output, dpi=300, bbox_inches='tight', papertype='A4')
    plt.close()
    plt.clf()
    plt.cla()
    return

def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Plot models weights')
    parser.add_argument('variable',type=str, nargs= 1,\
            help='Variable calibrated (prec or temp)')
    parser.add_argument('IC', type = int, nargs= 1,\
            help = 'Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type = int, nargs = 1,\
            help = 'Forecast leatime (in months, from 1 to 7)')
    parser.add_argument('mod_spread',  type = float, nargs = 1,\
            help = 'percentage of spread retained in each model (from 0 to 1)')
    parser.add_argument('--no-model', nargs = '+', choices = ['CFSv2','CanCM3','CanCM4',\
            'CM2p1', 'FLOR-A06', 'FLOR-B01', 'GEOS5', 'CCSM3', 'CCSM4'], dest ='no_model',\
            help = 'Models to be discarded')

    # Extract dates from args
    args=parser.parse_args()
    lista = glob.glob("/home/osman/proyectos/postdoc/modelos/*") 
    if args.no_model is not None: #si tengo que descartar modelos
        lista = [i for i in lista if [line.rstrip('\n') for line in open(i)][0] not in args.no_model]
    
    keys = ['nombre', 'instit', 'latn', 'lonn', 'miembros', 'plazos', 'fechai', 'fechaf','ext']
    modelos = []

    for i in lista:
        lines = [line.rstrip('\n') for line in open(i)]
        modelos.append(dict(zip(keys, [lines[0], lines[1], lines[2], lines[3], int(lines[4]), 
            int(lines[5]), int(lines[6]), int(lines[7]), lines[8]])))

    nmodels = len(modelos)
    ny = int(np.abs(coords['lat_n'] - coords['lat_s']) + 1)
    nx = int(np.abs (coords['lon_e'] - coords['lon_w']) + 1) #doen not work if domain goes beyond greenwich
    nyears = int(modelos[0]['fechaf'] - modelos[0]['fechai'] + 1)
    #defino ref dataset y target season
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i-12 if i>12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    rmean = {}
    IC = calendar.month_abbr[args.IC[0]]
    route = '/datos/osman/nmme_smn/DATA/calibrated_forecasts/'
    weight = np.empty([29, ny, nx, 0])
    Rmean = np.array([]).reshape(ny, nx, 0)
    for i in modelos:
        archivo = args.variable[0] + '_' + i['nombre']+ '_' + IC + '_' + SSS +\
                '_gp_01_hind_parameters.npz'
        data = np.load(route+archivo)
        Rm = data['Rm']
        Rmean = np.concatenate((Rmean, Rm[:, :, np.newaxis]), axis=2)

        peso = data['peso']
        weight = np.concatenate((weight, peso[:, 0, :, :][:, :, :,
                                                          np.newaxis]), axis=3)
    lat = data['lats']
    lon = data['lons']
    #maximo = np.ndarray.argmax(weight, axis=3) #posicion en donde se da el maximo
    #ntimes = np.shape(weight)[0]
    j = 0
    #for i in modelos:
    #    rmean[i['nombre']] = np.nanmean(maximo == j, axis=0)
    #    j += 1
    Rmean[np.where(np.logical_or(Rmean < 0, ~np.isfinite(Rmean)))] = 0
    Rmean[np.nansum(Rmean[:, :, :], axis=2) == 0, :] = 1
    for i in modelos:
        rmean[i['nombre']] = Rmean[:, :, j] / np.nansum(Rmean, axis=2)
        j += 1

    plot_weights(lat, lon, rmean, 'Model Weight ', './figures_paper/prec_weights_' +\
                 IC + '_' + SSS + '.png')

#===================================================================================================

start = time.time()

#abro archivo donde guardo coordenadas
                                                 
coordenadas = '../postdoc/coords'

lines = [line.rstrip('\n') for line in open(coordenadas)]

coords = {'lat_s' : float(lines[1]),
        'lat_n' : float(lines [2]),
        'lon_w' : float(lines[3]),
        'lon_e' : float(lines[4])}

main()

end = time.time()

print(end - start)

# =================================================================================
