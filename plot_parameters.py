"""
this code plots model parameters for each IC and leadtime
"""
#!/usr/bin/env python

import argparse #parse command line options
import time #test time consummed
import numpy as np
import glob 
from pathlib import Path 
import calendar
import math
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from matplotlib.colors import from_levels_and_colors

def plot_parameters(lat, lon, var, levs, modelos, titulo, output):
    #funcion para graficar scores (ergo barra entre -1 y 1)
    lats = np.min(lat)
    latn = np.max(lat)
    lonw = np.min(lon)
    lone = np.max(lon)
    # set desired contour levels.
    clevs = levs
    barra = plt.cm.get_cmap('coolwarm',11) #colorbar
    [dx,dy] = np.meshgrid (lon,lat)
    mapproj = ccrs.PlateCarree(central_longitude=(lonw + lone) / 2, globe=None)
    limits = [lonw, lone, lats, latn]
    fig = plt.figure(1, (9.7, 9.7), 300)
    #keys = sorted(var.keys())
    #index = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i in range(var.shape[2]):
        ax = plt.subplot(2, math.ceil(var.shape[2]/2), i + 1, projection=mapproj)
        #projection and map limits
        ax.set_extent(limits, crs=ccrs.PlateCarree())
        ax.coastlines(alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        CS1 = ax.pcolor(dx, dy, var[:, :, i], cmap = barra, vmin = clevs[0], vmax = clevs[-1],
                        transform=ccrs.PlateCarree())
        plt.title(modelos[i]['nombre'], fontsize=8, loc='left')
    plt.suptitle(titulo, fontsize=12, x=0.51, y=0.9)
    fig.subplots_adjust(bottom=0.17, top=0.82, hspace=0.1)
    cbar_ax = fig.add_axes([0.19, 0.1, 0.65, 0.02])
    cbar = fig.colorbar(CS1, cax=cbar_ax, orientation='horizontal',
                        ticks = levs)
    cbar.ax.tick_params(labelsize = 8)
    plt.savefig(output, dpi=600, bbox_inches='tight', papertype='A4')
    plt.clf()
    plt.cla()
    plt.close('all')
    return

def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Plot model parameters')
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
    lista = glob.glob("/datos/osman/nmme_smn/modelos/*")
 
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
    aux = np.empty([ny, nx, nmodels])
    parameters = {'peso': np.empty([nyears, ny, nx, nmodels]),
                  'Rm': aux, 'a1': aux, 'a2': aux, 'b1': aux, 'b2':aux,
                  'eps': aux, 'K': aux, 'Rb': aux}
    #defino ref dataset y target season
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i-12 if i>12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    i = 0
    for it in modelos:
        #abro archivo modelo
        archivo = Path('/datos/osman/nmme_smn/DATA/calibrated_forecasts/'+ args.variable[0] + '_' + it['nombre'
            ]+'_' + calendar.month_abbr[args.IC[0]] +'_'+ SSS +
                       '_gp_01_hind_parameters.npz')
        if archivo.is_file():
            data = np.load(archivo)
            for j in parameters.keys():
                #extraigo datos de cada modelo.
                if j == 'peso':
                    print(data[j].shape, parameters[j].shape)
                    parameters[j][:, :, :, i] = data [j]
                elif j == 'K':
                    parameters[j][:, :, i] = data [j][0, 0, : :]
                else:
                    parameters[j][:, :, i] = data [j]


            # weight =np.concatenate((weight,peso[:,0,:,:][:,:,:,np.newaxis]), axis = 3)
            # peso = []
            # Rm = data ['Rm']          
            # rmean = np.concatenate ((rmean, Rm[:,:,np.newaxis]), axis = 2)
        i += 1
    maximo = np.ndarray.argmax(parameters['peso'], axis=3) #posicion en donde se da el maximo
    ntimes = np.shape(parameters['peso'])[0]
    peso = np.empty([ny, nx, nmodels])
    for i in np.arange(nmodels):
        peso[:, :, i] = np.nanmean(maximo == i, axis=0)
    parameters['peso'] = peso 

    lat = data ['lats']
    lon = data ['lons']
    limits = {'a1': np.linspace(-1.1, 1.1, 11), 'a2': np.linspace(-1.1, 1.1, 11),
              'b1': np.linspace(-1.1, 1.1, 11),
              'b2': np.linspace(-0.006, 0.006, 11), 'eps': np.linspace(0, 2.2, 11),
              'K': np.linspace(0.75, 1.3, 11), 'peso': np.linspace(0, 0.6, 11),
              'Rm': np.linspace(-1.1, 1.1, 11), 'Rb': np.linspace(-1.1, 1.1, 11)}

    IC = calendar.month_abbr[args.IC[0]]
    route = '/home/osman/proyectos/assessment_ereg/figuras_parametros/'
    for j in parameters.keys():
        archivo = args.variable[0] + '_' + j + '_' + IC + '_' + SSS + '.png'
        plot_parameters(lat, lon, parameters[j], limits[j], modelos, j, route + archivo)

#===================================================================================================

start = time.time()

#abro archivo donde guardo coordenadas    
coordenadas = 'coords'

lines = [line.rstrip('\n') for line in open(coordenadas)]

coords = {'lat_s' : float(lines[1]),
        'lat_n' : float(lines [2]),
        'lon_w' : float(lines[3]),
        'lon_e' : float(lines[4])}

main()

end = time.time()

print(end - start)

# =================================================================================
