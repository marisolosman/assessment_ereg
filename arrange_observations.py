"""script to take consolidated forecast for SA and make netcdf file
M Osman - Dec 2019"""
import sys
import numpy as np
import os
#import observation
import calendar
import scipy.ndimage as ndimage
import xarray as xr
import pandas as pd
VAR = sys.argv[1]
RUTA = '/datos/osman/nmme_retrospective/output/comb_forecast/'
RUTA_OUT = '/datos/osman/nmme_results/'

SSS ={'1982': ['FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND'],
      '1983':  ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS']}


#obtengo datos observados
for i in np.arange(1982, 1984):
    sss = SSS[str(i)]
    observed_terciles = np.empty([9, 3, 29, 76, 56])
    ii = 0
    for j in sss:
        archivo = '/datos/osman/nmme_output/obs_' + VAR + '_'+ str(i) + '_' + j + '.npz'
        data = np.load(archivo)
        obs_terciles = data['cat_obs'] # 3 * 29 * 76 *56
        observed_terciles[ii, :, :, :, :] = obs_terciles
        ii +=1
    lat = data['lats_obs']
    lon = data['lons_obs']
    ds = xr.Dataset({'observed_terciles': (['season', 'category', 'year', 'latitude', 'longitude'],
                                       observed_terciles)},
                    coords={'lon': (['longitude'], lon), 'lat': (['latitude'], lat),
                            'year': np.arange(i, i +29), 'season': sss,
                            'category': ['below', 'normal', 'above']})
    #generate netcdf file
    ds.to_netcdf(RUTA_OUT + VAR + '_obs_category_' + str(i) + '.nc4')

