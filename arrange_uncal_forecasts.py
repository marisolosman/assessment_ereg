"""script to take consolidated forecast for SA and make netcdf file
M Osman - Dec 2019"""
import sys
import numpy as np
import os
import calendar
import scipy.ndimage as ndimage
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import xarray as xr
import pandas as pd
VAR = sys.argv[1]
RUTA = '/datos/osman/nmme_retrospective/output/comb_forecast/'
RUTA_OUT = '/datos/osman/nmme_results/'
forecast_terciles = np.empty([7, 29 * 12, 76, 56, 3])

for i in np.arange(0, 12):
    for j in np.arange(0, 7):
        seas = range(i + 1 + j + 1, i + 1 + j + 1 + 3)
        sss = [ii - 12 if ii > 12 else ii for ii in seas]
        SSS = "".join(calendar.month_abbr[ii][0] for ii in sss)
        IC = calendar.month_abbr[i + 1]
        #open 1982-2010 calibrated forecast for season with IC in month
        FILE = RUTA + VAR + "_mme_" + IC + "_" + SSS + "_gp_01_same_count_hind.npz"
        F = np.load(FILE)
        for k in range(29):
            for_terciles = np.squeeze(F['prob_terc_comb'][:, k, :, :])
            for_terciles = np.concatenate([for_terciles[0, :, :] [:, :, np.newaxis],
                                          (for_terciles[1, :, :] -\
                                           for_terciles[0, :, :])[:, :, np.newaxis],
                                          (1 - for_terciles[1, :, :])[:, :, np.newaxis]], axis=2)
            forecast_terciles[j, i + k * 12, :, :, :] = for_terciles
lat = F['lat']
lon = F['lon']
ds = xr.Dataset({'forecast_terciles': (['leadtime', 'time', 'latitude', 'longitude', 'category'],
                                       forecast_terciles)},
                coords={'lon': (['longitude'], lon), 'lat': (['latitude'], lat),
                        'time': pd.date_range('1982-01-01', periods=29 * 12, freq='M'),
                        'leadtime': np.arange(1, 8), 'category': ['below', 'normal', 'above']})

#generate netcdf file
ds.to_netcdf(RUTA_OUT + VAR + '_prob_forecast_same_count.nc4')

