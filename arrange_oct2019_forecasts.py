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
wtech = sys.argv[2]
ctech = sys.argv[3]
RUTA = '/datos/osman/Oct2019/'
RUTA_OUT = '/datos/osman/nmme_results/'
#forecast_terciles = np.empty([7, 29 * 12, 76, 56, 3])
#
#for i in np.arange(0, 12):
#    for j in np.arange(0, 7):
#        seas = range(i + 1 + j + 1, i + 1 + j + 1 + 3)
#        sss = [ii - 12 if ii > 12 else ii for ii in seas]
#        #year_begin = 1982 if seas[-1] <= 12 else 1983
#        #year_end = year_begin + 28
#        SSS = "".join(calendar.month_abbr[ii][0] for ii in sss)
#        IC = calendar.month_abbr[i + 1]
#        #open 1982-2010 calibrated forecast for season with IC in month
#        FILE = RUTA + VAR + "_mme_" + IC + "_" + SSS + "_gp_01_" + wtech + "_" + ctech + "_hind.npz"
#        F = np.load(FILE)
#        for k in range(29):
#            for_terciles = np.squeeze(F['prob_terc_comb'][:, k, :, :])
#            #agrego el prono de la categoria above normal
#            if VAR == 'prec':
#                below = ndimage.filters.gaussian_filter(for_terciles[0, :, :], 1, order=0,
#                                                        output=None, mode='reflect')
#                above = ndimage.filters.gaussian_filter(1 - for_terciles[1, :, :], 1, order=0,
#                                                        output=None, mode='reflect')
#            else:
#                for_terciles[np.isnan(for_terciles)] = 0
#                kernel = Gaussian2DKernel(x_stddev=1)
#                below = convolve(for_terciles[0, :, :], kernel)
#                above = convolve(1 - for_terciles[1, :, :], kernel)
#            near = 1 - (above + below)
#            for_terciles = np.concatenate([below[:, :, np.newaxis], near[:, :, np.newaxis],
#                                           above[:, :, np.newaxis]], axis=2)
#            forecast_terciles[j, i + k * 12 , :, :, :] = for_terciles
#lat = F['lat']
#lon = F['lon']# - 360
#ds = xr.Dataset({'forecast_terciles': (['leadtime', 'time', 'latitude', 'longitude', 'category'],
#                                       forecast_terciles)},
#                coords={'lon': (['longitude'], lon), 'lat': (['latitude'], lat),
#                        'time': pd.date_range('1982-01-01', periods=29 * 12, freq='M'),
#                        'leadtime': np.arange(1, 8), 'category': ['below', 'normal', 'above']})
#
##generate netcdf file
#
#ds.to_netcdf(RUTA_OUT + VAR + '_prob_forecast_' + wtech + '_' + ctech + '.nc4')
#
forecast_terciles = np.empty([7, 76, 56, 3])
##open 2011-2018 calibrated forecast
#for i in range(0, 12):
i = 9
year = [2019]
for j in range(0, 7):
        for k in year:
            seas = range(i + 1 + j + 1, i + 1 + j + 1 + 3)
            sss = [ii - 12 if ii > 12 else ii for ii in seas]
            SSS = "".join(calendar.month_abbr[ii][0] for ii in sss)
            IC = calendar.month_abbr[i + 1]
            FILE = RUTA + VAR + "_mme_" + IC + str(k) + '_' + SSS + "_gp_01_" + wtech + "_" +\
                    ctech + ".npz"
            if os.path.isfile(FILE):
                F = np.load(FILE)
                for_terciles = np.squeeze(F['prob_terc_comb'][:, :, :])
                if VAR == 'prec':
                    below = ndimage.filters.gaussian_filter(for_terciles[0, :, :], 1, order=0,
                                                            output=None, mode='reflect')
                    above = ndimage.filters.gaussian_filter(1 - for_terciles[1, :, :], 1, order=0,
                                                            output=None, mode='reflect')
                else:
                    kernel = Gaussian2DKernel(x_stddev=1)
                    below = convolve(for_terciles[0, :, :], kernel)
                    above = convolve(1 - for_terciles[1, :, :], kernel)
                near = 1 - (above + below)
                for_terciles = np.concatenate([below[:, :, np.newaxis], near[:, :, np.newaxis],
                                               above[:, :, np.newaxis]], axis=2)
                forecast_terciles[j, :, :, :] = for_terciles
            else:
                forecast_terciles[j, :, :, :] = np.nan
lat = F['lat']
lon = F['lon']
ds = xr.Dataset({'forecast_terciles': (['leadtime', 'latitude', 'longitude', 'category'],
                                   forecast_terciles)},
            coords={'lon': (['longitude'], lon), 'lat': (['latitude'], lat),
                    'time': pd.date_range('2019-10-01', periods= 1, freq='M'),
                    'leadtime': np.arange(1, 8), 'category': ['below', 'normal', 'above']})
ds.to_netcdf(RUTA_OUT + VAR + '_rt_prob_forecast_' + wtech + '_' + ctech + '.nc4')



