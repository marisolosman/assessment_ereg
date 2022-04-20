import numpy as np
from scipy import stats
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
FILE = 'nina34.data'
data = []
year = []
with open(FILE) as f:
    f.readline()
    for line in f:
        columns = line.split()
        year.append(np.float(columns[0]))
        data.append(np.asarray([float(i) for i in columns[1:]]))
        if columns[0] == '2019': break

data = np.concatenate(data, axis=0)
data[data==-99.99] = np.nan
time = pd.date_range('1948-01-15', freq='M', periods=np.shape(data)[0])
ninio34 = xr.DataArray(data, coords=[time], dims=['time'])
ninio34 = ninio34.sel(time=slice('1950-01-01', '2019-12-31'))
#compute monthly anomalies

ninio34 = ninio34.groupby('time.month') - ninio34.groupby('time.month').mean('time', skipna=True)
#compute 3-month running mean
ninio34_filtered = np.convolve(ninio34.values, np.ones((3,))/3, mode='same')
ninio34_f = xr.DataArray(ninio34_filtered, coords=[ninio34.time.values], dims=['time'])
ninio34_f = ninio34_f.resample(time='QS-Dec').mean(skipna=True)
ninio34_f = ninio34.sel(time=slice('1982-12-01', '2010-12-31'))
ninio34_f = ninio34_f.sel(time=(ninio34_f['time.month'] == 12))
ninio34_f.to_netcdf('ninio34_djf.nc4')
