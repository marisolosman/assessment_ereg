#modules
"""
Modulo para instanciar observaciones, filtrar tendencia y obtener terciles
Dado que los datos de referencia utilizados tiene problemas en la variable temporal quedan cosas
por mejorar, a saber:
Cambiar la coordenada temporal del netcdf para hacerla compatible o poder obtener el pivot year
como atributo con xarray
Esto afecta a la funcion manipular_nc
"""
import datetime
import warnings
import numpy as np
import xarray as xr
from pandas.tseries.offsets import *
ruta = '/datos/osman/nmme/monthly/'
hind_length = 28

def manipular_nc(archivo, variable, lat_name, lon_name, lats, latn, lonw, lone,
                 last_month, year_init):
    #hay problemas para decodificar las fechas, genero un xarray con mis fechas decodificadas
    dataset = xr.open_dataset(archivo, decode_times=False)
    var_out = dataset[variable].sel(**{lat_name: slice(lats, latn), lon_name:
                                       slice(lonw, lone)})
    lon = dataset[lon_name].sel(**{lon_name: slice(lonw, lone)})
    lat = dataset[lat_name].sel(**{lat_name: slice(lats, latn)})
    pivot = datetime.datetime(1960, 1, 1) #dificilmente pueda obtener este atributo del nc sin
    #poder decodificarlo con xarray
    time = [pivot + DateOffset(months=int(x), days=15) for x in dataset['T']]
    #genero xarray con estos datos para obtener media estacional
    ds = xr.Dataset({variable: (('time', lat_name, lon_name), var_out)},
                    coords={'time': time, lat_name: lat, lon_name: lon})
    #como el resampling trimestral toma el ultimo mes como parametro
    var_out = ds[variable].resample(time='Q-' + last_month).mean(dim='time')
    #selecciono trimestre de interes
    mes = datetime.datetime.strptime(last_month, '%b').month
    var_out = var_out.sel(time=np.logical_and(var_out['time.month'] == mes,
        np.logical_and(var_out['time.year'] >= year_init,var_out['time.year']
                       <= (year_init+hind_length))))
    return var_out, lat, lon

class Observ(object):
    def __init__(self, institution, var_name, lat_name, lon_name, date_begin,
                 date_end):
        #caracteristicas comunes de todas las observaciones
        self.institution = institution
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.date_begin = date_begin
        self.date_end = date_end

#methods
    def select_months(self, last_month, year_init, lats, latn, lonw, lone):
        """computes seasonal mean"""
        print("seasonal mean")
        file = ruta + self.var_name + '_monthly_nmme_' + self.institution +'.nc'
        [variable, latitudes, longitudes] = manipular_nc(file, self.var_name,
                                                         self.lat_name,
                                                         self.lon_name, lats,
                                                         latn, lonw, lone,
                                                         last_month,
                                                         year_init)
        #converts obs pp unit to (mm/day) in 30-day month type
        variable = np.array(variable)
        if self.var_name == 'prec':
            variable = variable / 30
        return latitudes, longitudes, variable


