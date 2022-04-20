"""This module computes calibrates models using ensemble regression"""
import warnings
import numpy as np
import xarray as xr

def manipular_nc(archivo, variable, lat_name, lon_name, lats, latn, lonw, lone):
    """gets netdf variables"""
    dataset = xr.open_dataset(archivo, decode_times=False)
    var_out = dataset[variable].sel(**{lat_name: slice(lats, latn), lon_name: slice(lonw, lone)})
    lon = dataset[lon_name].sel(**{lon_name: slice(lonw, lone)})
    lat = dataset[lat_name].sel(**{lat_name: slice(lats, latn)})
    return var_out, lat, lon

class Model(object):
    """Model definition"""
    def __init__(self, name, institution, var_name, lat_name,
                 lon_name, miembros_ensamble, leadtimes, hind_begin, hind_end,
                 extension, rt_ensamble):
        #caracteristicas comunes de todos los modelos
        self.name = name
        self.institution = institution
        self.ensembles = miembros_ensamble
        self.leadtimes = leadtimes
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.hind_begin = hind_begin
        self.hind_end = hind_end
        self.ext = extension
        self.rt_ensembles = rt_ensamble
    #imprimir caracteristicas generales del modelo
    def __str__(self):
        return "%s is a model from %s and has %s ensemble members and %s leadtimes" % (self.name,
                self.institution, self.ensembles, self.leadtimes)

#comienzo la definición de los métodos
    def select_months(self, init_cond, target, lats, latn, lonw, lone):
        """select forecasted season based on IC and target"""
        #init_cond en meses y target en meses ()
        final_month = init_cond + 11
        if final_month > 12:
            flag_end = 1
            final_month = final_month - 12
        else:
            flag_end = 0
        ruta = '/datos/osman/nmme/monthly/'
        #abro un archivo de ejemplo
        hindcast_length = self.hind_end - self.hind_begin + 1
        forecast = np.empty([hindcast_length, self.ensembles, int(np.abs(latn - lats)) + 1,
            int(np.abs(lonw - lone)) + 1])
        #loop sobre los anios del hindcast period
        for i in np.arange(self.hind_begin, self.hind_end+1):
            for j in np.arange(1, self.ensembles + 1):
                file = ruta + self.var_name + '_Amon_' + self.institution + '-' +\
                        self.name + '_' + str(i)\
                        + '{:02d}'.format(init_cond) + '_r' + str(j) + '_' + str(i) +\
                        '{:02d}'.format(init_cond) + '-' + str(i + flag_end) + '{:02d}'.format(
                            final_month) + '.' + self.ext

                [variable, latitudes, longitudes] = manipular_nc(file, self.var_name,
                                                                 self.lat_name, self.lon_name,
                                                                 lats, latn, lonw, lone)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        forecast[i - self.hind_begin, j - 1, :, :] = np.nanmean(
                            np.squeeze(np.array(variable))[target:target + 3, :, :], axis=0)
                        #como todos tiene en el 0 el prono del propio
                    except RuntimeWarning:
                        forecast[i - self.hind_begin, j - 1, :, :] = np.NaN
                variable = []
	# Return values of interest: latitudes longitudes forecast
        return latitudes, longitudes, forecast


