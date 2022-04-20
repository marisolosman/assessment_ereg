"""
This code compute the standar WMO verification scores for the combined 
forecast obtained through 
combination.py code
"""
#!/usr/bin/env python

import argparse #parse command line options
import time #test time consummed
from pathlib import Path
import calendar
import numpy as np
import xarray as xr
import verif_scores #calculo y grafico de los indices de verificacion
import numpy.ma as ma

def main():
    # Define parser data
    parser = argparse.ArgumentParser(description='Verify combined forecast')
    parser.add_argument('variable', type=str, nargs=1,\
            help='Variable to verify (prec or temp)')
    parser.add_argument('IC', type=int, nargs=1,\
            help='Month of intial conditions (from 1 for Jan to 12 for Dec)')
    parser.add_argument('leadtime', type=int,  nargs=1,\
            help='Forecast leatime (in months, from 1 to 7)')
    subparsers = parser.add_subparsers(help="Combination technique")
    wpdf_parser = subparsers.add_parser('wpdf', help='weighted sum of calibrated PDFs')
    wsereg_parser = subparsers.add_parser('wsereg',
                                          help='Ereg with the weighted '
                                          + 'superensemble')
    count_parser = subparsers.add_parser('count',
                                         help='Count members in each bin')
    count_parser.set_defaults(ctech='count', wtech=['same'])
    wpdf_parser.set_defaults(ctech='wpdf')
    wpdf_parser.add_argument("--weight_tech", required=True, nargs=1,
                             choices=['pdf_int', 'mean_cor', 'same'],
                             dest='wtech')
    wsereg_parser.set_defaults(ctech='wsereg')
    wsereg_parser.add_argument("--weight_tech", required=True, nargs=1,
                               choices=['pdf_int', 'mean_cor', 'same'],
                               dest='wtech', help='Relative weight between ' +
                               'models')
    # Extract dates from args
    args = parser.parse_args()
    #defino ref dataset y target season
    seas = range(args.IC[0] + args.leadtime[0], args.IC[0] + args.leadtime[0] + 3)
    sss = [i - 12 if i > 12 else i for i in seas]
    year_verif = 1982 if seas[-1] <= 12 else 1983
    SSS = "".join(calendar.month_abbr[i][0] for i in sss)
    #obtengo datos observados
    archivo = Path('/datos/osman/nmme_results/' + args.variable[0] + '_' + 'obs_category_' +
                   str(year_verif) + '.nc4')
    data = xr.open_dataset(archivo)
    obs_terciles = data.observed_terciles.sel(season=SSS).values
    #abro el prono consolidado
    forecast_path = '/datos/osman/nmme_results/'
    file_end = args.variable[0] + '_prob_forecast_' + args.wtech[0] + '_' + args.ctech
    data = xr.open_dataset(forecast_path + file_end + '.nc4')
    lat = data.lat.values
    lon = data.lon.values
    prob_terc_comb = data.forecast_terciles.sel(leadtime=args.leadtime[0]).isel(time=(data['time.month']==args.IC[0])).values
    prob_terc_comb = np.transpose(prob_terc_comb, (3, 0, 1, 2))
    [nlats, nlons] = prob_terc_comb.shape[2:4]
# =============================================================================
    # calculo y grafico los diferentes scores de verificacion:
    bins = np.arange(0, 1.1, 0.1)
    route = '/datos/osman/nmme_results/'
    ruta = '/datos/osman/nmme_results/verif_scores/'
    file_end = args.variable[0] + '_prob_forecast_' + str(args.IC[0]) + '_' + SSS + '_' + \
            args.wtech[0] + '_' + args.ctech
    print("Heidke Skill Score")
    hss = verif_scores.HSS(prob_terc_comb, obs_terciles)
    titulo = 'Heidke Skill Score'
    salida = route + 'hss/hss_'+ file_end + '.png'
    verif_scores.plot_scores(lat, lon, hss, titulo, salida)
    archivo = 'hss_' + file_end + '.npz'
    np.savez(ruta + archivo, hss=hss, lat=lat, lon=lon)
    print("Brier Skill Score")
    lsmask = "/datos/osman/nmme_smn/NMME/lsmask.nc"
    coordenadas = 'coords'
    domain = [line.rstrip('\n') for line in open(coordenadas)]  #Get domain limits
    coords = {'lat_s': float(domain[1]),
                        'lat_n': float(domain[2]),
                        'lon_w': float(domain[3]),
                        'lon_e': float(domain[4])}
    land = xr.open_dataset(lsmask)
    land = land.sel(Y=slice(coords['lat_n'], coords['lat_s']), X=slice(coords['lon_w'],
                                                                       coords['lon_e']))
    land = np.flipud(land.land.values)
    land = np.tile(land, (3, 29, 1, 1))

    salida = file_end + '_all'
    prob_terc_comb = ma.masked_array(prob_terc_comb,
                                             mask=np.logical_not(land))
    obs_terciles = ma.masked_array(obs_terciles, mask=np.logical_not(land))


    prob = np.reshape(prob_terc_comb[2, :, :, :], [prob_terc_comb.shape[1], nlats*nlons ])
    obs =  np.reshape(obs_terciles[2, :, :, :], [obs_terciles.shape[1], nlats*nlons ])
    BSS_above = verif_scores.BS_decomposition(prob, obs, bins)
    archivo = 'bss_above_' + file_end + '_all_paper.npz'
    np.savez(ruta + archivo, bss_above=BSS_above)
    prob = np.reshape(prob_terc_comb[0, :, :, :], [prob_terc_comb.shape[1], nlats*nlons ])
    obs =  np.reshape(obs_terciles[0, :, :, :], [obs_terciles.shape[1], nlats*nlons ])
    BSS_below = verif_scores.BS_decomposition(prob, obs, bins)
    archivo = 'bss_below_' + file_end + '_all_paper.npz'
    np.savez(ruta + archivo, bss_below=BSS_below, lat=lat, lon=lon)
        #SA tropical north of 20degree south 85W- 30W
    salida = file_end + '_trop_SA'
    latn = 15
    lats = -20
    lonw = 275
    lone = 330
    lati = np.argmin(abs(lat - lats))
    latf = np.argmin(abs(lat - latn)) + 1
    loni = np.argmin(abs(lonw - lon))
    lonf = np.argmin(abs(lone - lon)) + 1
    lat_trop_SA = lat[lati:latf]
    prob = np.reshape(prob_terc_comb[2, :, lati:latf, loni:lonf],
                      [prob_terc_comb.shape[1], (latf - lati) * (loni - lonf)])
    obs = np.reshape(obs_terciles[2, :, lati:latf, loni:lonf],
                      [obs_terciles.shape[1], (latf - lati) * (loni - lonf)])
    archivo = 'bss_above_' + salida + '_paper.npz'
    BSS_above = verif_scores.BS_decomposition(prob, obs, bins)
    np.savez(ruta + archivo, bss_above=BSS_above)
    BSS_below = np.empty([6])
    prob = np.reshape(prob_terc_comb[0, :, lati:latf, loni:lonf],
                      [prob_terc_comb.shape[1], (latf - lati) * (loni - lonf)])
    obs = np.reshape(obs_terciles[0, :, lati:latf, loni:lonf],
                      [obs_terciles.shape[1], (latf - lati) * (loni - lonf)])
    archivo = 'bss_below_' + salida + '_paper.npz'
    BSS_below = verif_scores.BS_decomposition(prob, obs, bins)
    np.savez(ruta + archivo, bss_below=BSS_below)
    #SA extratropical 20S-55S  292-308
    salida = file_end + '_extratrop_SA'
    latn = -20
    lats = -55
    lonw = 275
    lone = 330
    lati = np.argmin(abs(lat-lats))
    latf = np.argmin(abs(lat-latn)) + 1
    loni = np.argmin(abs(lonw-lon))
    lonf = np.argmin(abs(lone-lon)) + 1
    lat_extratrop_SA = lat[lati:latf]
    prob = np.reshape(prob_terc_comb[2, :, lati:latf, loni:lonf],
                      [prob_terc_comb.shape[1], (latf - lati) * (loni - lonf)])
    obs = np.reshape(obs_terciles[2, :, lati:latf, loni:lonf],
                      [obs_terciles.shape[1], (latf - lati) * (loni - lonf)])
    archivo = 'bss_above_' + salida + '_paper.npz'
    BSS_above = verif_scores.BS_decomposition(prob, obs, bins)
    np.savez(ruta + archivo, bss_above=BSS_above)
    BSS_below = np.empty([6])
    prob = np.reshape(prob_terc_comb[0, :, lati:latf, loni:lonf],
                      [prob_terc_comb.shape[1], (latf - lati) * (loni - lonf)])
    obs = np.reshape(obs_terciles[0, :, lati:latf, loni:lonf],
                      [obs_terciles.shape[1], (latf - lati) * (loni - lonf)])
    archivo = 'bss_below_' + salida + '_paper.npz'
    BSS_below = verif_scores.BS_decomposition(prob, obs, bins)
    np.savez(ruta + archivo, bss_below=BSS_below)
#===================================================================================================
start = time.time()
main()
end = time.time()
print(end - start)

# =================================================================================
