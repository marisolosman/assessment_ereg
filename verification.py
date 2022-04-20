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
    #bins = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1]
    route = '/datos/osman/nmme_results/'
    #ruta = '/datos/osman/nmme_results/verif_scores/'
    ruta = '/home/osman/'
    file_end = args.variable[0] + '_prob_forecast_' + str(args.IC[0]) + '_' + SSS + '_' + \
            args.wtech[0] + '_' + args.ctech
    print("Ranked Probabilistic Skill Score")
    rpss = verif_scores.RPSS(prob_terc_comb, obs_terciles)
    titulo = 'Ranked Probabilistic Skill Score'
    salida = route + 'rpss/rpss_'+ file_end + '.png'
    verif_scores.plot_scores(lat, lon, rpss, titulo, salida)
    archivo = 'rpss_' + file_end + '.npz'
    np.savez(ruta + archivo, rpss=rpss, lat=lat, lon=lon)
    print("Brier Skill Score")
    BSS_above = np.empty([6, nlats, nlons])
    for i in np.arange(nlats):
        for j in np.arange(nlons):
            BSS_above[:, i, j] = verif_scores.BS_decomposition\
            (prob_terc_comb[2, :, i, j], obs_terciles[2, :, i, j], bins)
    archivo = 'bss_above_' + file_end + '.npz'
    np.savez(ruta + archivo, bss_above=BSS_above, lat=lat, lon=lon)
    titulo = 'Brier Skill Score - Above Normal event'
    salida = route + 'bss/brierss_above_'+ file_end + '.png'
    BSS = 1 - BSS_above[0] / (0.33 * (1 - 0.33))
    verif_scores.plot_scores(lat, lon, BSS, titulo, salida)
    titulo = 'BSS - Resolution - Above Normal event'
    salida = route + 'bss/bss_res_above_' + file_end + '.png'
    verif_scores.plot_scores(lat, lon, BSS_above[2, :, :] / BSS_above[1, :, :],
                             titulo, salida)
    titulo = 'BSS - Reliability - Above Normal event'
    salida = route + 'bss/bss_rel_above_'+ file_end + '.png'
    verif_scores.plot_scores(lat, lon, -1 * BSS_above[3, :, :] /
                             BSS_above[1, :, :], titulo, salida)
    BSS_below = np.empty([6, nlats, nlons])
    for i in np.arange(nlats):
        for j in np.arange(nlons):
            BSS_below[:, i, j] = verif_scores.BS_decomposition(
                prob_terc_comb[0, :, i, j], obs_terciles[0, :, i, j], bins)
    archivo = 'bss_below_' + file_end + '.npz'
    np.savez(ruta + archivo, bss_below=BSS_below, lat=lat, lon=lon)
    titulo = 'Brier Skill Score - Below Normal event'
    salida = route + 'bss/brierss_below_' + file_end + '.png'
    BSS = 1 - BSS_below[0] / (0.33 * (1 - 0.33))
    verif_scores.plot_scores(lat, lon, BSS, titulo, salida)
    titulo = 'BSS - Resolution - Below Normal event'
    salida = route + 'bss/bss_res_below_'+ file_end + '.png'
    verif_scores.plot_scores(lat, lon, BSS_below[2, :, :] /
                             BSS_above[1, :, :], titulo, salida)
    titulo = 'BSS - Reliability - Below Normal event'
    salida = route + 'bss/bss_rel_below_' + file_end + '.png'
    verif_scores.plot_scores(lat, lon, -1 * BSS_below[3, :, :] /
                             BSS_above[1, :, :], titulo, salida)
    print("Area Under Roc Curve")
    auroc_above = verif_scores.auroc(prob_terc_comb[2, :, :, :],
                                     obs_terciles[2, :, :, :], lat, bins)
    archivo = 'auroc_above_' + file_end + '.npz'
    np.savez(ruta + archivo, auroc_above=auroc_above, lat=lat, lon=lon)
    titulo = 'Area under curve ROC - Above Normal event'
    salida = route + 'auroc/auroc_above_' + file_end + '.png'
    verif_scores.plot_scores(lat, lon, auroc_above[:, :], titulo, salida)

    auroc_below = verif_scores.auroc(prob_terc_comb[0, :, :, :],
                                     obs_terciles[0, :, :, :], lat, bins)
    archivo = 'auroc_below_' + file_end + '.npz'
    np.savez(ruta + archivo, auroc_below=auroc_below, lat=lat, lon=lon)
    titulo = 'Area under curve ROC - Below Normal event'
    salida = route + 'auroc/auroc_below_' + file_end + '.png'
    verif_scores.plot_scores(lat, lon, auroc_below[:, :], titulo, salida)
    print("Reliability and ROC diagrams")
    #todo el dominio
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
    prob_terc_comb_all = ma.masked_array(prob_terc_comb,
                                             mask=np.logical_not(land))
    obs_terciles_all = ma.masked_array(obs_terciles, mask=np.logical_not(land))

    [roc_above, roc_below, hrr_above, farr_above, hrr_below, farr_below,
     hrrd_above, hrrd_below, frd_above, frd_below] = verif_scores.rel_roc(prob_terc_comb_all,
                                                    obs_terciles_all, lat, bins,
                                                    route, salida)
    archivo = 'rel_roc_all_' + file_end + '_paper.npz'
    np.savez(ruta + archivo, roc_above=roc_above, roc_below=roc_below,
             hrr_above=hrr_above, farr_above=farr_above, hrr_below=hrr_below,
             farr_below=farr_below, hrrd_above=hrrd_above,
             hrrd_below=hrrd_below, frd_above=frd_above, frd_below=frd_below)
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
    prob_terc_comb_trop_SA = ma.masked_array(prob_terc_comb[:, :, lati:latf, loni:lonf],
                                             mask=np.logical_not(land[:, :, lati:latf, loni:lonf]))
    obs_terciles_trop_SA = ma.masked_array(obs_terciles[:, :, lati:latf, loni:lonf],
                                           mask=np.logical_not(land[:, :, lati:latf, loni:lonf]))
    [roc_above, roc_below, hrr_above, farr_above, hrr_below, farr_below,
     hrrd_above, hrrd_below, frd_above, frd_below] = verif_scores.rel_roc(prob_terc_comb_trop_SA,
                                                    obs_terciles_trop_SA,
                                                    lat_trop_SA, bins, route,
                                                    salida)
    archivo = 'rel_roc_trop_SA_' + file_end + '_paper.npz'
    np.savez(ruta + archivo, roc_above=roc_above, roc_below=roc_below,
             hrr_above=hrr_above, farr_above=farr_above, hrr_below=hrr_below,
             farr_below=farr_below, hrrd_above=hrrd_above,
             hrrd_below=hrrd_below, frd_above=frd_above, frd_below=frd_below)
    #SA extratropical 20S-55S  292-308
    salida = file_end + '_extratrop_SA'
    latn = -20
    lats = -55
    lonw = 292
    lone = 308
    lati = np.argmin(abs(lat-lats))
    latf = np.argmin(abs(lat-latn)) + 1
    loni = np.argmin(abs(lonw-lon))
    lonf = np.argmin(abs(lone-lon)) + 1
    lat_extratrop_SA = lat[lati:latf]
    prob_terc_comb_extratrop_SA = ma.masked_array(prob_terc_comb[:, :, lati:latf, loni:lonf],
                                                  mask=np.logical_not(land[:, :, lati:latf, loni:lonf]))
    obs_terciles_extratrop_SA = ma.masked_array(obs_terciles[:, :, lati:latf, loni:lonf],
                                                np.logical_not(land[:, :, lati:latf, loni:lonf]))
    [roc_above, roc_below, hrr_above, farr_above, hrr_below, farr_below,
     hrrd_above, hrrd_below, frd_above, frd_below] = verif_scores.rel_roc(
         prob_terc_comb_extratrop_SA, obs_terciles_extratrop_SA,
         lat_extratrop_SA, bins, route, salida)
    archivo = 'rel_roc_extratrop_SA_' + file_end + '_paper.npz'
    np.savez(ruta + archivo, roc_above=roc_above, roc_below=roc_below,
             hrr_above=hrr_above, farr_above=farr_above, hrr_below=hrr_below,
             farr_below=farr_below, hrrd_above=hrrd_above,
             hrrd_below=hrrd_below, frd_above=frd_above, frd_below=frd_below)
#===================================================================================================
start = time.time()
main()
end = time.time()
print(end - start)

# =================================================================================
