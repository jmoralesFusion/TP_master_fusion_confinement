#!/usr/bin/env python
# coding: utf-8

## Confinement analysis WEST

#%% Load modules

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy
import scipy.io as sio
import scipy.interpolate as sc_interp
import scipy.signal as sc_sig
from scipy.optimize import curve_fit
import sklearn
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
print('Logging version:', logging.__version__)
print('Matplotlib version:', matplotlib.__version__)
print('NumPy version:', np.__version__)
print('Pandas version:', pd.__version__)
print('Seaborn version:', sns.__version__)
print('Scipy version:', scipy.__version__)
print('Sklearn version:', sklearn.__version__)

# Local modules
import imas
try:
    import imas_west
except ImportError as err:
    print(' ')
    print('WARNING: no module imas_west, warning:', err)
    print(' ')
try:
    import pywed as pw
except ImportError as err:
    print(' ')
    print('WARNING: no module pywed, warning:', err)
    print(' ')

#%% Plot style
sns.set_style('whitegrid')

# To plot in a new window type in console: %matplotlib qt5

#%% Path to data

pathfile = '/Imas_public/public/plateau_statistics/west/'
filename = 'reduced_dataBase_C4_WEST.h'


#%% Read statistical data and signals names (keys)

stor = pd.HDFStore(pathfile+filename)
stor_keys = []
for ii in stor.keys():
    stor_keys.append(ii)
stats = stor['stats']
stor.close()


#%% Remove helium shots

helium_shots = [(55230, 55498), (55827, 55987)]

for iishot in helium_shots:
    #print(iishot[0])
    stats = stats[(stats['shot'] < iishot[0]) | (stats['shot'] > iishot[1])]

print('Stats shape:', stats.shape)
stats.tail()

#%% Read signals data if sig is not yet defined
try:
    sig
except NameError:
    sig = None

if sig is None:
    sig = {}
    for ii in stor_keys:
        if (('/signal/' in ii) and not ('contr_' in ii or 'gas_' in ii \
            or 'specv_' in ii or 'lang_' in ii or 'dist_wall_' in ii \
            or 'bolo_inv_' in ii or '_resist' in ii or '_phase' in ii)):
           print(ii.replace('/signal/', ''))
           sig[ii.replace('/signal/', '')] = pd.read_hdf(pathfile+filename, key=ii)


#%% Compute time relative to the plasma initiation (initiation at ignitron time)
sig['time_rel'] = {}
for ii in range(len(sig['time'])):
    ishot = sig['time'].index[ii]
    sig['time_rel'][ishot] = sig['time'][ishot] - sig['t_ignitron'][ishot]

sig['time_rel'] = pd.Series(sig['time_rel'])


#%% Plot W_mhd and total power as a function of time for shot 55799 with plateaus
shot = 55799
plt.figure(figsize=(9, 5))
sns.set_context('talk', font_scale=0.95)
plt.plot(sig['time_rel'][shot], 1E-6*sig['eq_w_mhd'][shot])
for ii in range(stats.loc[shot].shape[0]):
    t_ini = stats.loc[shot, 'tIni_plto'][ii]- stats.loc[shot, 't_ignitron'][ii]
    t_end = stats.loc[shot, 'tEnd_plto'][ii]- stats.loc[shot, 't_ignitron'][ii]
    t_plt = np.linspace(t_ini, t_end, 10)
    y_plt = 1E-6*stats.loc[shot, 'eq_w_mhd_mean_plto'][ii]*np.ones(10)
    plt.plot(t_plt, y_plt, label='Plateau '+str(ii), linewidth=4)
plt.xlabel('Time [s]')
plt.ylabel(r'$W_{MHD}$ [MJ]')
plt.title('Shot '+str(shot))
plt.legend()
plt.tight_layout()

#%%
plt.figure(figsize=(9, 5))
sns.set_context('talk', font_scale=0.95)
plt.plot(sig['time_rel'][shot], 1E-6*sig['P_TOT'][shot])
for ii in range(stats.loc[shot].shape[0]):
    t_ini = stats.loc[shot, 'tIni_plto'][ii]- stats.loc[shot, 't_ignitron'][ii]
    t_end = stats.loc[shot, 'tEnd_plto'][ii]- stats.loc[shot, 't_ignitron'][ii]
    t_plt = np.linspace(t_ini, t_end, 10)
    y_plt = 1E-6*stats.loc[shot, 'P_TOT_mean_plto'][ii]*np.ones(10)
    plt.plot(t_plt, y_plt, label='Plateau '+str(ii), linewidth=4)
plt.xlabel('Time [s]')
plt.ylabel(r'$P_{tot}$ [MW]')
plt.title('Shot '+str(shot))
plt.legend()
plt.tight_layout()

#%% Example: computation radiated fraction and P_rad divertor

stats['P_radDiv_mean_plto']  = stats['P_rad_mean_plto'] \
                             - stats['P_radBulk_mean_plto']

stats['f_rad_mean_plto']     = stats['P_rad_mean_plto'] \
                             / stats['P_TOT_mean_plto']
stats['f_radBulk_mean_plto'] = stats['P_radBulk_mean_plto'] \
                             / stats['P_TOT_mean_plto']
stats['f_radDiv_mean_plto']  = stats['P_radDiv_mean_plto'] \
                             / stats['P_TOT_mean_plto']

stats[['P_radDiv_mean_plto', 'f_rad_mean_plto']].describe()


#%% Filter data

stats = stats[(stats['eq_q_95_mean_plto'] < 20.) \
              & (stats['ne3_mean_plto'] > 0) \
              & (stats['P_COND_mean_plto'] > 0.) \
              & (stats['f_rad_mean_plto'] > 0.) \
              & (stats['eq_w_mhd_mean_plto'] > 0) \
              & (stats['contr_nelMax_mean_plto'] \
                    / (stats['neVol_mean_plto']) < 0.2) \
              & ((stats['shot'] < 55625) | (stats['shot'] > 55643))] # Filter wrong Te (ECE)


#%% First shot after boronisation

after_boro_shot = [53453, 54288, 54403, 54502, 54596, 54719, 54881, 55000, \
                   55138, 55499, 55548, 55747, 55795]

boro_shot_distance     = np.full(stats['shot'].size, np.nan)
boro_shot_distance_all = np.full(len(after_boro_shot), np.nan)
for jj in range(stats['shot'].size):
    for ii in range(len(after_boro_shot)):
        boro_shot_distance_all[ii] = stats['shot'][jj] - after_boro_shot[ii]
    if (~np.all(boro_shot_distance_all < 0) \
        and ~np.all(np.isnan(boro_shot_distance_all))):
        try:
            boro_shot_distance[jj] = \
              np.nanmin(boro_shot_distance_all[boro_shot_distance_all >= 0])
        except:
            print(jj)
            print(stats['shot'][jj])
            print(boro_shot_distance_all)
            raise
stats['boro_shot_distance'] = boro_shot_distance


#%% Plot (to plot in a new window type in console: %matplotlib qt5 )

plt.figure(figsize=(9, 5))
sns.set_context('talk', font_scale=0.95)
plt.scatter(stats['shot'], stats['boro_shot_distance'])
plt.xlabel('Shot number')
plt.ylabel('Distance to boronisation')
plt.tight_layout()


#%% Print columns (only "mean" columns)

for ii in stats.columns:
    if 'mean' in ii:
        print(ii)

#%% Confinement scaling law

# Quantities:
# - eq_w_mhd_mean_plto : confinement time [s]
# - Ip_mean_plto : toroidal current [A]
# - eq_b0_mean_plto : toroidal magnetic field [T]
# - P_TOT_mean_plto : total power (ohmic + auxiliary) [W]
# - ne_line3_mean_plto : line averaged density [m^-3]
# - eq_r0 : major radius [m]
# - eq_minor_rad_mean_plto : minor radius [m]
# - eq_elong_mean_plto : plasma elongation
# - Atomic mass of plasma particles can be fixed at 2 (deuterium)

# Before performing the regression you can explore these variables
# For example you can use histogram plots as:

plt.figure(figsize=(9, 5))
sns.set_context('talk', font_scale=0.95)
(1E-6*stats['Ip_mean_plto']).hist(bins=20)
plt.xlabel('Ip [MA]')
plt.ylabel('Nbr. plateaus')
plt.tight_layout()

# Also to visualize relation between variables use scatter plots:

plt.figure(figsize=(9, 5))
sns.set_context('talk', font_scale=0.95)
plt.scatter(1E-6*stats['Ip_mean_plto'], 1E-6*stats['eq_w_mhd_mean_plto'], \
            c=1E-6*stats['P_TOT_mean_plto'], \
            alpha=0.7, s=70, edgecolor='k', cmap='plasma', vmin=None, vmax=None)
plt.xlabel('Ip [MA]')
plt.ylabel(r'$W_{MHD}$ [MJ]')
plt.colorbar(label=r'$P_{tot}$ [MW]')
plt.tight_layout()

# To perform the linear regression you can use the linear_model module
# imported from scikit-learn (see line 21 of this file)
# More information: https://scikit-learn.org/stable/modules/linear_model.html
