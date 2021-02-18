#!/usr/bin/env python
# coding: utf-8

# # Confinement analysis WEST

# In[ ]:


import logging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
from sklearn import datasets, linear_model
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


# ## Read data

# ### C4

# In[ ]:


pathfile = '/Imas_public/public/plateau_statistics/west/'
filename = 'reduced_dataBase_C4_WEST.h'


# In[ ]:


stats = pd.read_hdf(pathfile+filename, key='stats')


# In[ ]:


stats.shape


# In[ ]:


stats.tail()


# ## Example: computation radiated fraction and P_rad divertor

# In[ ]:


stats['P_radDiv_mean_plto']  = stats['P_rad_mean_plto'] \
                             - stats['P_radBulk_mean_plto']


# In[ ]:


stats['f_rad_mean_plto']     = stats['P_rad_mean_plto'] \
                             / stats['P_TOT_mean_plto']
stats['f_radBulk_mean_plto'] = stats['P_radBulk_mean_plto'] \
                             / stats['P_TOT_mean_plto']
stats['f_radDiv_mean_plto']  = stats['P_radDiv_mean_plto'] \
                             / stats['P_TOT_mean_plto']


# In[ ]:


stats[['P_radDiv_mean_plto', 'f_rad_mean_plto']].describe()


# ## Filter data

# In[ ]:


stats = stats[(stats['eq_q_95_mean_plto'] < 20.) \
              & (stats['P_COND_mean_plto'] > 0.) \
              & (stats['f_rad_mean_plto'] > 0.26) \
              & (stats['eq_w_mhd_mean_plto'] > 0) \
              & ((stats['shot'] < 55625) | (stats['shot'] > 55643))]


# ## First shot after boronisation

# In[ ]:


after_boro_shot = [53453, 54288, 54403, 54502, 54596, 54719, 54881, 55000, \
                   55138, 55499, 55548, 55747, 55795]


# In[ ]:


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


# In[ ]:

sns.set_context('talk', font_scale=0.95)
plt.scatter(stats['shot'], stats['boro_shot_distance'])
plt.xlabel('Shot number')
plt.ylabel('Distance to boronisation')


# ## Helium shots

# In[ ]:


helium_shots = [(55230, 55498), (55827, 55987)]


# In[ ]:


for iishot in helium_shots:
    #print(iishot[0])
    stats = stats[(stats['shot'] < iishot[0]) | (stats['shot'] > iishot[1])]


# In[ ]:


stats.shape


# ## Print columns

# In[ ]:


for ii in stats.columns:
    if 'mean' in ii:
        print(ii)

