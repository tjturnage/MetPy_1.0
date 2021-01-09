# Copyright (c) 2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
NEXRAD Level 2 File
===================

Use MetPy to read information from a NEXRAD Level 2 (volume) file and plot
"""
import matplotlib.pyplot as plt
#from datetime import datetime
import numpy as np

from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_metpy_logo, add_timestamp, colortables
from metpy.calc import azimuth_range_to_lat_lon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys

###########################################

try:
    os.listdir('/usr')
    windows = False
    sys.path.append('/data/scripts/resources')
except:
    windows = True
    sys.path.append('C:/data/scripts/resources')

from custom_cmaps import plts

#z_norm, z_cmap = colortables.get_with_range('NWSStormClearReflectivity', -30, 80)
#v_norm, v_cmap = colortables.get_with_range('NWSVelocity', -30, 30)

# Open the file
name = get_test_data('KTLX20130520_201643_V06.gz', as_file_obj=False)
f = Level2File('C:/data/KGRR20190912_002621_V06')

# extract "constant" rda properties such as lon/lat
rda_info = f.sweeps[0][0]
rda_lon = rda_info.vol_consts.lon
rda_lat = rda_info.vol_consts.lat

crs = ccrs.PlateCarree(central_longitude=rda_lon)

sweeps = {}

###########################################


def build_radar_data_array(sweep=0,moment=b'REF'):


    az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
    hdr = f.sweeps[sweep][1][4][moment][0]
    gates = np.arange(hdr.num_gates) * hdr.gate_width + hdr.first_gate
    moment_data = np.array([ray[4][moment][1] for ray in f.sweeps[sweep]])
    data = np.ma.array(moment_data)
    data[np.isnan(data)] = np.ma.masked
    lons,lats = azimuth_range_to_lat_lon(az, gates*1000, rda_lon, rda_lat, geod=None)

    
    return az, gates, lons, lats, data


def sweep_info(f,elevation_limit=3):
    sweeps_dict = {}
    for s in range(0,len(f.sweeps)):
        this_elevation =  f.sweeps[s][0][0].el_angle
        rounded_elevation = int(10*this_elevation)

        if this_elevation < elevation_limit:
            try:
                test = f.sweeps[s][1][4][b'VEL'][0]
                sweep_type = 'Doppler'
                az,gates,lons,lats,data = build_radar_data_array(sweep=s,moment=b'VEL')
                sweeps_dict[s] = {'elevation': rounded_elevation,
                                  'sweep_type': sweep_type,
                                  'data':data, 'lons':lons,'lats':lats,
                                  'cmap': plts['Vel']['cmap'],'vmin': -120,
                                  'vmax': 120}
            except:
                sweep_type = 'Surveillance'
                az,gates,lons,lats,data = build_radar_data_array(sweep=s,moment=b'REF')
                sweeps_dict[s] = {'elevation': rounded_elevation,
                                  'sweep_type': sweep_type,
                                  'data':data, 'lons':lons,'lats':lats,
                                  'cmap': plts['Ref']['cmap'], 'vmin':-30,
                                  'vmax': 80}

        else:
            pass

    return sweeps_dict

def plot_extent(lons,lats,rda_lon,rda_lat,factor):
    """
    Parameters
    ----------
    lons : TYPE
        DESCRIPTION.
    lats : TYPE
        DESCRIPTION.
    rda_lon : TYPE
        DESCRIPTION.
    rda_lat : TYPE
        DESCRIPTION.
    factor : TYPE
        DESCRIPTION.

    Returns
    -------
    xmin : TYPE
        DESCRIPTION.
    xmax : TYPE
        DESCRIPTION.
    ymin : TYPE
        DESCRIPTION.
    ymax : TYPE
        DESCRIPTION.

    """
    lon_offset = factor * (lons.max() - rda_lon)
    lat_offset = factor * (lats.max() - rda_lat)
    xmin = rda_lon - lon_offset 
    xmax = rda_lon + lon_offset
    ymin = rda_lat - lat_offset 
    ymax = rda_lat + lat_offset
    return (xmin,xmax,ymin,ymax)


sweeps_dict = sweep_info(f,elevation_limit=1)
az, gates, lons, lats, data = build_radar_data_array()
extent = plot_extent(lons,lats,rda_lon,rda_lat,0.3)

fig, axes = plt.subplots(1, 2, figsize=(11, 6),subplot_kw={'projection': ccrs.PlateCarree()})
add_metpy_logo(fig, 20, 20, size='small')

plt.suptitle('{}'.format(str(f.stid)), fontsize=14)

#plt.titlesize : 24
#plt.suptitlesize : 20
plt.labelsize : 8
plt.tick_params(labelsize=8)


subset = ['AzShear_Storm','DivShear_Storm']
for y,a in zip([0,1],axes.ravel()):
    this_sweep = sweeps_dict[y]
    this_sweep['sweep_type']
    #this_title = plts[y]['title']
    a.set_extent(extent, crs=ccrs.PlateCarree())
    a.tick_params(axis='both', labelsize=8)
    a.pcolormesh(this_sweep['lons'], this_sweep['lats'], 
                 this_sweep['data'], cmap=this_sweep['cmap'],
                 vmin=this_sweep['vmin'],
                 vmax=this_sweep['vmax'],
                 transform=ccrs.PlateCarree())
    a.add_feature(cfeature.STATES, linewidth=0.5)
    a.set_aspect(1.25)
    add_timestamp(a, f.dt, y=0.02, high_contrast=True)

plt.savefig('C:/data/scripts/KGRR_metpy.png', format='png', bbox_inches="tight", dpi=150)
