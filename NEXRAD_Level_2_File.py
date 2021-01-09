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
# this is needed to map radar data to lon/lat
rda_info = f.sweeps[0][0]
rda_lon = rda_info.vol_consts.lon
rda_lat = rda_info.vol_consts.lat

crs = ccrs.PlateCarree(central_longitude=rda_lon)

sweeps = {}

###########################################

def build_radar_data_array(sweep=0,moment=b'REF'):
    """
    Requires
    ----------
    f: file object for NEXRAD_Level_2 file
    rda_lon, rda_lat:  longitude/latitude of radar in decimal degrees
    

    Parameters
    ----------
    sweep : integer, optional
        Selects sweep from the f.sweeps array
    moment : string, optional
        Some sweeps have velocity data, but not all due to split cut
        volume coverage patterns

    Returns
    -------
    az : numpy array
        all radials in the given sweep in degrees
    gates : numpy array
        all range gates along a radial in meters
    lons : numpy array
        
    lats : numpy array
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    """

    az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
    hdr = f.sweeps[sweep][1][4][moment][0]
    gates = np.arange(hdr.num_gates) * hdr.gate_width + hdr.first_gate
    moment_data = np.array([ray[4][moment][1] for ray in f.sweeps[sweep]])
    data = np.ma.array(moment_data)
    data[np.isnan(data)] = np.ma.masked
    lons,lats = azimuth_range_to_lat_lon(az, gates*1000, rda_lon, rda_lat, geod=None)
    
    return az, gates, lons, lats, data


def sweep_info(f,elevation_angle_limit=3):
    """
    Builds a dictionary consisting of information for each sweep at
    or below the 
    
    Parameters
    ----------
    f : file object
        The currently open NEXRAD_Level_2 file
    elevation_limit : integer or float, optional
        The elevation angle limit at which to process data. Typically,
        we don't care about plotting data at greater elevation angles.

    Returns
    -------
    sweeps_dict : dictionary
        Contains information pertaining to each sweep that can be
        retrieved later for plotting. Dictionary keys are integers
        representing the index number of the f.sweeps array, including:
            
        

    """
    sweeps_dict = {}
    for s in range(0,len(f.sweeps)):
        this_elevation =  f.sweeps[s][0][0].el_angle
        rounded_elevation = int(10*this_elevation)

        if this_elevation < elevation_angle_limit:
            try:
                # "test" array used only here to determine if
                # VEL data exist in the current sweep
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



def plot_extent(lons,lats,rda_lon,rda_lat,fraction):

    """
    Based on lats,lons arrays that indicate areal extent of data,
    Applies a fraction to this extent for plotting purposes
    
    Parameters
    ----------
    lons, lats : numpy arrays
        lons, lats of data points

    rda_lon, rda_lat : float
        location of radar in decimal degrees longitude,latitude

    fraction : float (between 0 and 1)
        fraction of full lon/lat domain axes to plot.
        Example: 0.5 means half the full range of lons/lats data,
        and roughly one quarter or the original area.


    Returns
    -------
    Tuple of extent to plot centered around rda location. Consists of:

    xmin, xmax : floats
        Minimum, maximum longitudes to plot in decimal degrees

    ymin, ymax : floats
        Minimum, maximum latitudes to plot in decimal degrees


    """
    lon_offset = fraction * (lons.max() - rda_lon)
    lat_offset = fraction * (lats.max() - rda_lat)
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
