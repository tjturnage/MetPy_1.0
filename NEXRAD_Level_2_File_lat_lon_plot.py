# -*- coding: utf-8 -*-
# Copyright (c) 2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
NEXRAD Level 2 File
===================

Use MetPy to read information from a NEXRAD Level 2 (volume) file and
plot then convert to latitude / longitude coordinates
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_metpy_logo, add_timestamp, colortables
from metpy.calc import azimuth_range_to_lat_lon


###########################################

z_norm, z_cmap = colortables.get_with_range('NWSStormClearReflectivity', -30, 80)
v_norm, v_cmap = colortables.get_with_range('NWSVelocity', -30, 30)

# Open the file
name = get_test_data('KTLX20130520_201643_V06.gz', as_file_obj=False)
radar_file = Level2File(name)

# extract "constant" rda properties such as lon/lat.
# This is needed to map radar data to lon/lat.S
rda_name = radar_file.stid.decode('utf-8')
rda_info = radar_file.sweeps[0][0]
rda_lon = rda_info.vol_consts.lon
rda_lat = rda_info.vol_consts.lat

crs = ccrs.PlateCarree(central_longitude=rda_lon)

###########################################

def make_ticks(this_min,this_max):
    """
    Determines range of tick marks to plot based on a provided range of
    degree coordinates.

    Parameters
    ----------
           this_min   :  float
                         minimum value of either a lat or lon extent
           this_max   :  float
                         maximum value of either a lat or lon extent

    Returns
    -------
           tick_array :  float list
                         tick labels for plotting

    """

    tick_min = round(this_min) - 0.5
    tick_max = round(this_max) + 0.5

    # build an expanded list of tick values using the padding above
    init_tick_list = np.arange(tick_min,tick_max,1)
    # build list including only ticks within the specified range
    tick_array = []


    for element,tick in enumerate(init_tick_list):
        if init_tick_list[element] >= this_min and init_tick_list[element] <= this_max:
            tick_array.append(tick)
        else:
            pass

    return tick_array



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
    longitudes : numpy float array
        all longitudes corresponding to data points in decimal degrees
    latitudes : numpy float array
        all latitudes corresponding to data points in decimal degrees
    this_data : numpy float array
        data to plot

    """

    azimuth = np.array([ray[0].az_angle for ray in radar_file.sweeps[sweep]])
    hdr = radar_file.sweeps[sweep][1][4][moment][0]
    gates = np.arange(hdr.num_gates) * hdr.gate_width + hdr.first_gate
    moment_data = np.array([ray[4][moment][1] for ray in radar_file.sweeps[sweep]])
    this_data = np.ma.array(moment_data)
    this_data[np.isnan(this_data)] = np.ma.masked

    #archive level 2 gate units appear to be in km
    longitudes,latitudes = azimuth_range_to_lat_lon(azimuth, gates*1000,
                                                    rda_lon, rda_lat, geod=None)

    return longitudes, latitudes, this_data


def sweep_info(this_file,elevation_angle_limit=3):
    """
    Builds a dictionary of information for each sweep at or below the
    specified elevation_angle_limit. This is important because not
    all sweeps contain velocity and spectrum width data. Also, there can
    be multiple sweeps associated with the lowest elevation angle(s) when
    using SAILS/ MESO-SAILS / MRLE scan strategies.

    Parameters
    ----------
    this_file :        Level2File objectfile
        The currently open NEXRAD_Level_2 file

    elevation_limit :  integer or float, optional
        upper elevation angle limit to process data.
        Typically, greater elevation angle data.

    Returns
    -------
    file_sweeps_dict :      dictionary
        Contains information pertaining to each sweep in the file that
        can be retrieved later for plotting. Dictionary keys are integers
        representing the index number of the f.sweeps array.

    """
    file_sweeps_dict = {}
    for sweep in range(0,len(this_file.sweeps)):
        this_elevation =  this_file.sweeps[sweep][0][0].el_angle
        rounded_elevation = int(10*this_elevation)

        if this_elevation < elevation_angle_limit:
            try:
                # "test" array used only created here to assess whether
                # VEL data exist in the current sweep
                test = this_file.sweeps[sweep][1][4][b'VEL'][0]
                sweep_type = 'Doppler'
                longitudes,latitudes,this_data = build_radar_data_array(sweep=sweep,
                                                                 moment=b'VEL')
                file_sweeps_dict[sweep] = {'elevation': rounded_elevation,
                                  'sweep_type': sweep_type,
                                  'data':this_data, 'lons':longitudes,
                                  'lats':latitudes,
                                  'cmap': v_cmap,'vmin': -25,
                                  'vmax': 25}

            except:
                sweep_type = 'Surveillance'
                longitudes,latitudes,this_data = build_radar_data_array(sweep=sweep,
                                                                 moment=b'REF')
                file_sweeps_dict[sweep] = {'elevation': rounded_elevation,
                                  'sweep_type': sweep_type,
                                  'data':this_data, 'lons':longitudes,
                                  'lats':latitudes,
                                  'cmap': z_cmap, 'vmin':-30,
                                  'vmax': 80}

        else:
            pass

    return file_sweeps_dict

def plot_extent(longitudes,latitudes,rda_longitude,rda_latitude,fraction):

    """
    Uses longitude/latitude arrays to assess areal extent of data, then
    defines a fraction of this extent to plot. Note that the
    plot vertical aspect ratio will increase with latitude.

    Parameters
    ----------
    longitudes, latitudes : numpy arrays
        longitudes, latitudes associated with radar data points

    rda_longitude, rda_latitude : float
        location of radar in decimal degrees longitude,latitude

    fraction : float (between 0 and 1)
        fraction of full lon/lat domain axes to plot.
        Example: 0.5 means half the full range of lons/lats data,
        and roughly one quarter or the original area.


    Returns
    -------
    extent: floats tuple
       Consists of xmin, xmax, ymin, ymax representing
        (minimum/maximum) values of (longitudes/latitudes) in decimal degrees

    xticks: float list
        longitude labels for plot

    yticks: float list
        latitude labels for plot

    """
    longitude_offset = fraction * (longitudes.max() - rda_longitude)
    latitude_offset = fraction * (latitudes.max() - rda_latitude)
    xmin = rda_longitude - longitude_offset
    xmax = rda_longitude + longitude_offset
    ymin = rda_latitude - latitude_offset
    ymax = rda_latitude + latitude_offset

    # call the make_ticks function using now that fractional ranges for
    # latitude and longitude have been defined
    xticks = make_ticks(xmin,xmax)
    yticks = make_ticks(ymin,ymax)
    this_extent = (xmin,xmax,ymin,ymax)
    return this_extent, xticks, yticks


###########################################

# Build dictionary of desired sweeps
sweeps_dict = sweep_info(radar_file,elevation_angle_limit=2)

# Create data array mappe to longitude/latitude
lons, lats, data = build_radar_data_array()

# Define extent of desired plot and get associated ticks
extent, x_ticks, y_ticks = plot_extent(lons,lats,rda_lon,rda_lat,0.45)

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(11, 6),sharey=True,
                         subplot_kw={'projection': ccrs.PlateCarree()})
add_metpy_logo(fig, 575, 55, size='small')

plt.labelsize : 8
plt.tick_params(labelsize=8)

for y,a in zip([0,1],axes.ravel()):
    this_sweep = sweeps_dict[y]
    a.set_extent(extent, crs=ccrs.PlateCarree())
    a.tick_params(axis='both', labelsize=8)
    a.pcolormesh(this_sweep['lons'], this_sweep['lats'],
                 this_sweep['data'], cmap=this_sweep['cmap'],
                 vmin=this_sweep['vmin'],
                 vmax=this_sweep['vmax'],
                 transform=ccrs.PlateCarree())
    a.add_feature(cfeature.STATES, linewidth=0.5)

    a.set_aspect(1.25)
    a.xformatter = LONGITUDE_FORMATTER
    a.yformatter = LATITUDE_FORMATTER
    subplot_title = '{} {}'.format(rda_name,this_sweep['sweep_type'])
    a.set_title(subplot_title, fontsize=11)
    gl = a.gridlines(color='gray',alpha=0.5,draw_labels=True)
    gl.xlabels_top, gl.ylabels_right = False, False
    gl.xlabel_style, gl.ylabel_style = {'fontsize': 7}, {'fontsize': 7}
    gl.xlocator = mticker.FixedLocator(x_ticks)
    gl.ylocator = mticker.FixedLocator(y_ticks)
    add_timestamp(a, radar_file.dt, y=0.02, high_contrast=True)

plt.show()
