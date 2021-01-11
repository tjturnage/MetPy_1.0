# Copyright (c) 2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
NEXRAD Level 2 File
===================

Use MetPy to read information from a NEXRAD Level 2 (volume) file and plot
"""
import matplotlib.pyplot as plt
import numpy as np

from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_metpy_logo, add_timestamp


from metpy.plots import add_metpy_logo, add_timestamp, colortables
from metpy.calc import azimuth_range_to_lat_lon
import cartopy.crs as ccrs
import cartopy.feature as cfeature



###########################################

# Open the file
name = get_test_data('KTLX20130520_201643_V06.gz', as_file_obj=False)
f = Level2File(name)

#print(f.sweeps[0][0])

# From first sweep, extract "constant" rda properties such as lon/lat
# this is needed to map radar data to lon/lat
rda_info = f.sweeps[0][0]
rda_lon = rda_info.vol_consts.lon
rda_lat = rda_info.vol_consts.lat

crs = ccrs.PlateCarree(central_longitude=rda_lon)

###########################################

# Pull data out of the file
sweep = 0   # this sweep has reflectivitiy data

# First item in ray is header, which has azimuth angle
ref_az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
ref_hdr = f.sweeps[sweep][1][4][b'REF'][0]
ref_gates = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
ref_data = np.array([ray[4]['bREF'][1] for ray in f.sweeps[sweep]])
ref_data = np.ma.array(ref_data)
ref_data[np.isnan(ref_data)] = np.ma.masked
ref_lons,ref_lats = azimuth_range_to_lat_lon(ref_az, ref_gates*1000, rda_lon, rda_lat, geod=None)
    

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

###########################################
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
add_metpy_logo(fig, 190, 85, size='large')
for var_data, var_range, ax in zip((ref, rho), (ref_range, rho_range), axes):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs = var_range * np.sin(np.deg2rad(az[:, np.newaxis]))
    ylocs = var_range * np.cos(np.deg2rad(az[:, np.newaxis]))

    # Plot the data
    ax.pcolormesh(xlocs, ylocs, data, cmap='viridis')
    ax.set_aspect('equal', 'datalim')
    ax.set_xlim(-40, 20)
    ax.set_ylim(-30, 30)
    add_timestamp(ax, f.dt, y=0.02, high_contrast=True)

plt.show()
