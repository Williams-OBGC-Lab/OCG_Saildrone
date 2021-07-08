# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 2021 Oceanography Camp for Girls (OCG) Saildrone Lab
# Developed by Nancy Williams, Veronica Tamsitt, Nicola Guisewhite at University of South Florida College of Marine Science
#

# Funded by the National Science Foundation Office of Polar Programs Grant Number PLR2048840: https://www.nsf.gov/awardsearch/showAward?AWD_ID=2048840
#

# ## To Do List:
# * add more markdown in the form of instructions, pictures, pulling variables out into their own cell so girls know where they can make changes to the code
# * try plotting previous 8-day chl-a snapshot to see if it has better coverage for the eddy crossing
# * Check Veronica's carbon flux calculation is correct and add units (Nancy)
# * Switch to xarray for Saildrone dataset
# * Look at Chelle Gentemann's notebook and see if anything you want to bring in https://github.com/python4oceanography/ocean_python_tutorial/blob/master/notebooks/Tutorial_08_Xarray-Collocate_gridded_data_with_experiment.ipynb
#
# If time:
# * in figure titles and filenames, change the variables from using the first four characters (currently var[:4]) to instead cutting off at the first underscore
# * edit to make it easy to adjust time series x-axis limits
#
# ***
#
# ## Data Sources:
# * Saildrone 1-minute physical and ADCP data available from: https://data.saildrone.com/data/sets/antarctica-circumnavigation-2019
# (login required, so cannot be accessed using an FTP. Will need to download ahead)
# * Saildrone hourly-ish CO2, pH data available from: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0221912
# * Satellite Chlorophyll: https://neo.sci.gsfc.nasa.gov/view.php?datasetId=MY1DMW_CHLORA&year=2019
# * SSH: https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview 
# (login required for chla and SSH, download ahead of time. Can also be downloaded using motuclient, login also required https://github.com/clstoulouse/motu-client-python)
#
# ***
#
# ## Structure of this lesson
# This lesson will guide you through loading data from the Saildrone 2019 Antarctic circumnavigation, exploring and plot the data, compare the Saildrone data to remotely sensed observations from satellites, and explore the relationships between different ocean variables. Along the way you will have the opportunity to manipulate the data and make some changes to the plots, and hopefully learn a little bit of Python code along the way.
# 1. Loading the Saildrone data
# 2. Mapping the Saildrone circumnavigation path and plotting physical data along the path
# 3. Comparing the Saildrone data to maps of satellite sea surface height and chlorophyll-a
# 4. Time series analysis of the Saildrone data, looking at the relationship between air-sea carbon fluxes and other ocean and atmospheric variables

# ### Load Python modules
#
# Before we start working with the data, the first step of any Python script is to import specific modules, these will be the toolkits you need to load, analyse, and plot the data. If you're interested and want to learn more about how Python modules work, check out this [link](https://www.w3schools.com/python/python_modules.asp)


# +
# Import the tools you need

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
from datetime import datetime
import plotly.graph_objects as go

# add something
# -

# ### Define file paths
# Next, we'll define file paths so that the code knows where to find the data files and where to save output, like figures

output_dir = 'Output/'
data_dir = 'Data/'

# ## 1. Load Saildrone data
#
# Run the following code to download the Saildrone carbon data directly from the web. Text will pop up below showing the progress, speed, and time of the data download.

os.chdir(data_dir) # Change the directory to the `Data/` folder
# Curl downloads the data files directly from the web and shows you the status while it works. 
# `!` at the beginning of the line tells you that this command is a unix shell command (not python code)
# ! curl -o 32DB20190119_ASV_Saildrone1020_Antarctic_Jan2019_Aug2019.csv https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0221912/32DB20190119_ASV_Saildrone1020_Antarctic_Jan2019_Aug2019.csv
os.chdir("..") # Use ".." to move back up one directory now that we've imported the data

# Next we will import the data file into our Python workspace into a data structure called `Saildrone_CO2`

Saildrone_CO2 = pd.read_csv(
    (data_dir + '32DB20190119_ASV_Saildrone1020_Antarctic_Jan2019_Aug2019.csv'),
    header=4,
    na_values=-999,
)

# To check that the data has imported correctly and show a list of the variables included in `Saildrone_CO2`, you can just enter `Saildrone_CO2` as a Python command and it will print details of the data structure as shown below.

Saildrone_CO2

# As well as the Saildrone carbon data, there is another data file containing one-minute averaged physical data that we will also import into the work space (this one has already been added to the data directory ahead of time).

ds = xr.open_dataset(data_dir + 'saildrone-gen_5-antarctica_circumnavigation_2019-sd1020-20190119T040000-20190803T043000-1_minutes-v1.1620360815446.nc')
Saildrone_phys = ds.to_dataframe()
Saildrone_phys

# ## 2. Map the Saildrone path
#
# Great, now the Saildrone data are all loaded into our workspace we can make a map showing the path of the Saildrone around the Antarctic continent.
#
# Before we do that, we are going to import data of the ocean fronts of the Antarctic Circumpolar Current, so we can show the fronts on the map and see where the Saildrone is relative to the fronts. These fronts show where there are sharp gradients between water with different properties (e.g. temperature, salinity, surface nutrient concentrations), and the regions between the fronts form 'zones' with similar properties. 

stf = pd.read_csv(data_dir + 'fronts/stf.txt', header=None, sep='\s+', 
                  na_values='%', names=['lon','lat'])
saf = pd.read_csv(data_dir + 'fronts/saf.txt', header=None, sep='\s+', 
                  na_values='%', names=['lon','lat'])
pf = pd.read_csv(data_dir + 'fronts/pf.txt', header=None, sep='\s+', 
                 na_values='%', names=['lon','lat'])
saccf = pd.read_csv(data_dir + 'fronts/saccf.txt', header=None, sep='\s+', 
                    na_values='%', names=['lon','lat'])
sbdy = pd.read_csv(data_dir + 'fronts/sbdy.txt', header=None, sep='\s+', 
                   na_values='%', names=['lon','lat'])

# Now finally we can make a map the Saildrone track and ACC fronts

# +
# Make the "bones" of the figure
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -30],ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
ax.gridlines()

# Compute a circle in axes coordinates, which we can use as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Plot the ACC fronts in various colors
ax.set_boundary(circle, transform=ax.transAxes)
#plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
#         label = 'Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label = 'Subantarctic Front')
#plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
#         label = 'Polar Front')
#plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
#         label = 'Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label = 'Southern Boundary of ACC')

# Plot the Saildrone in black dots
plt.scatter(Saildrone_phys.longitude, Saildrone_phys.latitude,
           transform=ccrs.PlateCarree(), c='black', s=3, label='Saildrone', zorder=1000)

# Turn on the legend
plt.legend()

# Save the figure in the output folder
plt.title('2019 Saildrone Antarctic Circumnavigation Track')
plt.savefig(output_dir + 'SaildroneMap' + '.jpg') # Changing the suffix will change the format
plt.show()
# -

# Congratulations! You've now made a map of the Saildrone circumnavigation track and it has been saved in the output folder as `SaildroneMap.jpg`.
#
# **Q: why is it useful to compare the Saildrone path with the position of the ocean fronts? How many different zones (between the ocean fronts) does the Saildrone pass through during it's circumnavigation?**
#
# We can make our map more interesting by using coloured scatter points to show a variable from the Saildrone data on the map. In the following code cell we define which data variable to plot on the map, which colormap to use to plot the data (see different colormap options [here](https://matplotlib.org/stable/tutorials/colors/colormaps.html)), and the lower and upper data values to use for the colormap. 

var = 'TEMP_CTD_RBR_MEAN'
cmap_1 = 'bwr'
v_min = -2
v_max = 15

# +
# Make the "bones" of the figure
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -30],ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
ax.gridlines()

theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Plot the ACC fronts in various colors
ax.set_boundary(circle, transform=ax.transAxes)
#plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
#         label = 'Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label = 'Subantarctic Front')
#plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
#         label = 'Polar Front')
#plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
#         label = 'Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label = 'Southern Boundary of ACC')

# Plot the Saildrone data
plt.scatter(Saildrone_phys.longitude, Saildrone_phys.latitude, 
            c=Saildrone_phys[var], cmap=cmap_1,
            transform=ccrs.PlateCarree(), s=5, vmin=v_min,vmax=v_max, zorder=1000)

# Turn on the legend
plt.legend()
cb1 = plt.colorbar()

# Save the figure in the output folder
plt.title('2019 Saildrone ' + var[:4])
plt.savefig(output_dir + var[:4] + 'SaildroneMap' + '.jpg') # Changing the suffix will change the format
plt.show()
# -

# ## 3. Comparing Saildrone with Satellite observations
#
# Now let's make a map of some satellite data to show the Saildrone crossing an ocean eddy.
#
# First we need to load in a single daily satellite sea surface height data file from Feb 10th 2019, the day the Saildrone crossed a large eddy.


satellite_ssh = xr.open_dataset(data_dir + 'ssh_2019_02_09.nc')


# Now plot the Saildrone path on a map of sea surface height for a region surrounding the Saildrone on Feb 10th

#set plot parameters (contour levels, colormap etc)
levels_1 = np.arange(-1.2,0.8,0.1) #contour levels
cmap_1 = 'viridis' #contour map colormap
c1 = 'black' #Saildrone track color

# +
#finding position of Saildrone on Feb 10
time_index = np.argwhere(Saildrone_phys.time.values==np.datetime64('2019-02-10'))[0]
tlon = Saildrone_phys.longitude.values[time_index]
tlat = Saildrone_phys.latitude.values[time_index]

#make a contour plot of satellite ssh
xr.plot.contourf(satellite_ssh.sla[0,:,:],levels=levels_1,cmap=cmap_1,size=8,aspect=2)
xr.plot.contour(satellite_ssh.sla[0,:,:],levels=levels_1,colors='k',linewidths=0.75)
plt.xlim(tlon+360-5,tlon+360+5)
plt.ylim(tlat-5,tlat+5)

#add Subantarctic Front
plt.plot(saf['lon']+365, saf['lat'], color='Orange', linewidth=3, label = 'Subantarctic Front')

#add Saildrone track
plt.scatter(Saildrone_phys.longitude+360, Saildrone_phys.latitude, c=c1, s=3, label='Saildrone', zorder=1000)
plt.legend()

#give the plot a title and save figure in the output folder
plt.title('Saildrone path across an eddy on Feb 10th')
plt.savefig(output_dir + 'Sea_surface_height_Saildrone_Feb10' + '.jpg')

# -

# ## Ocean eddies can be identified by closed rings of constant absolute dynamic topography (this is the anomaly in sea surface height from average sea level in meters, which represents changes in pressure). You can see the Saildrone's path crossing near the center of an eddy.
#
# We can do the same thing with satellite chlorophyll-a data. The chlorophyll-a data gives an approximate estimate of the relative phytoplankton biomass (in units of mg/m<sup>3</sup>) at the sea surface in different locations. 

#load satellite chl-a data file
satellite_chla = xr.open_dataset(data_dir + 'A20190412019048.L3m_8D_CHL_chlor_a_4km.nc')

# Here you can edit parameters (colors, range etc) for the map

#set plot parameters (contour levels, colormap etc)
levels_1 = np.arange(0,1.0,0.01) #contour levels
cmap_1 = 'YlGnBu' #contour map colormap
c1 = 'black' #Saildrone track color

# +
#make a contour plot of chl-a data 
satellite_chla.chlo_a.values[satellite_chla.chlo_a>1000] = np.nan
xr.plot.contourf(satellite_chla.chlo_a, levels = levels_1, cmap=cmap_1,size=8,aspect=2)
plt.xlim(tlon-5,tlon+5)
plt.ylim(tlat-5,tlat+5)

#add Saildrone track
plt.scatter(Saildrone_phys.longitude, Saildrone_phys.latitude, c=c1, s=3, label='Saildrone', zorder=1000)
plt.legend()

#save figure
plt.title('Saildrone path and chlorophyll-a concentration')
plt.savefig(output_dir + 'Sea_surface_chlorophylla_Saildrone_Feb10' + '.jpg')
# -

# Now we can add the Saildrone data observations on the map to start to see if there is a relationship between the satellite observations and what the Saildrone measured directly. Note that the Saildrone took a few days to cross this region, while the satellite data shown here is a snapshot for a single day, so it can be tricky to compare the two types of data because the Saildrone is moving in space AND time.

#choose which variable to plot
var = 'TEMP_CTD_RBR_MEAN'
#set minimum and maximum colorbar limits
v_min = 6
v_max = 12
#choose colormap for map
cmap_1 = 'viridis'
#choose colormap for Saildrone variable
cmap_2 = 'RdBu_r'

# +
#make a contour plot of satellite ssh
xr.plot.contourf(satellite_ssh.adt[0,:,:],levels=np.arange(-1.2,0.8,0.1),cmap=cmap_1,size=8,aspect=2)
xr.plot.contour(satellite_ssh.adt[0,:,:],levels=np.arange(-1.2,0.8,0.1),colors='black',linewidths=0.75)
plt.xlim(tlon+360-5,tlon+360+5)
plt.ylim(tlat-5,tlat+5)

#add Saildrone data scattered on top
plt.scatter(Saildrone_phys.longitude+360, Saildrone_phys.latitude, c=Saildrone_phys[var], s=15, cmap = cmap_2,
            vmin=v_min,vmax=v_max,label='Saildrone', zorder=1000)
plt.legend()
plt.colorbar()

#add title and save figure
plt.title('Saildrone ' + var[:4] + ' across an eddy on Feb 10th')
plt.savefig(output_dir + 'Sea_surface_height_Saildrone_' + var[:4] + '_Feb10' + '.jpg')
# -

# ## 4. Time series analysis 
# Now that we've plotted some Saildrone and satellite data on maps to see how different ocean variables are related, there are other ways we can look at the relationship between variables. 
#
# This includes scatter plots, which is a useful way to compare data from two variables collected at the same time and location to look for a relationship. In our case, we can compare two different variables collected simulatneously by the Saildrone. 

# +
#choose two variables from the Saildrone to compare
var1 = 'TEMP_CTD_RBR_MEAN'
var2 = 'O2_CONC_RBR_MEAN'

#choose lower and upper limits for the two variables for plotting
var1_min = -2
var1_max = 18
var2_min = 230
var2_max = 340

# +
#create scatter plot
plt.figure(figsize=(12,8))
plt.scatter(Saildrone_phys[var1], Saildrone_phys[var2], s=10)
plt.xlim(var1_min,var1_max)
plt.ylim(var2_min,var2_max)
plt.xlabel(var1)
plt.ylabel(var2)
plt.grid()

#add title and save figure
plt.title('Saildrone '+ var1[:4] + ' vs ' + var2[:4])
plt.savefig(output_dir + 'Saildrone_' + var1[:4] + '_vs_' + var2[:4] + '.jpg')
# -

# If we want to look at the relationship between more than two variables, one way we can look at this is by using a third variable to change the color of the scatter plot points.
#
# In this example, the scatter plot shows the same variable 1 and 2 from the Saildrone on the x and y axes as above, but we can choose a third Saildrone variable as the color of the scatter plot points.

# +
#choose a third variable 
var3 = 'latitude'

#set lower and upper limits of variable 3 for plotting
var3_min = -65
var3_max = -40

# +
#create scatter plot
plt.figure(figsize=(12,8))
plt.scatter(Saildrone_phys[var1], Saildrone_phys[var2], c=Saildrone_phys[var3], 
            s=10, vmin = var3_min, vmax = var3_max)
plt.xlim(var1_min,var1_max)
plt.ylim(var2_min,var2_max)
plt.xlabel(var1)
plt.ylabel(var2)
plt.grid()
cbar = plt.colorbar()
cbar.set_label(var3)

#add title and save figure
plt.title('Saildrone '+ var1[:4] + ' vs ' + var2[:4])
plt.savefig(output_dir + 'Saildrone_' + var1[:4] + '_vs_' + var2[:4] + '_vs_' + var3[:4] + '.jpg')
# -

# Plot time series of wind speed and pressure

#calc wind speed from u and v winds
Saildrone_phys['WSPD'] = np.sqrt(np.power(Saildrone_phys['UWND_MEAN'],2)+np.power(Saildrone_phys['VWND_MEAN'],2))

# +
#input plot parameters

#variables to plot
var1 = 'WSPD'
var2 = 'BARO_PRES_MEAN'

#set x axis limits

# +
#plot time series
plt.figure(figsize=(12,5))
ax1 = plt.subplot(211)
ax1.plot(Saildrone_phys.time,Saildrone_phys[var1])
plt.xlim(Saildrone_phys.time.values[0],Saildrone_phys.time.values[-1])
plt.ylabel(var1)

ax2 = plt.subplot(212)
ax2.plot(Saildrone_phys.time,Saildrone_phys[var2])
plt.xlim(Saildrone_phys.time.values[0],Saildrone_phys.time.values[-1])
plt.ylabel(var2)
plt.show()
# -
# Next, we can calculate the flux of carbon between the ocean and the atmosphere based on the difference in pCO2 between the atmosphere and the ocean. 


# +
#constants for CO2 flux calculation
#ocean/atmosphere variables needed as inputs
T = Saildrone_CO2['SST (C)'] #sea surface temperature
S  = Saildrone_CO2['Salinity'] #sea surface salinity
u = Saildrone_CO2['WSPD (m/s)'] #surface wind speed
dpCO2 = Saildrone_CO2['dpCO2'] #difference between ocean and atmosphere pCO2

#1. Calculate the transfer velocity (Wanninkhof et al. 2014)
#Schmidt number as a function of temperature 
Sc = 2116.8-136.25*T  + 4.7353*np.power(T,2) - 0.092307*np.power(T,3) + 0.000755*np.power(T,4)
K = 0.251*(u*u)*np.power((Sc/660),-0.5)
K = K

#2. calculate solubility constant as a function of temperature and salinity 
T_K = T + 273.15
K0 = -58.0931 + ( 90.5069*(100.0 /T_K) ) \
    + (22.2940 * (np.log(T_K/100.0))) + (S * (0.027766 +  ( (-0.025888)*(T_K/100.0)) \
    + (0.0050578*( (T_K/100.0)*(T_K/100.0) ) ) ) )
a = np.exp(K0)

#CO2 flux equation
Saildrone_CO2['FCO2'] = 0.24 * K * a * dpCO2  #FCO2 = K*a(dpCO2)
# -

# Let's plot the time series of carbon fluxes together with the time series of wind speed to see how they are related. Sign of FCO2?

# +
#variables to plot
var1 = 'WSPD (m/s)'
var2 = 'FCO2'

#colors
c1 = 'darkblue'
c2 = 'darkorange'

# +
#convert date and time from Saildrone_CO2 file to numpy datetime64 array
date_object = np.empty(len(Saildrone_CO2['Date'])).astype(datetime)
for t in range(len(Saildrone_CO2['Date'])):
  dt = Saildrone_CO2['Date'].values[t]
  tm = Saildrone_CO2['Time'].values[t]
  date_object[t] = np.datetime64(datetime.strptime(dt+' '+tm,'%m/%d/%Y %H:%M'))
Saildrone_CO2['datetime'] = date_object #save to dataframe

#plot time series
fig, ax1 = plt.subplots(figsize=(12,5))

#y axis 1
ax1.plot(Saildrone_CO2['datetime'],Saildrone_CO2[var1],color=c1)
ax1.set_xlabel('date')
ax1.set_ylabel(var1, color=c1)
ax1.tick_params(axis='y', labelcolor=c1)

#y axis 2
ax2 = ax1.twinx()
ax2.plot(Saildrone_CO2['datetime'],-Saildrone_CO2[var2],color=c2)
ax2.set_xlabel('date')
ax2.set_ylabel(var2, color=c2)
ax2.tick_params(axis='y', labelcolor=c2)

ax2.plot([Saildrone_CO2['datetime'].values[0], Saildrone_CO2['datetime'].values[-1]],[0,0],
         color='black', linewidth=0.5)
plt.xlim(Saildrone_CO2['datetime'][0],Saildrone_CO2['datetime'][1800])
fig.tight_layout()
plt.show()
# -


