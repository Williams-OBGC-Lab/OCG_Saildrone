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


import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
from datetime import datetime
#import plotly.graph_objects as go

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

# + active=""
# Now finally we can make a map the Saildrone track and northern and southern boundary fronts of the Antarctic Circumpolar Current. Note that it might take a few seconds for the code to finish generating the figure before it appears.

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
# **Q: why is it useful to compare the Saildrone path with the position of the fronts of the Antarctic Circumpolar Curreent? How much of the Saildrone path lies within the boundaries of the Antarctic Circumpolar Current?**
#
# We can make our map more interesting by using coloured scatter points to show a variable from the Saildrone data on the map. In the following code cell we define which data variable to plot on the map (defined as `var`), which colormap to use to plot the data (defined as `cmap_1`, see more colormap options [here](https://matplotlib.org/stable/tutorials/colors/colormaps.html)), and the lower and upper data values to use for the colormap (defined as `v_min` and `v_max`). Initially this code is set up to plot the sea surface temperature from the Saildrone physical data, but once you've made the figure below you can try changing `var` and re-run the code below to plot a different variable (to see a list of all the variables in `Saildrone_phys` you can run `Saildrone_phys.columns` in a code cell). If you plot a different variable you might also want to play with changing the colorbar and colorbar limits.

var = 'TEMP_CTD_RBR_MEAN'#which variable to plot?
cmap_1 = 'bwr'
v_min = -2
v_max = 15

# Now that the inputs for the map are defined the following code will create the map, similar to above.

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

# **Q: Can you describe the temperature variation along the Saildrone path? Is this related to the fronts of the Antarctic Circumpolar Current?**

# ## 3. Comparing Saildrone with satellite observations
#
# Now that we've looked at some of the Saildrone data on a map, it's useful to put the data into context by plotting the Saildrone path on maps of surface ocean properties obtained from satellites. In particular, satellite sea surface height (measured in meters relative to background sea level) gives us a measure of pressure differences at the ocean surface, and therefore the direction of the surface currents (similar to pressure lines on a weather map). Ocean eddies can be identified by closed rings of constant sea level anomaly (meters). We know that the Saildrone crossed an eddy on around Feb 10th 2019, so we can make a map of sea level anomaly on that day to see the path of the Saildrone across the eddy. 
#
# First we'll load the satellite data file, which is a single daily snapshot.


satellite_ssh = xr.open_dataset(data_dir + 'ssh_2019_02_09.nc')


# Now we can plot the Saildrone path on a map of sea level anomaly for a region surrounding the Saildrone location on Feb 10th. Again we'll set some of the inputs for the figure like the colormap and line color beforehand so you can try changing these if you'd like to.

levels_1 = np.arange(-0.6,0.4,0.05) #levels of sea level anomaly to contour
cmap_1 = 'viridis' #contour map colormap
c1 = 'magenta' #Saildrone track color

# Now make the map:

# +
#finding position of Saildrone on Feb 10
time_index = np.argwhere(Saildrone_phys.time.values==np.datetime64('2019-02-10'))[0]
tlon = Saildrone_phys.longitude.values[time_index]
tlat = Saildrone_phys.latitude.values[time_index]

#make a contour plot of satellite ssh
xr.plot.contourf(satellite_ssh.sla[0,:,:],levels=levels_1,cmap=cmap_1,size=8,aspect=2)
xr.plot.contour(satellite_ssh.sla[0,:,:],levels=levels_1,colors='k',linewidths=0.75)
plt.xlim(tlon+360-24,tlon+360+24)
plt.ylim(tlat-12,tlat+12)

#add Subantarctic Front
plt.plot(saf['lon']+365, saf['lat'], color='Orange', linewidth=3, label = 'Subantarctic Front')

#add Saildrone track
plt.scatter(Saildrone_phys.longitude+360, Saildrone_phys.latitude, c=c1, s=3, label='Saildrone', zorder=1000)
plt.legend()

#give the plot a title and save figure in the output folder
plt.title('Saildrone path across an eddy on Feb 10th')
plt.savefig(output_dir + 'Sea_surface_height_Saildrone_Feb10' + '.jpg')
# -

# **Q: What do you see in the sea level anomaly? Can you identify the eddies crossed by the Saildrone (magenta line)?**
#
# Next we are going to go on a choose your own eddy adventure!
# 1. Identify an eddy in the map above (hint: look for closed contours of sea level anomaly, it can be positive or negative) crossed by the Saildrone track
#
# 2. Note the latitude and longitude at the the center of the eddy you've chosen from the x and y axes and update the values for `eddy_longitude` and `eddy_latitude` to match your eddy center in the code cell below.
#
#

eddy_longitude = 212
eddy_latitude = -53

# Now that you've chosen an eddy, we can zoom in to look in more detail at the sea level anomaly. We can also add satellite chlorophyll-a data. The chlorophyll-a data gives an approximate estimate of the relative phytoplankton biomass (in units of mg/m<sup>3</sup>) at the sea surface. 
#
# Again we need to load in the satellite chlorophyll-a data first, this time an 8-day average around the time the Saildrone crossed the eddy.

satellite_chla = xr.open_dataset(data_dir + 'A20190412019048.L3m_8D_CHL_chlor_a_4km.nc')

# Here we are setting the inputs (colors, range etc) to plot the sea level anomaly (in black contours) and satellite chlorophyll-a data (in color) of your eddy on the same map:

levels_1 = np.arange(-0.6,0.4,0.05) #levels of sea level anomaly to contour
c1 = 'black' #sea level anomaly contour color
levels_2 = np.arange(0,1.0,0.01) #contour levels for chlorophyll-a
cmap_2 = 'YlGnBu' #chlorophyll-a contour map colormap
c2 = 'magenta' #Saildrone track color

# Make the map:

# +
satellite_chla.chlo_a.values[satellite_chla.chlo_a>1000] = np.nan
xr.plot.contour(satellite_ssh.sla[0,:,:],levels=levels_1,colors=c1,linewidths=0.75,size=8,aspect=1.3)
plt.contourf(satellite_chla.lon+360,satellite_chla.lat,satellite_chla.chlo_a, levels = levels_2, cmap=cmap_2)
plt.xlim(eddy_longitude-4,eddy_longitude+4)
plt.ylim(eddy_latitude-4,eddy_latitude+4)

#add colorbar
cbar = plt.colorbar()
cbar.set_label('chlorophyll-a concentration [mg m$^{-3}$]')

#add Saildrone track
plt.scatter(Saildrone_phys.longitude+360, Saildrone_phys.latitude, c=c2, s=3, label='Saildrone', zorder=1000)
plt.legend()

#save figure
plt.title('Saildrone path and chlorophyll-a concentration')
plt.savefig(output_dir + 'Sea_surface_chlorophylla_Saildrone_Feb10' + '.jpg')
# -

# **Q: There may be some gaps in the map of chlorophyll-a where there are no data (white areas), any ideas why this might be? Can you see the impact of eddies on chlorophyll-a concentrations? How do these compare with the sea level anomaly map above?**
#
# To take this one step further we can add the Saildrone data observations on a satellite map of your eddy to see if there is a relationship between the satellite observations and what the Saildrone measured directly in the ocean. Note that the Saildrone took a few days to cross this region, while the satellite data shown here is a snapshot for a single day, so it can be tricky to compare the two types of data because the Saildrone is moving in space AND time.
#
# We'll make a map of the sea level anomaly again as above, and choose a variable (`var`) from the Saildrone data to plot on top of the satellite map. As before we will start with sea surface temperature, but you can try changing the figure inputs in the code below to plot different variables and see which variables have a relationship to sea level anomaly.

var = 'TEMP_CTD_RBR_MEAN' #choose which variable to plot
var_name = 'Temperature (degrees C)' #name for variable to label colorbar
v_min = 6 #set minimum and maximum variable colorbar limits
v_max = 12
levels_1 = np.arange(-0.6,0.4,0.05) #levels of sea level anomaly to contour
cmap_1 = 'viridis'#choose colormap for sea level anomaly map
cmap_2 = 'RdBu_r' #choose colormap for Saildrone variable

# Make the map:

# +
#make a contour plot of satellite ssh
xr.plot.contourf(satellite_ssh.sla[0,:,:],levels=levels_1,cmap=cmap_1,size=8,aspect=1.5)
xr.plot.contour(satellite_ssh.sla[0,:,:],levels=levels_1,colors='black',linewidths=0.75)
plt.xlim(eddy_longitude-4,eddy_longitude+4)
plt.ylim(eddy_latitude-4,eddy_latitude+4)

#add Saildrone data scattered on top
plt.scatter(Saildrone_phys.longitude+360, Saildrone_phys.latitude, c=Saildrone_phys[var], s=15, cmap = cmap_2,
            vmin=v_min,vmax=v_max,label='Saildrone', zorder=1000)
plt.legend()
cbar = plt.colorbar()
cbar.set_label(var_name)

#add title and save figure
plt.title('Saildrone ' + var[:4] + ' across an eddy on Feb 10th')
plt.savefig(output_dir + 'Sea_surface_height_Saildrone_' + var[:4] + '_Feb10' + '.jpg')
# -

# **Q: What did you notice when you plotted temperature on the map of sea level anomaly? Are they related? Can you find any relationships between the sea level anomaly and other variables measured by the Saildrone?**

# ## 4. Time series analysis 
# Now that we've plotted some Saildrone and satellite data on maps to see how different ocean variables are related, there are other ways we can look at the relationship between ocean variables. 
#
# This includes time series and scatter plots, which are useful ways to compare data from two variables collected at the same time and location to look for a relationship. In our case, we can compare two different variables collected simultaneously by the Saildrone. 
#
# Choose two variables from the Saildrone to compare (and the upper and lower data limits) by setting the inputs `var1` and `var2` below. We'll start with sea surface temperature and oxygen concentration.

# +
var1 = 'TEMP_CTD_RBR_MEAN'
var2 = 'O2_CONC_RBR_MEAN'

#choose lower and upper limits for the two variables for plotting
var1_min = -2
var1_max = 18
var2_min = 230
var2_max = 340
# -

# First we'll plot the time series of the two variable for the entire Saildrone mission:

# +
plt.figure(figsize=(12,5))
ax1= plt.subplot(211)
ax1.plot(Saildrone_phys.time,Saildrone_phys[var1])
plt.xlim(Saildrone_phys.time.values[0],Saildrone_phys.time.values[-1])
plt.ylabel(var1)

ax2 = plt.subplot(212)
ax2.plot(Saildrone_phys.time,Saildrone_phys[var2])
plt.xlim(Saildrone_phys.time.values[0],Saildrone_phys.time.values[-1])
plt.ylabel(var2)
plt.show()
# -

# Next, we can make a scatter plot to compare the two variables, one on the x-axis and one on the y-axis:

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

# **Q: what do you notice about the pattern in the scatter plot? What does this imply about the relationship between sea surface temperature and oxygen concentration? Can you come up with any ideas of why temperature and oxygen concentrations are related? If you go back to the inputs defining `var1` and `var2` above and choose two different variables and re-run the code, can you find two different variables that are related? Is the relationship similar or different than the relationship between temperature and oxygen?**

# If we want to look at the relationship between more than two variables, one way to do this is by using a third variable to change the color of the scatter plot points.
#
# In this example, the scatter plot shows the same variable 1 and 2 from the Saildrone on the x and y axes as above, but we can choose a third Saildrone variable as the color of the scatter plot points. In this example we'll try Latitude as the third variable to begin with.

var3 = 'latitude'#choose a third variable 
var3_min = -65 #set lower and upper limits of variable 3 for plotting
var3_max = -45
cmap_1 = 'viridis' #colormap to use for third variable

# Now make a scatter plot, colored by the third variable:

# +
#create scatter plot
plt.figure(figsize=(12,8))
plt.scatter(Saildrone_phys[var1], Saildrone_phys[var2], c=Saildrone_phys[var3], 
            s=10, cmap = cmap_1, vmin = var3_min, vmax = var3_max)
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

# **Q: Is there a relationship between the third variable (color) and the two existing variables on the x and y axes? Try changing the third variable to something else to see if you get a different relationship. What did you find?**

# ***
#
# One thing we can investigate in the Saildrone time series data is how fluxes of carbon dioxide between the atmosphere and ocean are influenced by variations in properties of the atmosphere and ocean.
#
# Winds are an important driver of fluxes of carbon, so we can first compare the wind speed to the air-sea carbon flux.
# We can calculate the flux of carbon between the ocean and the atmosphere, which is based on the difference in pCO2 between the atmosphere and the ocean. 

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

# Let's plot the time series of carbon fluxes together with the time series of wind speed to see how they are related. **Nancy check sign/magnitude of FCO2?**

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
# **Q: What is the relationship between wind speed and air-sea carbon fluxes? How do you think wind impacts the exchange of carbon between the atmosphere and ocean? On what timescales are the wind speed and carbon fluxes varying? The magnitude and sign of the air-sea carbon fluxes are affected by the ocean as well, can you try comparing the carbon flux 'FCO2' time series to ocean variables from `Saildrone_CO2`? For example sea surface temperature (`SST (C)`), seawater pCO2 (`pCO2 SW (sat) uatm`). Remmember you can check the variable in the `Saildrone_CO2` file by running the Python command `Saildrone_CO2.columns`** 



