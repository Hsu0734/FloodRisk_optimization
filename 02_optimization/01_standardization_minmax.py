"""
Slope analysis
Author: Hanwen Xu
Date: Oct 01, 2023
"""

import whitebox_workflows as wbw
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source'

# web read DEM data
slope = wbe.read_raster('DEM_demo_slope.tif')
slope_standard = wbe.new_raster(slope.configs)

HAND = wbe.read_raster('DEM_demo_HAND.tif')
HAND_standard = wbe.new_raster(HAND.configs)

distance_to_stream = wbe.read_raster('DEM_demo_distance_to_stream.tif')
distance_to_stream_standard = wbe.new_raster(distance_to_stream.configs)

DWI = wbe.read_raster('DEM_demo_DWI.tif')
DWI_standard = wbe.new_raster(DWI.configs)

velocity = wbe.read_raster('DEM_demo_velocity.tif')
velocity_standard = wbe.new_raster(velocity.configs)

Slope_value = []
HAND_value = []
Distance_to_stream_value = []
DWI_value = []
Velocity_value = []

for row in range(slope_standard.configs.rows):
    for col in range(slope_standard.configs.columns):
        slope_standard[row, col] = slope[row, col]
        HAND_standard[row, col] = HAND[row, col]
        distance_to_stream_standard[row, col] = distance_to_stream[row, col]
        DWI_standard[row, col] = DWI[row, col]
        velocity_standard[row, col] = velocity[row, col]

        elev_01 = slope_standard[row, col]
        elev_02 = HAND_standard[row, col]
        elev_03 = distance_to_stream_standard[row, col]
        elev_04 = DWI_standard[row, col]
        elev_05 = velocity_standard[row, col]

        if elev_01 != slope_standard.configs.nodata:
            Slope_value.append(elev_01)
        if elev_02 != HAND_standard.configs.nodata:
            HAND_value.append(elev_02)
        if elev_03 != distance_to_stream_standard.configs.nodata:
            Distance_to_stream_value.append(elev_03)
        if elev_04 != DWI_standard.configs.nodata:
            DWI_value.append(elev_04)
        if elev_05 != velocity_standard.configs.nodata:
            Velocity_value.append(elev_05)

print(f'max slope:{max(Slope_value)}')
print(f'min slope:{min(Slope_value)}')
print(f'max HAND:{max(HAND_value)}')
print(f'min HAND:{min(HAND_value)}')
print(f'max Distance_to_stream:{max(Distance_to_stream_value)}')
print(f'min Distance_to_stream:{min(Distance_to_stream_value)}')
print(f'max DWI:{max(DWI_value)}')
print(f'min DWI:{min(DWI_value)}')
print(f'max velocity:{max(Velocity_value)}')
print(f'min velocity:{min(Velocity_value)}')


for row in range(slope_standard.configs.rows):
    for col in range(slope_standard.configs.columns):
        standard_01 = slope_standard[row, col]
        if standard_01 != slope_standard.configs.nodata:
            slope_standard[row, col] = (slope_standard[row, col] - min(Slope_value)) / (max(Slope_value) - min(Slope_value))
wbe.write_raster(slope_standard, 'DEM_demo_standard_slope.tif', compress=True)

for row in range(HAND_standard.configs.rows):
    for col in range(HAND_standard.configs.columns):
        standard_02 = HAND_standard[row, col]
        if standard_02 != HAND_standard.configs.nodata:
            HAND_standard[row, col] = (HAND_standard[row, col] - min(HAND_value)) / (max(HAND_value) - min(HAND_value))
wbe.write_raster(HAND_standard, 'DEM_demo_standard_HAND.tif', compress=True)

for row in range(distance_to_stream_standard.configs.rows):
    for col in range(distance_to_stream_standard.configs.columns):
        standard_03 = distance_to_stream_standard[row, col]
        if standard_03 != distance_to_stream_standard.configs.nodata:
            distance_to_stream_standard[row, col] = (distance_to_stream_standard[row, col] - min(Distance_to_stream_value))\
                                                    / (max(Distance_to_stream_value) - min(Distance_to_stream_value))
wbe.write_raster(distance_to_stream_standard, 'DEM_demo_standard_distance.tif', compress=True)

for row in range(DWI_standard.configs.rows):
    for col in range(DWI_standard.configs.columns):
        standard_04 = DWI_standard[row, col]
        if standard_04 != DWI_standard.configs.nodata:
            DWI_standard[row, col] = (DWI_standard[row, col] - min(DWI_value)) / (max(DWI_value) - min(DWI_value))
wbe.write_raster(DWI_standard, 'DEM_demo_standard_DWI.tif', compress=True)

for row in range(velocity_standard.configs.rows):
    for col in range(velocity_standard.configs.columns):
        standard_05 = velocity_standard[row, col]
        if standard_05 != velocity_standard.configs.nodata:
            velocity_standard[row, col] = (velocity_standard[row, col] - min(Velocity_value)) / (max(Velocity_value) - min(Velocity_value))
wbe.write_raster(velocity_standard, 'DEM_demo_standard_velocity.tif', compress=True)

# weighted averaging
weighted_averaging = wbe.raster_calculator(expression="'slope_standard'-'HAND_standard'-'distance_to_stream_standard'+'DWI_standard'+'velocity_standard'",
                                           input_rasters=[slope_standard, HAND_standard, distance_to_stream_standard, DWI_standard, velocity_standard])
wbe.write_raster(weighted_averaging, 'DEM_demo_standard_weighted.tif', compress=True)


# visualization
path_01 = '../00_data_source/DEM_demo_standard_slope.tif'
data_01 = rs.open(path_01)
dem_array = data_01.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_01, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Slope Standardization')
plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()



path_02 = '../00_data_source/DEM_demo_standard_HAND.tif'
data_02 = rs.open(path_02)
dem_array = data_02.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_02, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('HAND Standardization')
plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()



path_03 = '../00_data_source/DEM_demo_standard_distance.tif'
data_03 = rs.open(path_03)
dem_array = data_03.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_03, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('distance to stream Standardization')
plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()



path_04 = '../00_data_source/DEM_demo_standard_DWI.tif'
data_04 = rs.open(path_04)
dem_array = data_04.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_04, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('DWI Standardization')
plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()


path_05 = '../00_data_source/DEM_demo_standard_velocity.tif'
data_05 = rs.open(path_05)
dem_array = data_05.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_05, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Velocity Standardization')
plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()


path = '../00_data_source/DEM_demo_standard_weighted.tif'
data = rs.open(path)
dem_array = data.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Weighted Standardization')
plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()