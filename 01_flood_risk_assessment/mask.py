"""
flow accumulation analysis
Author: Hanwen Xu
Date: Oct 01, 2023
"""

import whitebox_workflows as wbw
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 20
wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source'


# web read DEM data
dem = wbe.read_raster('Greve_5m.tif')
mask = wbe.read_raster('DEM_demo_buffer.tif')

# slope analysis
new_mask = wbe.new_raster(dem.configs)

'''for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        elev = dem[row, col]
        mask_elev = mask[row, col]
        if mask_elev == 0.0:
            new_mask[row, col] = dem.configs.nodata
        elif elev == dem.configs.nodata:
            new_mask[row, col] = dem.configs.nodata
        else:
            new_mask[row, col] = 1.0'''

for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        elev = dem[row, col]
        mask_elev = mask[row, col]
        if mask_elev == -32768.0:
            new_mask[row, col] = dem.configs.nodata
        elif mask_elev == 0.0:
            new_mask[row, col] = dem.configs.nodata
        else:
            new_mask[row, col] = 1.0

wbe.write_raster(new_mask, 'Greve_mask_5m.tif', compress=True)


# visualization
path_01 = '../00_data_source/Greve_mask_5m.tif'
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
cbar.set_label('Hanwen_mask')

plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()