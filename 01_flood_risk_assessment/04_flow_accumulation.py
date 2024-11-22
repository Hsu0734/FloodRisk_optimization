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
dem = wbe.read_raster('Hanwen_2m.tif')

# slope analysis
flow_accu = wbe.d8_flow_accum(dem, out_type='cells')
wbe.write_raster(flow_accu, 'DEM_demo_flow_accu.tif', compress=True)
Flow_accum_value = []
for row in range(flow_accu.configs.rows):
    for col in range(flow_accu.configs.columns):
        elev = flow_accu[row, col]
        if elev != flow_accu.configs.nodata:
            Flow_accum_value.append(elev)

# print(Flow_accum_value)
print(max(Flow_accum_value))
print(min(Flow_accum_value))

# visualization
path_01 = '../00_data_source/DEM_demo_flow_accu.tif'
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
cbar.set_label('Flow accumulation')

plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()