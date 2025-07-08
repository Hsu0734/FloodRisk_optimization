"""
Distance to streams analysis
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

# distance to streams analysis
flow_accu = wbe.d8_flow_accum(dem, out_type='cells')
streams = wbe.new_raster(flow_accu.configs)
for row in range(flow_accu.configs.rows):
    for col in range(flow_accu.configs.columns):
        elev = flow_accu[row, col]
        if elev >= 15.0:
            streams[row, col] = 1.0
        elif elev == flow_accu.configs.nodata:
            streams[row, col] = flow_accu.configs.nodata
        else:
            streams[row, col] = 0.0

#streams = wbe.extract_streams(flow_accu, threshold=15)
distance = wbe.euclidean_distance(streams)
wbe.write_raster(distance, 'DEM_demo_distance_to_stream.tif', compress=True)

path_01 = '../00_data_source/DEM_demo_distance_to_stream.tif'
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
cbar.set_label('distance_to_stream (meters)')

plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()

Distance = []
for row in range(distance.configs.rows):
    for col in range(distance.configs.columns):
        elev = distance[row, col]
        if distance[row, col] != distance.configs.nodata:
            Distance.append(elev)

print(max(Distance))
print(min(Distance))