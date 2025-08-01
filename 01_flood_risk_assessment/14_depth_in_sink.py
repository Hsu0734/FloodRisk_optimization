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

plt.rcParams['font.size'] = 20
wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source'


# web read DEM data
dem = wbe.read_raster('Greve_5m.tif')
# dem_00 = wbe.fill_missing_data(dem, exclude_edge_nodata= True)
depth = wbe.depth_in_sink(dem, zero_background=False)
sink_area = wbe.new_raster(dem.configs)

Sink_area = []
a = 1

for row in range(sink_area.configs.rows):
    for col in range(sink_area.configs.columns):
        area = dem[row, col]
        area_sink = depth[row, col]
        if area != dem.configs.nodata and area_sink == depth.configs.nodata:
            sink_area[row, col] = 0.0
        elif area == dem.configs.nodata:
            sink_area[row, col] = dem.configs.nodata
        elif area_sink != depth.configs.nodata:
            sink_area[row, col] = depth[row, col]
            Sink_area.append(a)

wbe.write_raster(sink_area, 'DEM_depth_sink.tif', compress=True)

# visualization
path_01 = '../00_data_source/DEM_depth_sink.tif'
data_01 = rs.open(path_01)

dem_array = data_01.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(24, 16))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_01, ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Sink area')

plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()