"""
Slope analysis
Author: Hanwen Xu
Date: Jun 029, 2024
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
                        r'\FloodRisk_optimization\00_data_source\result'


# web read DEM data
dem = wbe.read_raster('min_sink_volume_dem.tif')
fill_dem = wbe.fill_depressions(dem)
sink_area = fill_dem - dem

retention_area = wbe.new_raster(dem.configs)

for row in range(sink_area.configs.rows):
    for col in range(sink_area.configs.columns):
        sink_volume = sink_area[row, col]
        if sink_volume == dem.configs.nodata:
            retention_area[row, col] = dem.configs.nodata
        elif sink_volume >= 0.05:
            retention_area[row, col] = sink_volume
        elif sink_volume <= 0.05 and sink_volume != dem.configs.nodata:
            retention_area[row, col] = 0.0

wbe.write_raster(retention_area, 'DEM_demo_sink_dem.tif', compress=True)

# visualization
path_01 = '../00_data_source/result/DEM_demo_sink_dem.tif'
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
cbar.set_label('Sink area')

plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()


Retention_volume = []
for row in range(retention_area.configs.rows):
    for col in range(retention_area.configs.columns):
        sink_volume = retention_area[row, col]
        if sink_volume != retention_area.configs.nodata:
            volume = retention_area[row, col] * 25    # resolution = 5m
            Retention_volume.append(volume)

Total_volume = sum(Retention_volume)
print(Total_volume)
