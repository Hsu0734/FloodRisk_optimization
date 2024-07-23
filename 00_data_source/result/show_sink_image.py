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
                        r'\FloodRisk_optimization\00_data_source\result'


# web read DEM data
dem = wbe.read_raster('min_sink_area_dem.tif')
#dem = wbe.read_raster('min_earth_volume_dem.tif')

# dem_00 = wbe.fill_missing_data(dem, exclude_edge_nodata= True)
fill_dem = wbe.fill_depressions(dem, max_depth=0.05)
sink = wbe.sink(fill_dem)
sink_area = wbe.new_raster(dem.configs)

Sink_value = []

for row in range(sink_area.configs.rows):
    for col in range(sink_area.configs.columns):
        area = dem[row, col]
        area_sink = sink[row, col]
        if area != dem.configs.nodata and area_sink == sink.configs.nodata:
            sink_area[row, col] = 0.0
        elif area == dem.configs.nodata:
            sink_area[row, col] = dem.configs.nodata
        elif area_sink != sink.configs.nodata:
            sink_area[row, col] = 1.0
            Sink_value.append(sink_area[row, col])

wbe.write_raster(sink_area, 'DEM_demo_sink.tif', compress=True)

# visualization
path_01 = '../result/DEM_demo_sink.tif'
data_01 = rs.open(path_01)

fig, ax = plt.subplots(figsize=(32, 32))
ax.tick_params(axis='both', which='major', labelsize=40)
show(data_01, title=f'DEM5m_sink_area_average.tif', ax=ax)
plt.ticklabel_format(style='plain')
#ax.grid(True, linestyle='--', color='grey')

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.19, 0.03, 0.3])  # 调整颜色条的位置和大小
cbar = plt.colorbar(ax.images[0], cax=cbar_ax)  # 使用 ax.images[0] 获取图像数据用于颜色条

# 调整颜色条上刻度标签的字体大小
cbar.ax.tick_params(labelsize=40)

plt.show()


# value

# print(Slope_value)
print(sum(Sink_value))