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
                        r'\FloodRisk_optimization\00_data_source'


# web read DEM data
dem = wbe.read_raster('Hanwen_2m.tif')
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
path_01 = '../00_data_source/DEM_demo_sink_dem.tif'
data_01 = rs.open(path_01)

fig, ax = plt.subplots(figsize=(32, 32))
ax.tick_params(axis='both', which='major', labelsize=40)

vmin = 0
vmax = 3
show(data_01, title=f'DEM_volume_5m_average', ax=ax, vmin=vmin, vmax=vmax)
#show(data_01, title=f'DEM_volume_5m_average', ax=ax)
plt.ticklabel_format(style='plain')
# ax.grid(True, linestyle='--', color='grey')

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.19, 0.03, 0.3])  # 调整颜色条的位置和大小
cbar = plt.colorbar(ax.images[0], cax=cbar_ax)  # 使用 ax.images[0] 获取图像数据用于颜色条

# 调整颜色条上刻度标签的字体大小
cbar.ax.tick_params(labelsize=40)

plt.show()


Retention_volume = []
for row in range(retention_area.configs.rows):
    for col in range(retention_area.configs.columns):
        sink_volume = retention_area[row, col]
        if sink_volume != retention_area.configs.nodata:
            volume = retention_area[row, col] * 4   # resolution = 5m
            Retention_volume.append(volume)

Total_volume = sum(Retention_volume)
print(Total_volume)
