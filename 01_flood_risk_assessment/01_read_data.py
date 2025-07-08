"""
Read DEM data
Author: Hanwen Xu
Date: Oct 01, 2023
"""

import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20
# read and print the DEM data
path_00 = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\DEM5m_after_0.tif'
data_00 = rs.open(path_00)

# creat a plot
fig, ax = plt.subplots(figsize=(32, 32))
show(data_00, title='Greve_5m', ax=ax)
plt.ticklabel_format(style='plain')

# grid and show plot
#ax.grid(True, linestyle='-', color='grey')

plt.show()


#################################
path_01 = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\DEM5m_sink_volume_0.tif'
data_01 = rs.open(path_01)

fig, ax = plt.subplots(figsize=(32, 32))
ax.tick_params(axis='both', which='major', labelsize=40)

vmin = 0
vmax = 3


show(data_01, title=f'DEM_volume_5m_average', ax=ax, vmin=vmin, vmax=vmax)
plt.ticklabel_format(style='plain')
# ax.grid(True, linestyle='--', color='grey')

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.19, 0.03, 0.3])  # 调整颜色条的位置和大小
cbar = plt.colorbar(ax.images[0], cax=cbar_ax)  # 使用 ax.images[0] 获取图像数据用于颜色条

# 调整颜色条上刻度标签的字体大小
cbar.ax.tick_params(labelsize=40)

plt.show()