import whitebox_workflows as wbw
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt

wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\03_visualization'

# 初始化一个用于存储累加结果的变量，初始值设置为None
accumulated_raster = None

# 循环读取每个文件，并累加
for n in range(50):
    filename = f'DEM5m_sink_area_{n}.tif'
    current_raster = wbe.read_raster(filename)
    if accumulated_raster is None:
        accumulated_raster = current_raster
    else:
        accumulated_raster += current_raster

# 计算平均值
average_raster = accumulated_raster / 50

# 创建一个新的栅格文件用于保存平均后的结果
wbe.write_raster(average_raster, 'DEM5m_sink_area_average.tif', compress=True)

path_01 = f'../03_visualization/DEM5m_sink_area_average.tif'
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