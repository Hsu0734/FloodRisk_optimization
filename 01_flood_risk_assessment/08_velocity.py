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
slope = wbe.read_raster('DEM_demo_slope.tif')
flow_accum = wbe.read_raster('DEM_demo_flow_accu.tif')
velocity = wbe.new_raster(slope.configs)

for row in range(slope.configs.rows):
    for col in range(slope.configs.columns):
        elev = slope[row, col]
        if elev == slope.configs.nodata:
            velocity[row, col] = slope.configs.nodata

        elif elev != slope.configs.nodata:
            #velocity[row, col] = (((flow_accum[row, col] * 100 * 0.000004215717) ** 0.4) * ((slope[row, col] / 100) ** 0.3))/((10 ** 0.4) * (0.03 ** 0.6))
            #velocity[row, col] = ((((slope[row, col] / 100) ** 0.5) * (flow_accum[row, col] * 100 * 0.000004215717 / 10) ** (2/3)) / 0.03) ** 0.6
            slope_factor = (slope[row, col] / 100) ** 0.5
            flow_factor = (flow_accum[row, col] * 4 * 0.00001042) ** (2 / 3)
            velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6

wbe.write_raster(velocity, 'DEM_demo_velocity.tif', compress=True)

# visualization
path_01 = '../00_data_source/DEM_demo_velocity.tif'
data_01 = rs.open(path_01)

dem_array = data_01.read(1, masked=True)  # 使用 masked=True 来自动处理 nodata 值
min_elevation = np.min(dem_array[~dem_array.mask])
max_elevation = np.max(dem_array[~dem_array.mask])

# Use imshow to display the DEM data with the correct color mapping
fig, ax = plt.subplots(figsize=(30, 24))
image = ax.imshow(dem_array, vmin=min_elevation, vmax=max_elevation)
show(data_01, cmap='Blues', ax=ax)

# Create a colorbar
cbar = plt.colorbar(image, ax=ax, orientation='vertical', cmap='Blues', shrink=0.5)
cbar.set_label('Velocity (meters/second)')

plt.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', color='grey')
plt.show()

# value
Velocity_value = []
for row in range(velocity.configs.rows):
    for col in range(velocity.configs.columns):
        elev = velocity[row, col]
        if elev != flow_accum.configs.nodata:
            Velocity_value.append(elev)

# print(Velocity_value)
print(max(Velocity_value))
print(min(Velocity_value))
# print(np.median(Velocity_value))