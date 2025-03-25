import pandas as pd
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt
import whitebox_workflows as wbw

# 读取CSV文件
df = pd.read_csv('output_variable_DEM5m.csv')

wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source'
dem = wbe.read_raster('Hanwen_mask_2m.tif')
dem5m = wbe.read_raster('Hanwen_2m.tif')

# 循环处理CSV文件中的0到99行
for n in range(100):

    row_v = df.iloc[int(n)]
    row_list = row_v.tolist()

    layer = wbe.new_raster(dem5m.configs)
    m = 0
    for row in range(dem5m.configs.rows):
        for col in range(dem5m.configs.columns):
            if dem5m[row, col] == dem.configs.nodata:
                layer[row, col] = dem5m.configs.nodata
            elif dem5m[row, col] != dem.configs.nodata and dem[row, col] == dem.configs.nodata:
                layer[row, col] = 0.0
            elif dem5m[row, col] != dem.configs.nodata:
                layer[row, col] = row_list[m]
                m += 1

    layer_after = dem5m - layer

    wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\03_visualization'

    output_filename = f'DEM5m_after_{n}.tif'
    wbe.write_raster(layer_after, output_filename, compress=True)

    # visualization
    '''path_01 = f'../03_visualization/{output_filename}'
    data_01 = rs.open(path_01)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.tick_params(axis='both', which='major', labelsize=20)
    show(data_01, title=f'DEM_sink_volume_{n}', ax=ax)
    plt.ticklabel_format(style='plain')
    plt.show()'''


    layer_after = wbe.read_raster(output_filename)

    fill_dem = wbe.fill_depressions(layer_after)
    sink_area = fill_dem - layer_after

    retention_area = wbe.new_raster(dem5m.configs)

    for row in range(dem5m.configs.rows):
        for col in range(dem5m.configs.columns):
            sink_volume = sink_area[row, col]
            if sink_volume == dem5m.configs.nodata:
                retention_area[row, col] = dem.configs.nodata
            elif sink_volume >= 0.05:
                retention_area[row, col] = sink_volume
            elif sink_volume <= 0.05 and sink_volume != dem5m.configs.nodata:
                retention_area[row, col] = 0.0

    wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                            r'\FloodRisk_optimization\03_visualization'

    output_filename_02 = f'DEM5m_sink_volume_{n}.tif'
    wbe.write_raster(retention_area, output_filename_02, compress=True)

    # visualization
    path_01 = f'../03_visualization/{output_filename_02}'
    data_01 = rs.open(path_01)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.tick_params(axis='both', which='major', labelsize=20)

    vmin = 0
    vmax = 3

    show(data_01, title=f'DEM_sink_volume_{n}', ax=ax, vmin=vmin, vmax=vmax)
    plt.ticklabel_format(style='plain')
    plt.show()



    Retention_volume = []
    for row in range(retention_area.configs.rows):
        for col in range(retention_area.configs.columns):
            sink_volume = retention_area[row, col]
            if sink_volume != retention_area.configs.nodata:
                volume = retention_area[row, col] * 25  # resolution = 5m
                Retention_volume.append(volume)

    Total_volume = sum(Retention_volume)
    print(Total_volume)