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
dem = wbe.read_raster('Hanwen_mask.tif')
dem5m = wbe.read_raster('Hanwen_5m.tif')

# 循环处理CSV文件中的0到99行
for n in range(50):

    row = df.iloc[int(n)]
    row_list = row.tolist()

    layer = wbe.new_raster(dem5m.configs)
    m = 0
    for row in range(dem5m.configs.rows):
        for col in range(dem5m.configs.columns):
            if dem5m[row, col] == dem.configs.nodata:
                layer[row, col] = dem.configs.nodata
            elif dem5m[row, col] != dem.configs.nodata and dem[row, col] == dem.configs.nodata:
                layer[row, col] = 0.0
            elif dem5m[row, col] != dem.configs.nodata:
                layer[row, col] = row_list[m]
                m += 1

    #dem_solution = dem5m - layer
    layer = dem5m - layer

    fill_dem = wbe.fill_depressions(layer)
    sink_area = fill_dem - layer

    retention_area = wbe.new_raster(dem5m.configs)

    for row in range(sink_area.configs.rows):
        for col in range(sink_area.configs.columns):
            sink_volume = sink_area[row, col]
            if sink_volume == dem5m.configs.nodata:
                retention_area[row, col] = dem.configs.nodata
            elif sink_volume >= 0.05:
                retention_area[row, col] = sink_volume
            elif sink_volume <= 0.05 and sink_volume != dem.configs.nodata:
                retention_area[row, col] = 0.0

    wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\03_visualization'
    output_filename = f'DEM_sink_volume_DEM10m_{n}.tif'
    wbe.write_raster(retention_area, output_filename, compress=True)

    # visualization
    path_01 = f'../03_visualization/{output_filename}'
    data_01 = rs.open(path_01)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.tick_params(axis='both', which='major', labelsize=20)
    show(data_01, title=f'DEM_sink_{n}', ax=ax)
    plt.ticklabel_format(style='plain')
    plt.show()
    plt.close(fig)  # 关闭图形以释放内存