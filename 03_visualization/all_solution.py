import pandas as pd
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt
import whitebox_workflows as wbw


wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\03_visualization'

# 循环处理CSV文件中的0到99行
for n in range(100):

    dem = wbe.read_raster(f'DEM_solution_DEM5m_{n}.tif')
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

    output_filename = f'DEM_volume_{n}.tif'
    wbe.write_raster(retention_area, output_filename, compress=True)

    # visualization
    path_01 = f'../03_visualization/{output_filename}'
    data_01 = rs.open(path_01)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.tick_params(axis='both', which='major', labelsize=20)
    show(data_01, title=f'DEM_sink_{n}', ax=ax)
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