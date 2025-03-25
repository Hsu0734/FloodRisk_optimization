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
dem = wbe.read_raster('DEM_demo_resample_5m.tif')
dem5m = wbe.read_raster('DEM_demo_resample_5m.tif')

# 循环处理CSV文件中的0到99行
for n in range(100):

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
    dem_solution = wbe.raster_calculator(expression="'dem5m' - 'layer'", input_rasters=[dem5m, layer])

    flow_accum = wbe.d8_flow_accum(dem_solution, out_type='cells')
    path_length = wbe.new_raster(flow_accum.configs)
    for row in range(flow_accum.configs.rows):
        for col in range(flow_accum.configs.columns):
            elev = flow_accum[row, col]  # Read a cell value from a Raster
            if elev >= 29.12 and elev != flow_accum.configs.nodata:
                path_length[row, col] = 1.0
            elif elev == flow_accum.configs.nodata:
                path_length[row, col] = flow_accum.configs.nodata
            elif elev < 29.12:
                path_length[row, col] = 0.0


    wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\03_visualization'
    output_filename = f'DEM5m_path_length_{n}.tif'
    wbe.write_raster(path_length, output_filename, compress=True)

    # visualization
    path_01 = f'../03_visualization/{output_filename}'
    data_01 = rs.open(path_01)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.tick_params(axis='both', which='major', labelsize=20)
    show(data_01, title=f'DEM_path_{n}', ax=ax)
    plt.ticklabel_format(style='plain')
    plt.show()
    plt.close(fig)  # 关闭图形以释放内存