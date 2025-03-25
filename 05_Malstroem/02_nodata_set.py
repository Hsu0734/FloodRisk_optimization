import rasterio
import numpy as np

# 输入和输出文件路径
input_file = 'Kaifu.tif'
output_file = 'output_dem_float32.tif'

# 打开输入文件
with rasterio.open(input_file) as src:
    # 读取数据
    data = src.read(1)

    # 获取原始nodata值
    nodata = src.nodata

    # 将int16转换为float32，同时保留nodata值
    data_float32 = data.astype(np.float32)

    # 如果原始数据有nodata值，将其保留
    if nodata is not None:
        data_float32[data == nodata] = nodata  # 将原始nodata位置的值设置为nodata

    # 获取元数据
    meta = src.meta

    # 更新元数据中的数据类型和nodata值
    meta.update(dtype=rasterio.float32, nodata=nodata)

    # 写入输出文件
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(data_float32, 1)