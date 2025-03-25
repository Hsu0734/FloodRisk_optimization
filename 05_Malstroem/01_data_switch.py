import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# 输入和输出 DEM 文件路径
input_dem = "changsha_zcq_2.tif"  # 替换为你的原始 DEM 文件路径
output_dem = "changsha_zcq_bilinear_float32.tif"  # 输出 DEM 路径（包含重采样）

# NoData 值设置
new_nodata = np.nan  # 可替换为 np.nan 如果需要

# 目标 CRS (CGCS2000 / 3-degree Gauss-Kruger CM 111E - EPSG:4534)
# target_crs = 'EPSG:4546'
target_crs = 'EPSG:4546'

# 目标分辨率（30x30米）
target_resolution = 30

# 读取原始 DEM
with rasterio.open(input_dem) as src:
    dem_data = src.read(1).astype(np.float32)  # 转换为 float32
    original_crs = src.crs
    original_transform = src.transform
    original_nodata = src.nodata if src.nodata is not None else 0  # 如果没有 NoData 值，默认 0

    # 替换原始 NoData 值
    dem_data[dem_data == original_nodata] = new_nodata

    # 计算重投影和重采样的仿射变换和尺寸
    transform, width, height = calculate_default_transform(
        original_crs, target_crs, src.width, src.height, *src.bounds,
        resolution=(target_resolution, target_resolution)  # 指定分辨率为30x30m
    )

    # 更新元数据
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'dtype': 'float32',
        'nodata': new_nodata
    })

    # 创建输出 DEM 并进行重投影 + 重采样
    with rasterio.open(output_dem, 'w', **kwargs) as dst:
        reproject(
            source=dem_data,
            destination=rasterio.band(dst, 1),
            src_transform=original_transform,
            src_crs=original_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

print(f"DEM 预处理完成，输出文件保存在: {output_dem}")
