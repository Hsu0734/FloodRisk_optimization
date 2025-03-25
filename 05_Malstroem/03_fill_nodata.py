import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

# 输入文件
input_dem = "input_dem.tif"
output_dem = "filled_dem.tif"

with rasterio.open(input_dem) as src:
    dem = src.read(1)
    nodata = src.nodata
    mask = (dem != nodata)  # 有效数据区域（True=有效）

    # 标记边缘Nodata（通过形态学腐蚀找到内部破洞）
    from scipy.ndimage import binary_erosion
    inner_mask = binary_erosion(mask, iterations=3)  # 迭代次数控制边缘宽度
    holes_mask = mask & ~inner_mask  # 仅内部破洞（被有效数据包围的Nodata）

    # 仅填充内部破洞
    filled = np.where(holes_mask, np.nan, dem)  # 将边缘Nodata设为NaN，避免被填充
    filled = rasterio.fill.fillnodata(filled, mask=holes_mask, max_search_distance=50)

    # 恢复边缘Nodata
    final_dem = np.where(mask, filled, nodata)  # 保留原始边缘Nodata

    # 保存结果
    with rasterio.open(output_dem, 'w', **src.profile) as dst:
        dst.write(final_dem, 1)

print("内部破洞填充完成，边缘Nodata保留！")