"""
DEM revision
Author: Hanwen Xu
Date: Jun 029, 2024
"""

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
dem = wbe.read_raster('Hanwen_2m_buffer.tif')
mask = wbe.new_raster(dem.configs)

for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        elev = dem[row, col]
        if elev == dem.configs.nodata:
            mask[row, col] = dem.configs.nodata
        elif elev != dem.configs.nodata:
            mask[row, col] = 1.0

wbe.write_raster(mask, 'Hanwen_2m_mask.tif', compress=True)