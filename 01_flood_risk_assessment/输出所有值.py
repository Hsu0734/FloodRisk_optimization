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
dem = wbe.read_raster('Hanwen_mask.tif')

for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        elev = dem[row, col]
        print(elev)