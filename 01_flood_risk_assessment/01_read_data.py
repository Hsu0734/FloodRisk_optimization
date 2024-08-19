"""
Read DEM data
Author: Hanwen Xu
Date: Oct 01, 2023
"""

import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20
# read and print the DEM data
path_00 = '../00_data_source/Hanwen_5m_mask.tif'
data_00 = rs.open(path_00)

# creat a plot
fig, ax = plt.subplots(figsize=(30, 24))
show(data_00, title='DEM_demo_2m', ax=ax)
plt.ticklabel_format(style='plain')

# grid and show plot
ax.grid(True, linestyle='-', color='grey')

plt.show()