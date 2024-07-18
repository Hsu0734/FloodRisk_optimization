"""
Multi-objective optimization: Sink volume & retention volume & earthwork
Author: Hanwen Xu
Version: 1
Date: July 16, 2024
"""

import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd


wbe = wbw.WbEnvironment()
wbe.verbose = False

wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
            r'\FloodRisk_optimization\00_data_source'
mask = wbe.read_raster('Hanwen_mask_2.tif')
dem = wbe.read_raster('Hanwen_5m.tif')

# creat a blank raster image of same size as the dem
layer = wbe.new_raster(mask.configs)

# number of valid grid
grid = []
q = 1
for row in range(mask.configs.rows):
    for col in range(mask.configs.columns):
        if dem[row, col] == mask.configs.nodata:
            layer[row, col] = mask.configs.nodata
        elif dem[row, col] != mask.configs.nodata and mask[row, col] == mask.configs.nodata:
            layer[row, col] = 0.0
        elif dem[row, col] != mask.configs.nodata:
            layer[row, col] = 0.0
            grid.append(q)
n_grid = sum(grid)
print(n_grid)


# ------------------------------------------ #
# define MOO problem
class MyProblem(ElementwiseProblem):

    def __init__(self, n_grid, **kwargs):
        super().__init__(n_var=int(n_grid),
                         n_obj=3,
                         n_ieq_constr=0,
                         n_eq_constr=0,
                         xl=np.array([0] * n_grid),
                         xu=np.array([0.5] * n_grid),
                         **kwargs)
        self.n_grid = n_grid

    def _evaluate(self, x, out, *args, **kwargs):
        #var_list = [float(value) for value in x]

        earth_volume_function = sum(abs(i) for i in x) * 25
        sink_volume_function, sink_area_function = sink_sum_calculation(x)

        out["F"] = [earth_volume_function, sink_volume_function, sink_area_function]

def sink_sum_calculation(var_list):
    i = 0
    cut_and_fill = wbe.new_raster(dem.configs)
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                cut_and_fill[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata and mask[row, col] == mask.configs.nodata:
                cut_and_fill[row, col] = 0.0
            elif dem[row, col] != dem.configs.nodata and mask[row, col] != mask.configs.nodata:
                cut_and_fill[row, col] = var_list[i]
                i = i + 1

    # creat dem_pop
    # dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and_fill'", input_rasters=[dem, cut_and_fill])
    dem_pop = dem - cut_and_fill
    wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                            r'\FloodRisk_optimization\04_iteration_file'

    output_filename = f'DEM_iteration.tif'
    wbe.write_raster(dem_pop, output_filename, compress=True)
    dem_pop = wbe.read_raster(output_filename)

    # sink volume calculation
    fill_dem = wbe.fill_depressions(dem_pop)
    sink_area = fill_dem - dem_pop

    wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                            r'\FloodRisk_optimization\04_iteration_file'

    output_filename_2 = f'DEM_iteration_sink.tif'
    wbe.write_raster(sink_area, output_filename_2, compress=True)
    sink_area = wbe.read_raster(output_filename_2)

    retention_area = wbe.new_raster(dem_pop.configs)

    for row in range(sink_area.configs.rows):
        for col in range(sink_area.configs.columns):
            sink_volume = sink_area[row, col]
            if sink_volume == dem.configs.nodata:
                retention_area[row, col] = dem.configs.nodata
            elif sink_volume >= 0.05:
                retention_area[row, col] = sink_volume
            elif sink_volume <= 0.05 and sink_volume != dem.configs.nodata:
                retention_area[row, col] = 0.0

    Retention_volume = []
    for row in range(retention_area.configs.rows):
        for col in range(retention_area.configs.columns):
            sink_volume = retention_area[row, col]
            if sink_volume != retention_area.configs.nodata:
                volume = retention_area[row, col] * 25  # resolution = 5m
                Retention_volume.append(volume)

    # sink area calculation
    fill_depression = wbe.fill_depressions(dem_pop, max_depth=0.05)
    sink = wbe.sink(fill_depression, zero_background=False)
    Sink_value = []
    a = 0.0
    b = 1.0

    for row in range(sink.configs.rows):
        for col in range(sink.configs.columns):
            num_sink = sink[row, col]
            if num_sink == sink.configs.nodata:
                Sink_value.append(a)
            else:
                Sink_value.append(b)

    sink_volume_sum = -sum(Retention_volume)
    sink_area_sum = -sum(Sink_value)
    return sink_volume_sum, sink_area_sum

problem = MyProblem(n_grid)


# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

algorithm = NSGA2(
    pop_size=50,
    n_offsprings=20,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.8, eta=15),
    mutation=PM(eta=15),
    eliminate_duplicates=True)


termination = get_termination("n_gen", 50)

from pymoo.optimize import minimize
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F

# Visualization of Objective space or Variable space
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# 3D Visualization
plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.show()

#output_filename = f'DEM_sink_S3_1_{i}.tif'

plot_figure_path = 'scatter_plot_DEM5m.png'
plot.save(plot_figure_path)

# 2D Pairwise Scatter Plots

# save the data
result_df = pd.DataFrame(F)
result_df.to_csv('output_solution_DEM5m.csv', index=False)
result_df = pd.DataFrame(X)
result_df.to_csv('output_variable_DEM5m.csv', index=False)


### Decision making ###
### Min Decision ###
min_earth_volume = np.argmin(F[:, 0])
min_sink_volume = np.argmin(F[:, 1])
min_sink_area = np.argmin(F[:, 2])

min_earth_volume_solution = res.X[min_earth_volume]
min_sink_volume_solution = res.X[min_sink_volume]
min_sink_area_solution = res.X[min_sink_area]

min_earth_volume_dem = wbe.new_raster(dem.configs)
min_sink_volume_dem = wbe.new_raster(dem.configs)
min_sink_area_dem = wbe.new_raster(dem.configs)

wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
            r'\FloodRisk_optimization\00_data_source\result'
t = 0
for row in range(mask.configs.rows):
    for col in range(mask.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            min_earth_volume_dem[row, col] = mask.configs.nodata
            min_sink_volume_dem[row, col] = mask.configs.nodata
            min_sink_area_dem[row, col] = mask.configs.nodata

        elif dem[row, col] != dem.configs.nodata and mask[row, col] == mask.configs.nodata:
            min_earth_volume_dem[row, col] = 0.0
            min_sink_volume_dem[row, col] = 0.0
            min_sink_area_dem[row, col] = 0.0

        elif dem[row, col] != dem.configs.nodata:
            min_earth_volume_dem[row, col] = min_earth_volume_solution[t]
            min_sink_volume_dem[row, col] = min_sink_volume_solution[t]
            min_sink_area_dem[row, col] = min_sink_area_solution[t]
            t = t + 1

wbe.write_raster(min_earth_volume_dem, file_name='min_earth_volume_solution', compress=True)
wbe.write_raster(min_sink_volume_dem, file_name='min_sink_volume_solution', compress=True)
wbe.write_raster(min_sink_area_dem, file_name='min_sink_area_solution', compress=True)


after_dem_minEV = dem - min_earth_volume_dem
after_dem_minSV = dem - min_sink_volume_dem
after_dem_minSA = dem - min_sink_area_dem

wbe.write_raster(after_dem_minEV, file_name='min_earth_volume_dem', compress=True)
wbe.write_raster(after_dem_minSV, file_name='min_sink_volume_dem', compress=True)
wbe.write_raster(after_dem_minSA, file_name='min_sink_area_dem', compress=True)
