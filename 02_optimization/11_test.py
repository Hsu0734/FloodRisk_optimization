"""
Multi-objective optimization
Author: Hanwen Xu
Version: 1
Date: Jan 6, 2025
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
### Setting raster of the mask as variable grid cells by using mask layer.

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
                         n_obj=2,
                         n_ieq_constr=0,
                         n_eq_constr=0,
                         xl=np.array([0] * n_grid),
                         xu=np.array([1.5] * n_grid),
                         **kwargs)
        self.n_grid = n_grid

    def _evaluate(self, x, out, *args, **kwargs):
        var_list = []
        for i in range(n_grid):
            var_list.append(x[i])
        #var_list = [float(value) for value in x]

        earth_volume_function = sum(abs(i) for i in var_list) * 25
        sink_area_function = sink_sum_calculation(var_list)

        out["F"] = [earth_volume_function, sink_area_function]

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
                cut_and_fill[row, col] = round(var_list[i], 3)
                i = i + 1

    # creat dem_pop
    # dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and_fill'", input_rasters=[dem, cut_and_fill])
    dem_pop = wbe.new_raster(dem.configs)
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                dem_pop[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                dem_pop[row, col] = dem[row, col] - cut_and_fill[row, col]

    # sink area calculation
    depth_in_sink = wbe.depth_in_sink(dem_pop, zero_background=False)
    sink = wbe.new_raster(depth_in_sink.configs)
    for row in range(depth_in_sink.configs.rows):
        for col in range(depth_in_sink.configs.columns):
            if depth_in_sink[row, col] == depth_in_sink.configs.nodata:
                sink[row, col] = depth_in_sink.configs.nodata
            elif depth_in_sink[row, col] >= 0.05:
                sink[row, col] = depth_in_sink[row, col]
            elif depth_in_sink[row, col] <= 0.05 and depth_in_sink[row, col] != dem.configs.nodata:
                sink[row, col] = depth_in_sink.configs.nodata

    Sink_value = []
    a = 0.0
    b = 1.0

    for row in range(sink.configs.rows):
        for col in range(sink.configs.columns):
            if sink[row, col] == sink.configs.nodata:
                Sink_value.append(a)
            else:
                Sink_value.append(b)

    path_sum = -sum(Sink_value)

    return path_sum

problem = MyProblem(n_grid)


# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination


class RoundedSBX(SBX):
    def __init__(self, prob, eta, decimals=3):
        super().__init__(prob=prob, eta=eta)
        self.decimals = decimals

    def _do(self, problem, X, **kwargs):
        # 调用父类方法进行交叉操作
        X = super()._do(problem, X, **kwargs)
        # 保留指定小数位数
        X = np.round(X, self.decimals)
        return X


class RoundedPM(PM):
    def __init__(self, eta, decimals=3):
        super().__init__(eta=eta)
        self.decimals = decimals

    def _do(self, problem, X, **kwargs):
        # 调用父类方法进行变异操作
        X = super()._do(problem, X, **kwargs)
        # 保留指定小数位数
        X = np.round(X, self.decimals)
        return X


class RoundedFloatRandomSampling(FloatRandomSampling):
    def __init__(self, decimals=3):
        super().__init__()
        self.decimals = decimals

    def _do(self, problem, n_samples, **kwargs):
        # 调用父类方法生成随机数
        X = super()._do(problem, n_samples, **kwargs)
        # 使用 np.round 保留小数点后指定的位数
        X = np.round(X, self.decimals)
        return X

# 使用自定义采样器，并指定保留3位小数

algorithm = NSGA2(
    pop_size=80,
    n_offsprings=40,
    sampling=RoundedFloatRandomSampling(decimals=3),
    crossover=RoundedSBX(prob=0.8, eta=15, decimals=3),
    mutation=RoundedPM(eta=15, decimals=2),
    eliminate_duplicates=True)


termination = get_termination("n_gen", 80)

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