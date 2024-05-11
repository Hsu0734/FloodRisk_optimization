"""
Multi-objective optimization: topographic modification optimization
Author: Hanwen Xu
Version: 1
Date: May 11, 2024
"""
import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd

wbe = wbw.WbEnvironment()
wbe.verbose = False

wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source'
dem = wbe.read_raster('DEM2m.tif')

# creat a blank raster image of same size as the dem
cut_and_fill = wbe.new_raster(dem.configs)

# number of valid grid
n_grid = 0

for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            cut_and_fill[row, col] = dem.configs.nodata
        elif dem[row, col] != dem.configs.nodata:
            cut_and_fill[row, col] = 0.0
            n_grid = n_grid + 1
print(n_grid)


# threshold of flow path
flow_accum = wbe.d8_flow_accum(dem, out_type='cells')
Flow_accum_value = []
for row in range(flow_accum.configs.rows):
    for col in range(flow_accum.configs.columns):
        accum = flow_accum[row, col]  # Read a cell value from a Raster
        if accum != flow_accum.configs.nodata:
            Flow_accum_value.append(accum)
threshold = max(Flow_accum_value) * 0.01
print(threshold)
'''
# ------------------------------------------ #
# define MOO problem
class MyProblem(ElementwiseProblem):

    def __init__(self, n_grid, **kwargs):
        super().__init__(n_var=int(n_grid),
                         n_obj=3,
                         n_ieq_constr=0,
                         n_eq_constr=0,
                         xl=np.array([0] * n_grid),
                         xu=np.array([1] * n_grid),
                         **kwargs)
        self.n_grid = n_grid

    def _evaluate(self, x, out, *args, **kwargs):

        var_list = [float(value) for value in x]

        # notice your function should be Min function
        # resolution area: 4m^2  depth: 0.5m
        earth_volume_function = sum(abs(i) for i in var_list) * 4
        flow_length_function, sink_function = path_sum_calculation(var_list)

        # notice your function should be <= 0
        # g1 = var_list.count(0) - 2324
        # g1 = sum(abs(i) for i in var_list) - 592
        #g2 = (n_BRC * 0.9) - sum(abs(i) for i in var_list)

        out["F"] = [earth_volume_function, flow_length_function, sink_function]
        #out["G"] = [g1]
        #out["H"] = [g1]

def path_sum_calculation(var_list):
    i = 0
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                cut_and_fill[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                cut_and_fill[row, col] = var_list[i]
                i = i + 1

    # creat dem_pop
    dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and _fill'", input_rasters=[dem, cut_and_fill])

    # path length calculation
    flow_accum = wbe.d8_flow_accum(dem_pop, out_type='cells')
    slope = wbe.slope(dem_pop, units="percent")
    sink = wbe.sink(dem_pop)

    path_length = wbe.new_raster(flow_accum.configs)
    velocity = wbe.new_raster(flow_accum.configs)
    sink_area = wbe.new_raster(flow_accum.configs)

    for row in range(flow_accum.configs.rows):
        for col in range(flow_accum.configs.columns):
            elev = flow_accum[row, col]
            velo = flow_accum[row, col]
            sarea = sink[row, col]
            if elev >= threshold and elev != flow_accum.configs.nodata:
                path_length[row, col] = 1.0
            elif elev < threshold or elev == flow_accum.configs.nodata:
                path_length[row, col] = 0.0

            if velo == flow_accum.configs.nodata:
                velocity[row, col] = slope.configs.nodata
            elif velo != flow_accum.configs.nodata:
                slope_factor = (slope[row, col] / 100) ** 0.5
                flow_factor = (flow_accum[row, col] * 4 * 0.00001042) ** (2 / 3)
                velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6

            if sarea == flow_accum.configs.nodata:
                sink_area[row, col] = flow_accum.configs.nodata
            elif sarea == sink.configs.nodata:
                sink_area[row, col] = 0.0
            elif sarea != sink.configs.nodata:
                sink_area[row, col] = 1.0

    # 找到path length和max velocity
    path = []
    for row in range(path_length.configs.rows):
        for col in range(path_length.configs.columns):
            path.append(path_length[row, col])

    velocity_value = []
    for row in range(velocity.configs.rows):
        for col in range(velocity.configs.columns):
            velocity_value.append(velocity[row, col])

    sink_value =[]
    for row in range(sink_area.configs.rows):
        for col in range(sink_area.configs.columns):
            sink_value.append(sink_area[row, col])

    path_sum = -sum(path)
    max_velocity = max(velocity_value)
    sink_sum = -sum(sink_value)
    return path_sum, sink_sum

problem = MyProblem(n_grid)


# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.knapsack import create_random_knapsack_problem

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True)

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=50,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),  # 适合二元变量的交叉操作
    mutation=BitflipMutation(prob=0.1),
    eliminate_duplicates=True)


termination = get_termination("n_gen", 15)

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
plot_figure_path = 'scatter_plot_S4.png'
plot.save(plot_figure_path)

# 2D Pairwise Scatter Plots
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=20, facecolors='none', edgecolors='blue')
plt.title("Flow path length (y) and total cost (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 1], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Max velocity (y) and flow path length (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 0], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Max velocity (y) and total cost (x)")
plt.grid()
plt.show()


# save the data
result_df = pd.DataFrame(F)
result_df.to_csv('output_solution_S4.csv', index=False)
# 将True/False转换为1/0
X_int = X.astype(int)  # 转换成0/1
result_df = pd.DataFrame(X_int)
result_df.to_csv('output_variable_S4.csv', index=False)


### Decision making ###
### Min Decision ###
min_earth_volume = np.argmin(F[:, 0])
min_flow_length = np.argmin(F[:, 1])
min_velocity = np.argmin(F[:, 2])

min_earth_volume_solution = res.X[min_earth_volume]
min_flow_length_solution = res.X[min_flow_length]
min_velocity_solution = res.X[min_velocity]

min_earth_volume_dem = wbe.new_raster(dem.configs)
min_flow_length_dem = wbe.new_raster(dem.configs)
min_velocity_dem = wbe.new_raster(dem.configs)

wbe.working_directory = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\04_result'
t = 0
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            min_earth_volume_dem[row, col] = dem.configs.nodata
            min_flow_length_dem[row, col] = dem.configs.nodata
            min_velocity_dem[row, col] = dem.configs.nodata

        elif dem[row, col] != dem.configs.nodata:
            min_earth_volume_dem[row, col] = min_earth_volume_solution[t]
            min_flow_length_dem[row, col] = min_flow_length_solution[t]
            min_velocity_dem[row, col] = min_velocity_solution[t]
            t = t + 1

wbe.write_raster(min_earth_volume_dem, file_name='min_earth_volume_solution', compress=True)
wbe.write_raster(min_flow_length_dem, file_name='min_flow_length_solution', compress=True)
wbe.write_raster(min_velocity_dem, file_name='min_velocity_solution', compress=True)

after_dem_minEV = dem - min_earth_volume_dem
after_dem_minFL = dem - min_flow_length_dem
after_dem_minV = dem - min_velocity_dem

wbe.write_raster(after_dem_minEV, file_name='min_earth_volume_dem', compress=True)
wbe.write_raster(after_dem_minFL, file_name='min_flow_length_dem', compress=True)
wbe.write_raster(after_dem_minV, file_name='min_velocity_dem', compress=True)

### balance Decision ###
from pymoo.decomposition.asf import ASF

weights = np.array([0.333, 0.333, 0.333])
approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
decomp = ASF()
k = decomp.do(nF, 1/weights).argmin()
print("Best regarding ASF: Point \nk = %s\nF = %s" % (k, F[k]))

plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.add(F[k], s=50, color="red")
plot.show()

balance_solution = res.X[k]
balance_dem = wbe.new_raster(dem.configs)
q = 0
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            balance_dem[row, col] = dem.configs.nodata
        elif dem[row, col] != dem.configs.nodata:
            balance_dem[row, col] = balance_solution[q]
            q = q + 1

wbe.write_raster(balance_dem, file_name='balance_solution', compress=True)
after_dem_balance = dem - balance_dem
wbe.write_raster(after_dem_balance, file_name='balance_dem', compress=True)


# visualization of solution set
for i in range(20):
    solution = res.X[10 * i] # 每隔十个取一个解
    solution_dem = wbe.new_raster(dem.configs)

    p = 0
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                solution_dem[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                solution_dem[row, col] = solution[p]
                p = p + 1

    after_dem = dem - solution_dem
    filename = f'DEM_after_{10 * i}.tif'    #地形改动之后的结果
    wbe.write_raster(after_dem, file_name=filename, compress=True)

    filename_X = f'DEM_solution_{10 * i}.tif'   #地形自身的改动量
    wbe.write_raster(solution_dem, file_name=filename_X, compress=True)'''
