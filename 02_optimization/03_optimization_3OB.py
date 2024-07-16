"""
Multi-objective optimization: Sink optimization & cost & velocity
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
dem = wbe.read_raster('Hanwen_10m.tif')

# creat a blank raster image of same size as the dem
layer = wbe.new_raster(dem.configs)

# number of valid grid
grid = []
q = 1
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            layer[row, col] = dem.configs.nodata
        elif dem[row, col] != dem.configs.nodata:
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

        # notice your function should be Min function
        earth_volume_function = sum(abs(i) for i in x) * 100
        velocity_function, sink_function = path_sum_calculation(x)

        # notice your function should be <= 0
        # g1 = sum(abs(i) for i in var_list) - 592

        out["F"] = [earth_volume_function, velocity_function, sink_function]
        #out["G"] = [g1]

def path_sum_calculation(var_list):
    i = 0
    cut_and_fill = wbe.new_raster(dem.configs)
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
    sink = wbe.sink(dem_pop, zero_background=False)
    velocity = wbe.new_raster(dem_pop.configs)

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

    for row in range(flow_accum.configs.rows):
        for col in range(flow_accum.configs.columns):
            velo = flow_accum[row, col]

            if velo == flow_accum.configs.nodata:
                velocity[row, col] = flow_accum.configs.nodata
            elif velo != flow_accum.configs.nodata:
                slope_factor = (slope[row, col] / 100) ** 0.5
                flow_factor = (flow_accum[row, col] * 100 * 0.000004215717) ** (2 / 3)
                velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6

    # 找到max velocity
    velocity_value = []
    for row in range(velocity.configs.rows):
        for col in range(velocity.configs.columns):
            velocity_value.append(velocity[row, col])

    # return 结果
    sink_sum = -sum(Sink_value)
    max_velocity = max(velocity_value)
    return max_velocity, sink_sum

problem = MyProblem(n_grid)


# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=40,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=20),
    mutation=PM(eta=20),
    eliminate_duplicates=True)


termination = get_termination("n_gen", 100)

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
plot_figure_path = 'scatter_plot_Hanwen_10m.png'
plot.save(plot_figure_path)

# 2D Pairwise Scatter Plots
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=20, facecolors='none', edgecolors='blue')
plt.title("Max velocity (y) and total cost (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 1], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Sink area (y) and Max velocity (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 0], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Sink area (y) and total cost (x)")
plt.grid()
plt.show()


# save the data
result_df = pd.DataFrame(F)
result_df.to_csv('output_solution_Hanwen_10m.csv', index=False)
result_df = pd.DataFrame(X)
result_df.to_csv('output_variable_Hanwen_10m.csv', index=False)


### Decision making ###
### Min Decision ###
min_earth_volume = np.argmin(F[:, 0])
min_velocity = np.argmin(F[:, 1])
min_sink = np.argmin(F[:, 2])

min_earth_volume_solution = res.X[min_earth_volume]
min_velocity_solution = res.X[min_velocity]
min_sink_solution = res.X[min_sink]

min_earth_volume_dem = wbe.new_raster(dem.configs)
min_velocity_dem = wbe.new_raster(dem.configs)
min_sink_dem = wbe.new_raster(dem.configs)

wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source\result'
t = 0
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            min_earth_volume_dem[row, col] = dem.configs.nodata
            min_velocity_dem[row, col] = dem.configs.nodata
            min_sink_dem[row, col] = dem.configs.nodata

        elif dem[row, col] != dem.configs.nodata:
            min_earth_volume_dem[row, col] = min_earth_volume_solution[t]
            min_velocity_dem[row, col] = min_velocity_solution[t]
            min_sink_dem[row, col] = min_sink_solution[t]
            t = t + 1

wbe.write_raster(min_earth_volume_dem, file_name='min_earth_volume_solution', compress=True)
wbe.write_raster(min_velocity_dem, file_name='min_velocity_solution', compress=True)
wbe.write_raster(min_sink_dem, file_name='min_sink_solution', compress=True)

after_dem_minEV = dem - min_earth_volume_dem
after_dem_minFL = dem - min_velocity_dem
after_dem_minV = dem - min_sink_dem

wbe.write_raster(after_dem_minEV, file_name='min_earth_volume_dem', compress=True)
wbe.write_raster(after_dem_minFL, file_name='min_veloctiy_dem', compress=True)
wbe.write_raster(after_dem_minV, file_name='min_sink_dem', compress=True)

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
