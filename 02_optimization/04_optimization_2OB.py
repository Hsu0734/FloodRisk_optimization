"""
Multi-objective optimization: Sink optimization & cost
Author: Hanwen Xu
Version: 1
Date: May 12, 2024
"""

import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd

for i in range(10):
    success = False  # 用于控制重试机制的标志
    while not success:
        try:
            print(i)

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
                                     n_obj=2,
                                     n_ieq_constr=0,
                                     n_eq_constr=0,
                                     xl=np.array([0] * n_grid),
                                     xu=np.array([0.5] * n_grid),
                                     **kwargs)
                    self.n_grid = n_grid

                def _evaluate(self, x, out, *args, **kwargs):
                    #var_list = [float(value) for value in x]

                    earth_volume_function = sum(abs(i) for i in x) * 4
                    sink_function = path_sum_calculation(x)

                    out["F"] = [earth_volume_function, sink_function]

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
                dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and_fill'", input_rasters=[dem, cut_and_fill])
                # dem_pop = dem - cut_and_fill

                # path length calculation
                sink = wbe.sink(dem_pop, zero_background=False)
                sink_area = wbe.new_raster(dem_pop.configs)

                for row in range(sink_area.configs.rows):
                    for col in range(sink_area.configs.columns):
                        num_sink = sink[row, col]
                        if num_sink == sink.configs.nodata:
                            sink_area[row, col] = 0.0
                        else:
                            sink_area[row, col] = 1.0

                Sink_value = []
                for row in range(sink_area.configs.rows):
                    for col in range(sink_area.configs.columns):
                        Sink_value.append(sink_area[row, col])

                sink_sum = -sum(Sink_value)
                return sink_sum

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
                crossover=SBX(prob=0.9, eta=20),
                mutation=PM(eta=20),
                eliminate_duplicates=True)


            termination = get_termination("n_gen", 200)

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

            plot_figure_path = 'scatter_plot_DEM10m.png'
            plot.save(plot_figure_path)

            # 2D Pairwise Scatter Plots

            # save the data
            result_df = pd.DataFrame(F)
            result_df.to_csv('output_solution_DEM10m.csv', index=False)
            result_df = pd.DataFrame(X)
            result_df.to_csv('output_variable_DEM10m.csv', index=False)


            ### Decision making ###
            ### Min Decision ###
            min_earth_volume = np.argmin(F[:, 0])
            min_sink = np.argmin(F[:, 1])

            min_earth_volume_solution = res.X[min_earth_volume]
            min_flow_length_solution = res.X[min_sink]

            min_earth_volume_dem = wbe.new_raster(dem.configs)
            min_flow_length_dem = wbe.new_raster(dem.configs)

            wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization' \
                        r'\FloodRisk_optimization\00_data_source\result'
            t = 0
            for row in range(dem.configs.rows):
                for col in range(dem.configs.columns):
                    if dem[row, col] == dem.configs.nodata:
                        min_earth_volume_dem[row, col] = dem.configs.nodata
                        min_flow_length_dem[row, col] = dem.configs.nodata

                    elif dem[row, col] != dem.configs.nodata:
                        min_earth_volume_dem[row, col] = min_earth_volume_solution[t]
                        min_flow_length_dem[row, col] = min_flow_length_solution[t]
                        t = t + 1

            wbe.write_raster(min_earth_volume_dem, file_name='min_earth_volume_solution', compress=True)
            wbe.write_raster(min_flow_length_dem, file_name='min_sink_area_solution', compress=True)


            after_dem_minEV = dem - min_earth_volume_dem
            after_dem_minFL = dem - min_flow_length_dem

            wbe.write_raster(after_dem_minEV, file_name='min_earth_volume_dem', compress=True)
            wbe.write_raster(after_dem_minFL, file_name='min_sink_area_dem', compress=True)
            success = True

        except:
            print("error")