import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd

# 设置工作环境
wbe = wbw.WbEnvironment()
wbe.verbose = False

wbe.working_directory = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\00_data_source'

# 读入栅格
mask = wbe.read_raster('Greve_mask_5m.tif')
dem = wbe.read_raster('Greve_5m.tif')

# 创建一个与DEM尺寸一致、用于初始化的栅格
layer = wbe.new_raster(dem.configs)

# 统计有效格点，将其索引保存到 grid 列表
grid = []
q = 1  # 仅用于示例
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        # 如果是DEM无效值，则直接设NoData
        if dem[row, col] == dem.configs.nodata:
            layer[row, col] = dem.configs.nodata
        # 如果mask是无效值，说明不在目标范围，设为0.0（不改高程）
        elif mask[row, col] == mask.configs.nodata:
            layer[row, col] = 0.0
        else:
            # 既不是DEM无效值，也不是mask无效值 → 有效像元
            layer[row, col] = 0.0
            grid.append(q)
            q += 1

# 计数有效格点的数量
n_grid = len(grid)
print(f"Number of valid cells (n_grid) = {n_grid}")

# ----------------------
# 定义MOO问题
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

class MyProblem(ElementwiseProblem):

    def __init__(self, n_grid, **kwargs):
        super().__init__(
            n_var=int(n_grid),
            n_obj=3,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=np.array([0.0] * n_grid),
            xu=np.array([2.0] * n_grid),
            **kwargs
        )
        self.n_grid = n_grid

    def _evaluate(self, x, out, *args, **kwargs):
        # x 本身就是一个数组，长度为 n_grid
        var_list = x

        # 目标函数1: earth_volume_function
        # 假设每个单元格面积是5m×5m = 25㎡，这里对 x[i] 做绝对值再相加
        earth_volume_function = sum(abs(val) for val in var_list) * 25

        # 目标函数2,3: sink_volume_function, sink_area_function
        sink_volume_function, sink_area_function = sink_sum_calculation(var_list)

        out["F"] = [earth_volume_function, sink_volume_function, sink_area_function]

def sink_sum_calculation(var_list):
    """
    传入一个长度为 n_grid 的列表 var_list，
    按与 DEM/mask 相同的逻辑逐像元赋值，得到新的 DEM，并计算洼地深度。
    """
    i = 0
    cut_and_fill = wbe.new_raster(dem.configs)

    # 给cut_and_fill赋值
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                cut_and_fill[row, col] = dem.configs.nodata
            elif mask[row, col] == mask.configs.nodata:
                cut_and_fill[row, col] = 0.0
            else:
                # 从var_list里依次拿值
                cut_and_fill[row, col] = round(var_list[i], 3)
                i += 1

    # 构造新的dem_pop = dem - cut_and_fill
    dem_pop = wbe.new_raster(dem.configs)
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                dem_pop[row, col] = dem.configs.nodata
            else:
                dem_pop[row, col] = dem[row, col] - cut_and_fill[row, col]

    # 计算洼地深度
    depth_in_sink = wbe.depth_in_sink(dem_pop, zero_background=False)

    # 蓄水量和洼地面积
    sink = wbe.new_raster(depth_in_sink.configs)
    Retention_volume = []

    for row in range(depth_in_sink.configs.rows):
        for col in range(depth_in_sink.configs.columns):
            if depth_in_sink[row, col] == depth_in_sink.configs.nodata:
                sink[row, col] = depth_in_sink.configs.nodata
            else:
                # 只将深度>=0.05视为可计算水深
                if depth_in_sink[row, col] >= 0.05:
                    sink[row, col] = depth_in_sink[row, col]
                    Retention_volume.append(depth_in_sink[row, col] * 25)
                else:
                    sink[row, col] = depth_in_sink.configs.nodata

    # 计算“洼地面积”（计数非NoData像元）
    # 这里 sink_value 为 1 if 不是NoData, 否则0
    sink_value = []
    for row in range(sink.configs.rows):
        for col in range(sink.configs.columns):
            if sink[row, col] == sink.configs.nodata:
                sink_value.append(0)
            else:
                sink_value.append(1)

    # 因为多目标优化中是最小化，所以把保水量/洼地面积取负号
    sink_volume_sum = -sum(Retention_volume)
    sink_area_sum = -sum(sink_value)
    return sink_volume_sum, sink_area_sum

# ----------------------
# 构造NSGA2算法
class RoundedSBX(SBX):
    def __init__(self, prob, eta, decimals=3):
        super().__init__(prob=prob, eta=eta)
        self.decimals = decimals

    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        return np.round(X, self.decimals)

class RoundedPM(PM):
    def __init__(self, eta, decimals=3):
        super().__init__(eta=eta)
        self.decimals = decimals

    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        return np.round(X, self.decimals)

class RoundedFloatRandomSampling(FloatRandomSampling):
    def __init__(self, decimals=3):
        super().__init__()
        self.decimals = decimals

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.round(X, self.decimals)

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=40,
    sampling=RoundedFloatRandomSampling(decimals=3),
    crossover=RoundedSBX(prob=0.8, eta=15, decimals=3),
    mutation=RoundedPM(eta=15, decimals=3),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 80)

from pymoo.optimize import minimize
problem = MyProblem(n_grid)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

# 取最终解和目标函数值
X = res.X
F = res.F

# 3D可视化
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.show()

plot_figure_path = 'scatter_plot_DEM5m.png'
plot.save(plot_figure_path)

# 保存前两列或全部结果
result_obj_df = pd.DataFrame(F, columns=['EarthVolume', 'SinkVolume', 'SinkArea'])
result_obj_df.to_csv('output_solution_DEM5m.csv', index=False)

result_var_df = pd.DataFrame(X)
result_var_df.to_csv('output_variable_DEM5m.csv', index=False)


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