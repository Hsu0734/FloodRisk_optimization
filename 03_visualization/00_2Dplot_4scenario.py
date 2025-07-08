import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


# ——— 自定义截取渐变色图函数 ———
def truncated_cmap(cmap_name, minval=0.3, maxval=0.9, n=256):
    base = plt.cm.get_cmap(cmap_name, n)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f'{cmap_name}_{minval}_{maxval}',
        base(np.linspace(minval, maxval, n))
    )
    return new_cmap


# ── 四组数据文件、配色、标签 ──
file_paths = [
    (r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\output_solution_DEM5m_scenario1.csv', 'scenario[0, 0.5]', truncated_cmap('Reds')),
    (r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\output_solution_DEM5m_scenario2.csv', 'scenario[0, 1.0]', truncated_cmap('Greens')),
    (r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\output_solution_DEM5m_scenario3.csv', 'scenario[0, 1.5]', truncated_cmap('Blues')),
    (r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\output_solution_DEM5m_scenario4.csv', 'scenario[0, 2.0]', truncated_cmap('YlOrBr'))
]

fig, ax = plt.subplots(figsize=(30, 20))

sm_list, legend_handles = [], []

for path, lbl, cmap in file_paths:
    df = pd.read_csv(path).abs()
    df.iloc[:, 2] *= 25  # 第 3 列 ×25（洼地面积）

    x, y, c = df.iloc[:, 2], df.iloc[:, 1], df.iloc[:, 0]
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=150, alpha=0.9, edgecolors='none')

    sm = ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=c.min(), vmax=c.max()))
    sm_list.append(sm)

    # 图例取中间色
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', linestyle='', markersize=22,
                   markerfacecolor=cmap(0.5), label=lbl)
    )

# 轴标签 & 样式
ax.set_xlabel(r'$A_{sink area}$ ($m^2$)', fontsize=36, labelpad=24)
ax.set_ylabel(r'$V_{retention volume}$ ($m^3$)', fontsize=36, labelpad=24)
ax.tick_params(axis='both', labelsize=35)
ax.grid(True, linestyle='--', alpha=0.5)

# 图例
ax.legend(handles=legend_handles, fontsize=26, loc='upper left',
          title='Scenarios', title_fontsize=28, frameon=False)

# 独立色条
divider = make_axes_locatable(ax)
for i, (sm, lbl) in enumerate(zip(sm_list, [f[1] for f in file_paths])):
    cax = divider.append_axes("right", size="1.5%", pad=0.5 + i * 0.12)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f'{lbl}\n$V_{{total\\ earthwork}}$ (m$^3$)', fontsize=22, labelpad=10)
    cb.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig(r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\pareto_front_4scenarios.png',
            dpi=300, bbox_inches='tight')
plt.show()