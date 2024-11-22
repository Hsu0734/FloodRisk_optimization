import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据
sorted_data = {
    'Items': [
        # Water management
        "Groundwater level control", "Water drainage", "Water infiltration", "Water pollution control",
        "Water purification", "Water recycling and utilization", "Water retention and detention",
        # Ecological environment
        "Air quality", "Biodiversity protection", "Landscape scenery resource", "Local microclimate",
        "Plant growth", "Waterbody protection",
        # Service management
        "Commercial facilities", "Entertainment facilities", "Facilities maintenance",
        "Landmark and logo design", "Lighting infrastructure", "Rest facilities", "Sanitation service",
        # Space design
        "Barrier-Free Design", "Construction process and material", "Elevation design",
        "Facilities layout", "Road Accessibility", "Space Size",
        # Users perception
        "Cultural influence", "Education effect", "Livability of surrounding places",
        "Mental well-being", "Neighbourhood relationship", "Physical well-being", "Safety of surrounding places"
    ],
    'Scores': [
        4.1863, 4.2745, 4.4020, 4.2941, 4.3333, 4.2353, 4.3529,
        4.5784, 4.3529, 4.4804, 4.4608, 4.5686, 4.2353,
        3.7549, 4.0980, 4.3137, 4.4118, 4.2745, 4.4216, 4.4608,
        4.4314, 4.3529, 4.4412, 4.3333,  4.3039, 4.4706,
        4.3529, 4.3235, 4.4412, 4.5294, 4.4608, 4.5392, 4.4706
    ]
}

# 创建 DataFrame
df_sorted = pd.DataFrame(sorted_data)

# 定义颜色列表对应每个类别
category_colors = [
    '#a6cee3', # Water management
    '#a6cee3', # Water management
    '#a6cee3', # Water management
    '#a6cee3', # Water management
    '#a6cee3', # Water management
    '#a6cee3', # Water management
    '#a6cee3', # Water management
    '#b2df8a', # Ecological environment
    '#b2df8a', # Ecological environment
    '#b2df8a', # Ecological environment
    '#b2df8a', # Ecological environment
    '#b2df8a', # Ecological environment
    '#b2df8a', # Ecological environment
    '#fdbf6f', # Service management
    '#fdbf6f', # Service management
    '#fdbf6f', # Service management
    '#fdbf6f', # Service management
    '#fdbf6f', # Service management
    '#fdbf6f', # Service management
    '#fdbf6f', # Service management
    '#cab2d6', # Space design
    '#cab2d6', # Space design
    '#cab2d6', # Space design
    '#cab2d6', # Space design
    '#cab2d6', # Space design
    '#cab2d6', # Space design
    '#fb9a99', # Users perception
    '#fb9a99', # Users perception
    '#fb9a99', # Users perception
    '#fb9a99', # Users perception
    '#fb9a99', # Users perception
    '#fb9a99', # Users perception
    '#fb9a99'  # Users perception
]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制条形图
bars = ax.bar(df_sorted['Items'], df_sorted['Scores'], color=category_colors)

# 设置横纵坐标标签及标题
ax.set_title('Fuzzy Comprehensive Evaluation Results by Category', fontsize=16)
ax.set_ylabel('Scores', fontsize=14)
ax.set_xlabel('Items', fontsize=14)

# 旋转横坐标标签 45 度
plt.xticks(rotation=45, ha='right')

# 添加次坐标轴用于折线显示 0-3 段
#ax2 = ax.twinx()
#ax2.set_ylim(0, 3)  # 设置 0 到 3 范围的折线
#ax2.plot(df_sorted['Items'], np.minimum(df_sorted['Scores'], 3), linestyle='--', color='black')

# 主要坐标轴限制为 3 到 5，集中显示
ax.set_ylim(3, 5)

# 添加图例，使用不可见的柱子创建图例
legend_labels = ['Water management', 'Ecological environment', 'Service management', 'Space design', 'Users perception']
legend_colors = ['#a6cee3', '#b2df8a', '#fdbf6f', '#cab2d6', '#fb9a99']

for color, label in zip(legend_colors, legend_labels):
    plt.bar(0, 0, color=color, label=label)  # 添加不可见的柱用于图例

plt.legend(loc='upper right')

# 调整图表布局
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('fuzzy_evaluation_results02.png', dpi=300, bbox_inches='tight')
plt.show()
