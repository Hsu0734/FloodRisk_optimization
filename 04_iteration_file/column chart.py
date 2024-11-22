import matplotlib.pyplot as plt

category_colors = [
    '#a6cee3', # Water management
    '#b2df8a', # Ecological environment
    '#fdbf6f', # Service management
    '#cab2d6', # Space design
    '#fb9a99'  # Users perception
]

# Define the number of bars belonging to each category
categories = [7, 6, 7, 6, 7]

# Assign colors to each bar according to the category
colors = []
for i, count in enumerate(categories):
    colors.extend([category_colors[i]] * count)

# 指标名称
labels = [
    "Groundwater level control", "Water drainage", "Water infiltration", "Water pollution control",
    "Water purification", "Water recycling and utilization", "Water retention and detention", "Air quality",
    "Biodiversity protection", "Landscape scenery resource", "Local microclimate",
    "Plant growth", "Waterbody protection", "Commercial facilities",
    "Entertainment facilities", "Facilities maintenance", "Landmark and logo design", "Lighting infrastructure",
    "Rest facilities", "Sanitation service", "Barrier-Free Design",
    "Construction process and material", "Elevation design", "Facilities layout",
    "Road Accessibility", "Space Size", "Cultural influence",
    "Education effect", "Livability of surrounding places", "Mental well-being", "Neighborhood relationship",
    "Physical well-being", "Safety of surrounding places"
]

# 平均权重值
weights = [
    0.027514286, 0.038142857, 0.026714286, 0.026014286, 0.013828571,
    0.030614286, 0.037185714, 0.012614286, 0.030442857, 0.041642857,
    0.0254, 0.043614286, 0.046328571, 0.009871429, 0.020671429,
    0.0669, 0.013071429, 0.0219, 0.032614286, 0.035042857, 0.022328571,
    0.029114286, 0.040357143, 0.038, 0.0346, 0.035628571, 0.022228571,
    0.043742857, 0.042385714, 0.016257143, 0.020571429, 0.020085714, 0.0348
]

# 创建柱状图
plt.figure(figsize=(12, 7))
bars = plt.bar(labels, weights, color=colors)

# 设置图表标题和标签
plt.title("Average Weights of Indicators")
plt.xlabel("Indicators")
plt.ylabel("Weight")

# 旋转横坐标标签
plt.xticks(rotation=45, ha='right')

legend_labels = ['Water management', 'Ecological environment', 'Service management', 'Space design', 'Users perception']
legend_colors = category_colors
for color, label in zip(legend_colors, legend_labels):
    plt.bar(0, 0, color=color, label=label)  # Adding invisible bars for legend

plt.legend(loc='upper right')

# 显示图表
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('weight.png', dpi=300, bbox_inches='tight')
plt.show()
