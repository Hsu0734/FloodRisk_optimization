import pandas as pd
import matplotlib.pyplot as plt

# Load the newly uploaded fourth dataset
file_path_fourth = r'D:\PhD career\05 SCI papers\08 Topographic modification optimization\FloodRisk_optimization\03_visualization\output_solution_DEM5m.csv'
data_fourth = pd.read_csv(file_path_fourth)

# Step 1: Convert all data to positive numbers
data_fourth_positive = data_fourth.abs()

# Step 2: Multiply the third column by 25
data_fourth_positive.iloc[:, 2] = data_fourth_positive.iloc[:, 2] * 25

# Step 3: Extract objectives for 2D visualization
objective_1 = data_fourth_positive.iloc[:, 0]
objective_2 = data_fourth_positive.iloc[:, 1]
objective_3 = data_fourth_positive.iloc[:, 2]

# Create the 2D scatter plot
plt.figure(figsize=(12, 10))
sc = plt.scatter(objective_3, objective_2, c=objective_1, cmap='viridis', s=30, marker='o')

# Label axes with the updated names
plt.xlabel(r'$A_{sink area}$ ($m^2$)', fontsize=24)
plt.ylabel(r'$V_{retention volume}$ ($m^3$)', fontsize=24)

# Add a color bar to represent Objective 1 (Earthwork volume)
plt.colorbar(sc, label=r'Total Earthwork Volume $V_{total earthwork}$ ($m^3$)')

#标签数字大小
plt.tick_params(axis='both', which='major', labelsize=20)

# Show plot
plt.grid(True)
plt.savefig('pareto_front_2d_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
