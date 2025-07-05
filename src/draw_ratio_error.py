import pandas as pd
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('model_performance_results.csv')

# 2. 提取不同任务的 R² Train 和 R² Test 数据
tasks = ["Gold", "Bronze", "All", "Silver"]
train_ratios = data["Train Ratio"].unique()

# 创建一个字典来存储每个任务的 R² Train 和 R² Test
task_data = {task: {"R² Train": [], "R² Test": []} for task in tasks}

for task in tasks:
    task_df = data[data["Task"] == task]
    task_data[task]["R² Train"] = task_df["R² Train"].values
    task_data[task]["R² Test"] = task_df["R² Test"].values

# 3. 绘制折线图
plt.figure(figsize=(12, 8))

# 定义线条的样式和颜色
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
colors = ['b', 'g', 'r', 'c']  # Blue, Green, Red, Cyan

# 为每个任务绘制折线图
for i, task in enumerate(tasks):
    plt.plot(train_ratios, task_data[task]["R² Train"], label=f'{task} - R² Train',
             marker=markers[i], linestyle=line_styles[i], color=colors[i], markersize=8)
    plt.plot(train_ratios, task_data[task]["R² Test"], label=f'{task} - R² Test',
             marker=markers[i], linestyle=line_styles[i], color=colors[i], markersize=8, alpha=0.7)

# 4. 设置图表属性
plt.title("Model Evaluation: R² Train and R² Test by Train Ratio", fontsize=16)
plt.xlabel("Train Ratio", fontsize=14)
plt.ylabel("R²", fontsize=14)

# 调整图例位置，放置在图表外部
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# 添加网格
plt.grid(True)

# 调整图表的布局，使图例不重叠
plt.tight_layout()

# 显示图表
plt.show()
