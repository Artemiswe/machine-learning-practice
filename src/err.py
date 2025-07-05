import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 数据
df = pd.read_csv('error.csv')

# 将 'R² Test' 和 'R² Train' 转换为长格式
df_melted = df.melt(id_vars=['task', 'Model'], value_vars=['R² Test', 'R² Train'],
                    var_name='R² Type', value_name='R² Value')

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制盒须图
plt.figure(figsize=(16, 10))  # 增大图像大小

# 使用不同颜色区分任务和R²类型
ax = sns.boxplot(x='task', y='R² Value', hue='R² Type', data=df_melted, showfliers=False, palette="Set2")

# 添加任务间的分隔线
tasks = df['task'].unique()
for i in range(1, len(tasks)):
    plt.axvline(x=i - 0.5, color='black', linestyle='--', lw=1)  # 任务间分隔线

# 添加数据标注（显示中位数、四分位数等）
for i, box in enumerate(ax.artists):
    # 获取箱子的位置和大小
    median = ax.lines[i * 6 + 3].get_ydata()[0]
    q1 = ax.lines[i * 6 + 1].get_ydata()[0]
    q3 = ax.lines[i * 6 + 5].get_ydata()[0]

    # 在箱线图上显示数据
    ax.annotate(f'Median: {median:.2f}\nQ1: {q1:.2f}\nQ3: {q3:.2f}',
                xy=(box.get_x() + box.get_width() / 2, median),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=12, color='black')

# 设置标题和标签
plt.title('Performance Variability of GradientBoosting Model for Different Tasks', fontsize=22)  # 增大标题字号
plt.xlabel('Task', fontsize=18)  # 增大X轴标签字号
plt.ylabel('R² Value', fontsize=18)  # 增大Y轴标签字号
plt.xticks(rotation=45, fontsize=14)  # 增大X轴刻度字号
plt.yticks(fontsize=14)  # 增大Y轴刻度字号

# 显示图例
plt.legend(title='R² Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)  # 增大图例字体

# 调整布局，使得标签不重叠
plt.tight_layout()

# 显示图形
plt.show()
