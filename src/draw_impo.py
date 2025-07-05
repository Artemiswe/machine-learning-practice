import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 体育项目名称列表
sport_names = [
    "Aquatics", "Archery", "Athletics", "Badminton", "Baseball and Softball", "Basketball",
    "Basque Pelota", "Boxing", "Breaking", "Canoeing", "Cricket", "Croquet", "Cycling",
    "Equestrian", "Fencing", "Field Hockey", "Flag Football", "Football", "Golf", "Gymnastics",
    "Handball", "Jeu De Paume", "Judo", "Karate", "Lacrosse", "Modern Pentathlon", "Polo",
    "Rackets", "Roque", "Rowing", "Rugby", "Sailing", "Shooting", "Skateboarding", "Sport Climbing",
    "Squash", "Surfing", "Table Tennis", "Taekwondo", "Tennis", "Triathlon", "Tug Of War",
    "Volleyball", "Water Motorsports", "Weightlifting", "Wrestling", "Skating", "Ice Hockey"
]

# 映射sport1到sport48到实际的项目名称
sport_mapping = {f'sport{i}': sport_names[i-1] for i in range(1, 49)}

# 读取四个任务的特征重要性文件
gold_importance = pd.read_csv('gb_Gold_feature_importance.csv')
silver_importance = pd.read_csv('gb_Silver_feature_importance.csv')
bronze_importance = pd.read_csv('gb_Bronze_feature_importance.csv')
total_importance = pd.read_csv('gb_All_feature_importance.csv')

# 替换特征中的 sport1 到 sport48
gold_importance['Feature'] = gold_importance['Feature'].replace(sport_mapping)
silver_importance['Feature'] = silver_importance['Feature'].replace(sport_mapping)
bronze_importance['Feature'] = bronze_importance['Feature'].replace(sport_mapping)
total_importance['Feature'] = total_importance['Feature'].replace(sport_mapping)

# 选取每个任务的前 10 个特征
top_10_gold = gold_importance.head(10)
top_10_silver = silver_importance.head(10)
top_10_bronze = bronze_importance.head(10)
top_10_total = total_importance.head(10)

# 为了在同一张图上绘制，我们将任务名称加入每个数据框
top_10_gold['Task'] = 'Gold'
top_10_silver['Task'] = 'Silver'
top_10_bronze['Task'] = 'Bronze'
top_10_total['Task'] = 'Total'

# 合并所有任务的数据
all_tasks = pd.concat([top_10_gold, top_10_silver, top_10_bronze, top_10_total])

# 设置颜色调色板
palette = sns.color_palette("Set2", n_colors=4)

# 创建子图：2行2列，分别展示不同任务的特征重要性
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)

# 分别为每个子图绘制不同任务的特征重要性
tasks = ['Gold', 'Silver', 'Bronze', 'Total']
for i, task in enumerate(tasks):
    task_data = all_tasks[all_tasks['Task'] == task]
    sns.barplot(x='Importance', y='Feature', data=task_data, ax=axes[i//2, i%2], palette=palette)
    axes[i//2, i%2].set_title(f'{task} Medal Task')
    axes[i//2, i%2].set_xlabel('Importance')
    axes[i//2, i%2].set_ylabel('Feature')

    # 在条形图上标出特征重要性数值（保留三位小数）
    for index, value in enumerate(task_data['Importance']):
        axes[i//2, i%2].text(value, index, f'{value:.3f}', va='center', ha='left', fontsize=10)

# 调整布局，使得子图不重叠
plt.tight_layout()
plt.show()
