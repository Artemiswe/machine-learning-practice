import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from matplotlib import font_manager

# 设置字体
font_path = "C:/Windows/Fonts/simsun.ttc"  # SimSun 字体路径
font_prop = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子
seed = 42
np.random.seed(seed)

# 抑制特定警告信息
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")

# 读取城市和乡村的数据
data_urban = pd.read_csv(r'C:\Users\13825\Desktop\data1.csv', encoding='latin1')
data_rural = pd.read_csv(r'C:\Users\13825\Desktop\data2.csv', encoding='latin1')  # 假设乡村数据文件路径为data2.csv

# 添加标记列
data_urban['region_type'] = 'urban'
data_rural['region_type'] = 'rural'

# 合并数据集
data = pd.concat([data_urban, data_rural], ignore_index=True)

# 分别计算VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# 计算城市和乡村的VIF值
data_urban_scaled = StandardScaler().fit_transform(data_urban.drop(columns=['year', 'ASMR', 'region_type']))
vif_urban = calculate_vif(
    pd.DataFrame(data_urban_scaled, columns=data_urban.columns.difference(['year', 'ASMR', 'region_type'])))
print("城市数据的VIF值:\n", vif_urban)

data_rural_scaled = StandardScaler().fit_transform(data_rural.drop(columns=['year', 'ASMR', 'region_type']))
vif_rural = calculate_vif(
    pd.DataFrame(data_rural_scaled, columns=data_rural.columns.difference(['year', 'ASMR', 'region_type'])))
print("乡村数据的VIF值:\n", vif_rural)

# 保留城市和乡村中VIF值都低于阈值的变量
vif_threshold = 10
selected_features_urban = vif_urban[vif_urban["VIF"] <= vif_threshold]["feature"].tolist()
selected_features_rural = vif_rural[vif_rural["VIF"] <= vif_threshold]["feature"].tolist()

# 取交集
selected_features = list(set(selected_features_urban) & set(selected_features_rural))
selected_features.extend(['year', 'ASMR', 'region_type'])

data_selected = data[selected_features]

# 分年份数据
data_2010 = data_selected[data_selected['year'] == 2010]
data_2020 = data_selected[data_selected['year'] == 2020]

# 分城市和乡村数据
data_urban_2010 = data_2010[data_2010['region_type'] == 'urban']
data_urban_2020 = data_2020[data_2020['region_type'] == 'urban']
data_rural_2010 = data_2010[data_2010['region_type'] == 'rural']
data_rural_2020 = data_2020[data_2020['region_type'] == 'rural']

# 模型训练和评估函数
def model_and_evaluate(data, year, region_type):
    X = data.drop(columns=['ASMR', 'year', 'region_type'])
    y = data['ASMR']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": lgb.LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    params = {
        "RandomForest": {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'max_features': Categorical(['sqrt', 'log2', None])
        },
        "GradientBoosting": {
            'n_estimators': Integer(100, 300),
            'learning_rate': Real(0.01, 0.2),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5)
        },
        "XGBoost": {
            'n_estimators': Integer(100, 300),
            'learning_rate': Real(0.01, 0.2),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.5, 1.0)
        },
        "LightGBM": {
            'n_estimators': Integer(100, 300),
            'learning_rate': Real(0.01, 0.2),
            'num_leaves': Integer(31, 100),
            'max_depth': Integer(-1, 30),
            'min_data_in_leaf': Integer(20, 100)
        },
        "CatBoost": {
            'iterations': Integer(100, 300),
            'learning_rate': Real(0.01, 0.2),
            'depth': Integer(3, 10)
        }
    }

    results = {}
    for name, model in models.items():
        param_space = params[name]
        search = BayesSearchCV(model, param_space, n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=seed)
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_

        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        results[name] = {
            "best_params": best_model.get_params(),
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "r2": r2,
            "mae": mae,
            "feature_importances": getattr(best_model, 'feature_importances_', None)
        }
        print(f"{year} 年 {region_type} {name} 最佳参数: {best_model.get_params()}")
        print(f"{year} 年 {region_type} {name} 训练均方误差(MSE): {train_mse}")
        print(f"{year} 年 {region_type} {name} 测试均方误差(MSE): {test_mse}")
        print(f"{year} 年 {region_type} {name} 训练均方根误差(RMSE): {train_rmse}")
        print(f"{year} 年 {region_type} {name} 测试均方根误差(RMSE): {test_rmse}")
        print(f"{year} 年 {region_type} {name} 决定系数(R^2): {r2}")
        print(f"{year} 年 {region_type} {name} 平均绝对误差(MAE): {mae}")
        print()

    return results, X  # 返回X以便后续使用

# 分别训练和优化2010年和2020年城市和乡村的数据
results_urban_2010, X_urban_2010 = model_and_evaluate(data_urban_2010, 2010, '城市')
results_urban_2020, X_urban_2020 = model_and_evaluate(data_urban_2020, 2020, '城市')
results_rural_2010, X_rural_2010 = model_and_evaluate(data_rural_2010, 2010, '乡村')
results_rural_2020, X_rural_2020 = model_and_evaluate(data_rural_2020, 2020, '乡村')

# 优化后的模型比较
model_names = ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
years = [2010, 2020]
regions = ['城市', '乡村']

# 准备数据
model_data = []
for year in years:
    for region in regions:
        results = results_urban_2010 if year == 2010 and region == '城市' else \
            results_urban_2020 if year == 2020 and region == '城市' else \
                results_rural_2010 if year == 2010 and region == '乡村' else \
                    results_rural_2020
        for model in model_names:
            if model in results:
                model_data.append({
                    'Model': model,
                    'Year': year,
                    'Region': region,
                    'MSE': results[model]['test_mse'],
                    'RMSE': results[model]['test_rmse'],
                    'R²': results[model]['r2'],
                    'MAE': results[model]['mae']
                })

# 创建 DataFrame
model_comparison = pd.DataFrame(model_data)

# 计算平均性能和方差
model_performance = model_comparison.groupby('Model').agg(
    avg_rmse=('RMSE', 'mean'),
    avg_r2=('R²', 'mean'),
    var_rmse=('RMSE', 'var'),
    var_r2=('R²', 'var')
).reset_index()

# 综合评分
# 这里，我们定义一个综合评分函数，假设我们同等考虑平均性能和方差
model_performance['score'] = model_performance['avg_rmse'] + \
                             (1 - model_performance['avg_r2']) + \
                             model_performance['var_rmse'] + \
                             model_performance['var_r2']

# 选择综合评分最低的模型
best_model_name = model_performance.loc[model_performance['score'].idxmin(), 'Model']
print(f"在2010年和2020年城乡都表现较优的模型是: {best_model_name}")

# 获取最优模型
def get_best_model_from_results(results, model_name):
    for year in years:
        for region in regions:
            results = results_urban_2010 if year == 2010 and region == '城市' else \
                results_urban_2020 if year == 2020 and region == '城市' else \
                    results_rural_2010 if year == 2010 and region == '乡村' else \
                        results_rural_2020
            if model_name in results:
                return results[model_name]
    return None

best_model_results = get_best_model_from_results(
    {**results_urban_2010, **results_urban_2020, **results_rural_2010, **results_rural_2020},
    best_model_name
)

# 绘制特征重要性图
def plot_feature_importance(model_results, X, year, region):
    if 'feature_importances' in model_results and model_results['feature_importances'] is not None:
        feature_importances = model_results['feature_importances']
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importances of Best Model ({best_model_name}) for {year} {region}')
        plt.show()

# 分析变量的非线性趋势
def plot_partial_dependence(model, X, features, year, region):
    fig, ax = plt.subplots(figsize=(12, 8))
    pdp = partial_dependence(model, X, features=features)
    for i, feature in enumerate(features):
        ax.plot(pdp['values'][i], pdp['average'][i], label=feature)
    ax.set_xlabel('Feature value')
    ax.set_ylabel('Partial dependence')
    ax.legend()
    plt.title(f'Partial Dependence Plots of Best Model ({best_model_name}) for {year} {region}')
    plt.show()

# 获取最佳模型及其数据集
for year in years:
    for region in regions:
        if year == 2010 and region == '城市':
            X_best = X_urban_2010
            best_model_results = results_urban_2010[best_model_name]
        elif year == 2020 and region == '城市':
            X_best = X_urban_2020
            best_model_results = results_urban_2020[best_model_name]
        elif year == 2010 and region == '乡村':
            X_best = X_rural_2010
            best_model_results = results_rural_2010[best_model_name]
        elif year == 2020 and region == '乡村':
            X_best = X_rural_2020
            best_model_results = results_rural_2020[best_model_name]

        plot_feature_importance(best_model_results, X_best, year, region)

        # 确保使用部分依赖图绘制功能的模型
        if isinstance(best_model_results, (xgb.XGBRegressor, lgb.LGBMRegressor, CatBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)):
            plot_partial_dependence(best_model_results, X_best, X_best.columns, year, region)
