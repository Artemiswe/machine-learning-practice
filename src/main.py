import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate,cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
import random
import os

# 设置随机种子
np.random.seed(42)
random.seed(42)
# 数据预处理函数
def preprocess_data(df):
    df['Year_diff'] = df['Year'] - 1972
    df['Host'] = df['Host'].astype(int)
    df['Country'] = df['Country'].str.replace(' ', '_', regex=False)
    return df


# 加载训练数据
df = pd.read_csv('AllAverage.csv')
df = preprocess_data(df)

# 特征和目标变量
features = ['PreMedal', 'Host', 'AthletesNum', 'Year_diff'] + [f'sport{i}' for i in range(1, 49)]
y = df['MedalNum']
X = df[features]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 贝叶斯优化的目标函数
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               random_state=42)

    # 使用交叉验证评估模型性能
    score = cross_val_score(rf, X_train, y_train, n_jobs=-1, cv=6, scoring='neg_mean_squared_error')
    return -score.mean()


def objective_gb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    gb = GradientBoostingRegressor(n_estimators=n_estimators,
                                   learning_rate=learning_rate,
                                   max_depth=max_depth,
                                   random_state=42)

    score = cross_val_score(gb, X_train, y_train, n_jobs=-1, cv=6, scoring='neg_mean_squared_error')
    return -score.mean()


def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)  # 改为 suggest_float
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)  # 改为 suggest_float

    xgb = XGBRegressor(n_estimators=n_estimators,
                       learning_rate=learning_rate,
                       max_depth=max_depth,
                       subsample=subsample,
                       colsample_bytree=colsample_bytree,
                       random_state=42)

    score = cross_val_score(xgb, X_train, y_train, n_jobs=-1, cv=6, scoring='neg_mean_squared_error')
    return -score.mean()


# 贝叶斯优化 - 随机森林
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=50)
best_rf_params = study_rf.best_params
print(f"Best Random Forest Parameters: {best_rf_params}")

# 贝叶斯优化 - 梯度提升
study_gb = optuna.create_study(direction='minimize')
study_gb.optimize(objective_gb, n_trials=50)
best_gb_params = study_gb.best_params
print(f"Best Gradient Boosting Parameters: {best_gb_params}")

# 贝叶斯优化 - XGBoost
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50)
best_xgb_params = study_xgb.best_params
print(f"Best XGBoost Parameters: {best_xgb_params}")

# 使用最佳参数训练模型
best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
best_gb_model = GradientBoostingRegressor(**best_gb_params, random_state=42)
best_xgb_model = XGBRegressor(**best_xgb_params, random_state=42,  early_stopping_rounds=None)

# 集成学习：Stacking
estimators = [('rf', best_rf_model),
              ('gb', best_gb_model),
              ('xgb', best_xgb_model)]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)


# 评估模型的性能
def evaluate_on_train_test(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, train_pred)
    mse_train = mean_squared_error(y_train, train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, train_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, test_pred)

    return mae_train, mse_train, rmse_train, r2_train, mae_test, mse_test, rmse_test, r2_test


# 模型字典
models = {
    'Random Forest': best_rf_model,
    'Gradient Boosting': best_gb_model,
    'XGBoost': best_xgb_model,
    'Stacking': stacking_model
}

results = {
    'Model': [],
    'MAE Train': [],
    'MSE Train': [],
    'RMSE Train': [],
    'R² Train': [],
    'MAE Test': [],
    'MSE Test': [],
    'RMSE Test': [],
    'R² Test': [],
}

# 定义输出文件名
output_file = "model_evaluation_results.csv"


# 定义函数来存储评估结果
def save_evaluation_results_to_csv(models, X_train, y_train, X_test, y_test, output_file):
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} on Train and Test sets:")
        mae_train, mse_train, rmse_train, r2_train, mae_test, mse_test, rmse_test, r2_test = evaluate_on_train_test(
            model, X_train, y_train, X_test, y_test
        )

        print(f"\nTraining Set Evaluation for {model_name}:")
        print(f"MAE: {mae_train}, MSE: {mse_train}, RMSE: {rmse_train}, R²: {r2_train}")

        print(f"\nTest Set Evaluation for {model_name}:")
        print(f"MAE: {mae_test}, MSE: {mse_test}, RMSE: {rmse_test}, R²: {r2_test}")

        results['Model'].append(model_name)
        results['MAE Train'].append(mae_train)
        results['MSE Train'].append(mse_train)
        results['RMSE Train'].append(rmse_train)
        results['R² Train'].append(r2_train)
        results['MAE Test'].append(mae_test)
        results['MSE Test'].append(mse_test)
        results['RMSE Test'].append(rmse_test)
        results['R² Test'].append(r2_test)

    results_df = pd.DataFrame(results)

    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode='a', index=False, header=False)
    else:
        results_df.to_csv(output_file, mode='w', index=False, header=True)

    print(f"\nEvaluation results saved to {output_file}")



save_evaluation_results_to_csv(models, X_train, y_train, X_test, y_test, output_file)
results_df = pd.DataFrame(results)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
width = 0.2
x = np.arange(len(results_df['Model']))  
offset = width  


bars_train = axes[0, 0].bar(x - offset, results_df['MAE Train'], width=width, label='Train')
bars_test = axes[0, 0].bar(x, results_df['MAE Test'], width=width, label='Test')
axes[0, 0].set_title('Mean Absolute Error (MAE)')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(results_df['Model'])
axes[0, 0].legend()


for bar in bars_train:
    yval = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
for bar in bars_test:
    yval = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')


bars_train = axes[0, 1].bar(x - offset, results_df['RMSE Train'], width=width, label='Train')
bars_test = axes[0, 1].bar(x, results_df['RMSE Test'], width=width, label='Test')
axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(results_df['Model'])
axes[0, 1].legend()


for bar in bars_train:
    yval = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
for bar in bars_test:
    yval = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')

bars_train = axes[1, 0].bar(x - offset, results_df['R² Train'], width=width, label='Train')
bars_test = axes[1, 0].bar(x, results_df['R² Test'], width=width, label='Test')
axes[1, 0].set_title('R² Score')
axes[1, 0].set_ylabel('R²')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(results_df['Model'])
axes[1, 0].legend()


for bar in bars_train:
    yval = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
for bar in bars_test:
    yval = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')


bars_train = axes[1, 1].bar(x - offset, results_df['MSE Train'], width=width, label='Train')
bars_test = axes[1, 1].bar(x, results_df['MSE Test'], width=width, label='Test')
axes[1, 1].set_title('Mean Squared Error (MSE)')
axes[1, 1].set_ylabel('MSE')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(results_df['Model'])
axes[1, 1].legend()

for bar in bars_train:
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
for bar in bars_test:
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 特征重要性分析
rf_importance = best_rf_model.feature_importances_
rf_importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_importance})
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=False)

gb_importance = best_gb_model.feature_importances_
gb_importance_df = pd.DataFrame({'Feature': features, 'Importance': gb_importance})
gb_importance_df = gb_importance_df.sort_values(by='Importance', ascending=False)


xgb_importance = best_xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importance})
xgb_importance_df = xgb_importance_df.sort_values(by='Importance', ascending=False)

top_10_rf_df = rf_importance_df.head(10)
top_10_gb_df = gb_importance_df.head(10)
top_10_xgb_df = xgb_importance_df.head(10)

print("\nTop 10 Features by Importance (Random Forest):")
print(top_10_rf_df)
print("\nTop 10 Features by Importance (Gradient Boosting):")
print(top_10_gb_df)
print("\nTop 10 Features by Importance (XGBoost):")
print(top_10_xgb_df)


gb_importance_df.to_csv('gb_All_feature_importance.csv', index=False)

plt.figure(figsize=(12, 8))

plt.subplot(131)
plt.barh(top_10_rf_df['Feature'], top_10_rf_df['Importance'], color='b')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

plt.subplot(132)
plt.barh(top_10_gb_df['Feature'], top_10_gb_df['Importance'], color='g')
plt.title('Gradient Boosting Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

plt.subplot(133)
plt.barh(top_10_xgb_df['Feature'], top_10_xgb_df['Importance'], color='r')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# 模型训练
best_rf_model.fit(X_train, y_train)
best_gb_model.fit(X_train, y_train)
best_xgb_model.fit(X_train, y_train)

# 预测结果
y_train_pred_rf = best_rf_model.predict(X_train)
y_train_pred_gb = best_gb_model.predict(X_train)
y_train_pred_xgb = best_xgb_model.predict(X_train)

# 可视化训练集预测与真实值的对比
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred_rf, alpha=0.7, color='orange', label='Random Forest')
plt.scatter(y_train, y_train_pred_gb, alpha=0.7, color='blue', label='Gradient Boosting')
plt.scatter(y_train, y_train_pred_xgb, alpha=0.7, color='green', label='XGBoost')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--',
         label='Perfect Prediction (y=x)')
plt.xlabel('Actual Total Medals (Train Data)')
plt.ylabel('Predicted Total Medals (Train Data)')
plt.title('Actual vs Predicted Total Medals (Train Data)')
plt.legend()
plt.show()

# 测试集预测
y_test_pred_rf = best_rf_model.predict(X_test)
y_test_pred_gb = best_gb_model.predict(X_test)
y_test_pred_xgb = best_xgb_model.predict(X_test)

# 可视化测试集预测与真实值的对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_rf, alpha=0.7, color='orange', label='Random Forest')
plt.scatter(y_test, y_test_pred_gb, alpha=0.7, color='blue', label='Gradient Boosting')
plt.scatter(y_test, y_test_pred_xgb, alpha=0.7, color='green', label='XGBoost')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
         label='Perfect Prediction (y=x)')
plt.xlabel('Actual Total Medals (Test Data)')
plt.ylabel('Predicted Total Medals (Test Data)')
plt.title('Actual vs Predicted Total Medals (Test Data)')
plt.legend()
plt.show()


# 计算预测区间的函数
def calculate_prediction_interval(model, X, confidence_interval=0.80):

    y_pred = model.predict(X)

    if hasattr(model, 'estimators_'): 
        if isinstance(model, RandomForestRegressor):
            pred_std = np.std([tree.predict(X) for tree in model.estimators_], axis=0)
        elif isinstance(model, GradientBoostingRegressor):
            pred_std = np.std([tree[0].predict(X) for tree in model.estimators_], axis=0)
    elif hasattr(model, 'get_booster'): 
        trees_preds = np.array([model.predict(X) for _ in model.get_booster().get_dump()])
        pred_std = np.std(trees_preds, axis=0)
    else:
        raise ValueError("Unsupported model type")


    z_score = 1.28 if confidence_interval == 0.80 else 1.96  # 1.28 for 80% CI, 1.96 for 95% CI
    lower_bound = y_pred - z_score * pred_std
    upper_bound = y_pred + z_score * pred_std

    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = np.maximum(upper_bound, 0) 

    return y_pred, lower_bound, upper_bound


df_2028 = pd.read_csv('All_Predict.csv')
df_2028 = preprocess_data(df_2028)


y_2028_pred_rf, total_lower_rf, total_upper_rf = calculate_prediction_interval(best_rf_model, df_2028[features])
y_2028_pred_gb, total_lower_gb, total_upper_gb = calculate_prediction_interval(best_gb_model, df_2028[features])
y_2028_pred_xgb, total_lower_xgb, total_upper_xgb = calculate_prediction_interval(best_xgb_model, df_2028[features])

df_2028['Total_2028_pred_rf'] = y_2028_pred_rf
df_2028['Total_2028_pred_gb'] = y_2028_pred_gb
df_2028['Total_2028_pred_xgb'] = y_2028_pred_xgb
df_2028['Total_2028_lower_rf'] = total_lower_rf
df_2028['Total_2028_upper_rf'] = total_upper_rf
df_2028['Total_2028_lower_gb'] = total_lower_gb
df_2028['Total_2028_upper_gb'] = total_upper_gb
df_2028['Total_2028_lower_xgb'] = total_lower_xgb
df_2028['Total_2028_upper_xgb'] = total_upper_xgb


df_unique = df_2028[['Country', 'Total_2028_pred_rf', 'Total_2028_lower_rf', 'Total_2028_upper_rf',
                     'Total_2028_pred_gb', 'Total_2028_lower_gb', 'Total_2028_upper_gb',
                     'Total_2028_pred_xgb', 'Total_2028_lower_xgb', 'Total_2028_upper_xgb']].drop_duplicates(
    subset='Country', keep='last')

top_10_countries = df_unique.sort_values(by='Total_2028_pred_rf', ascending=False).head(10)


plt.figure(figsize=(12, 6))
x = top_10_countries['Country']
y_pred_rf = top_10_countries['Total_2028_pred_rf']
y_pred_gb = top_10_countries['Total_2028_pred_gb']
y_pred_xgb = top_10_countries['Total_2028_pred_xgb']
y_lower_rf = top_10_countries['Total_2028_lower_rf']
y_upper_rf = top_10_countries['Total_2028_upper_rf']
y_lower_gb = top_10_countries['Total_2028_lower_gb']
y_upper_gb = top_10_countries['Total_2028_upper_gb']
y_lower_xgb = top_10_countries['Total_2028_lower_xgb']
y_upper_xgb = top_10_countries['Total_2028_upper_xgb']

bar_width = 0.25
index = np.arange(len(x))
plt.bar(index - bar_width, y_pred_rf, bar_width, label='RF Predicted 2028 Total Medals', color='orange', alpha=0.7)
plt.bar(index, y_pred_gb, bar_width, label='GB Predicted 2028 Total Medals', color='blue', alpha=0.7)
plt.bar(index + bar_width, y_pred_xgb, bar_width, label='XGB Predicted 2028 Total Medals', color='green', alpha=0.7)


plt.errorbar(index - bar_width, y_pred_rf, yerr=[y_pred_rf - y_lower_rf, y_upper_rf - y_pred_rf], fmt='none',
             color='black', capsize=5)
plt.errorbar(index, y_pred_gb, yerr=[y_pred_gb - y_lower_gb, y_upper_gb - y_pred_gb], fmt='none', color='black',
             capsize=5)
plt.errorbar(index + bar_width, y_pred_xgb, yerr=[y_pred_xgb - y_lower_xgb, y_upper_xgb - y_pred_xgb], fmt='none',
             color='black', capsize=5)

plt.xlabel('Country')
plt.ylabel('Medals')
plt.title('Predicted Total Medals for 2028 (Top 10 Countries)')
plt.xticks(index, x, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


df_results = df_unique[['Country', 'Total_2028_pred_rf', 'Total_2028_lower_rf', 'Total_2028_upper_rf',
                        'Total_2028_pred_gb', 'Total_2028_lower_gb', 'Total_2028_upper_gb',
                        'Total_2028_pred_xgb', 'Total_2028_lower_xgb', 'Total_2028_upper_xgb']]
df_results.to_csv('2028_Medal_predictions_comparison.csv', index=False)
print("Predictions saved to '2028_Bronze_predictions_comparison.csv'.")


def save_proportion_to_csv(proportion_rf, proportion_gb, proportion_xgb, file_name='All_proportions.csv'):
    new_data = {
        'Random Forest Proportion': [proportion_rf],
        'Gradient Boosting Proportion': [proportion_gb],
        'XGBoost Proportion': [proportion_xgb]
    }

    new_data_df = pd.DataFrame(new_data)

    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        updated_df.to_csv(file_name, index=False)
    else:
        new_data_df.to_csv(file_name, index=False)

    print(f"数据已保存到 {file_name}")

y_test_pred_rf, total_lower_rf, total_upper_rf = calculate_prediction_interval(best_rf_model, X_test,0.95)
y_test_pred_gb, total_lower_gb, total_upper_gb = calculate_prediction_interval(best_gb_model, X_test,0.95)
y_test_pred_xgb, total_lower_xgb, total_upper_xgb = calculate_prediction_interval(best_xgb_model, X_test,0.95)

def calculate_proportion_within_interval(y_true, lower_bound, upper_bound):
    """
    计算实际值落在预测区间内的比例
    """
    within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    return np.mean(within_interval)


proportion_rf = calculate_proportion_within_interval(y_test, total_lower_rf, total_upper_rf)
proportion_gb = calculate_proportion_within_interval(y_test, total_lower_gb, total_upper_gb)
proportion_xgb = calculate_proportion_within_interval(y_test, total_lower_xgb, total_upper_xgb)


print(f"Random Forest模型的实际值在预测区间内的比例: {proportion_rf:.2f}")
print(f"Gradient Boosting模型的实际值在预测区间内的比例: {proportion_gb:.2f}")
print(f"XGBoost模型的实际值在预测区间内的比例: {proportion_xgb:.2f}")

save_proportion_to_csv(proportion_rf, proportion_gb, proportion_xgb)

sport_features = [f'sport{i}' for i in range(1, 49)]
country_best_sport = {}
countries = df['Country'].unique()

for country in countries:
    df_country = df[df['Country'] == country]
    X_country = df_country[sport_features]
    y_country = df_country['Medal']

    rf_country = best_gb_model
    rf_country.fit(X_country, y_country)

    importances_country = rf_country.feature_importances_


    best_sport_idx = importances_country.argmax() 
    best_sport = sport_features[best_sport_idx] 
    country_best_sport[country] = best_sport


country_best_sport_df = pd.DataFrame(list(country_best_sport.items()), columns=['Country', 'BestSport'])
country_best_sport_df.to_csv('Medal_Country_Best_Sport.csv', index=False)
print("Best sports per country saved to 'Bronze_Country_Best_Sport.csv'.")

