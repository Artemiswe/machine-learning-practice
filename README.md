# Medal Prediction using Ensemble Learning

This project explores the application of ensemble learning techniques—such as Gradient Boosting, Random Forest, and XGBoost—to predict Olympic medal counts (gold, silver, bronze, and total) based on structured national-level data. The focus lies on both **predictive performance** and **model interpretability**, with extended analyses such as SHAP value interpretation and uncertainty evaluation.

## 🔍 Motivation

In scenarios involving moderately sized structured datasets with complex feature dependencies, ensemble models are often capable of achieving high prediction accuracy while maintaining robustness. This project aims to evaluate such models' performance in multi-target regression settings and investigate how feature importance and error distribution vary across sub-tasks.

## 📁 Project Structure

```bash
machine-learning-practice/
├── data/                        # Cleaned CSV files for medal prediction tasks
│   ├── AllAverage.csv
│   ├── GoldAverage.csv
│   ├── SilverAverage.csv
│   ├── BronzeAverage.csv
│   ├── All_Predict.csv
│   ├── Gold_Predict.csv
│   ├── Silver_Predict.csv
│   └── Bronze_Predict.csv
├── src/                         # Python scripts
│   ├── main.py                  # Main training script (for total medal task)
│   ├── drawmap.py               # Country-wise SHAP visualizer
│   ├── draw_impo.py             # Feature importance plotter
│   ├── draw_ratio_error.py      # Error/r² trend visualizer
│   └── err.py                   # Custom error metric calculation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .gitignore
````

## 📊 Datasets

All datasets are stored in `./data/` and are pre-cleaned. They include historical medal performance, athlete numbers, and country-level sports indicators.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Artemiswe/machine-learning-practice.git
cd machine-learning-practice
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the main script

```bash
python src/main.py
```

Additional visualization scripts (e.g., SHAP maps, error trend analysis) can be run independently after training.

## 📈 Sample Outputs

* Feature Importance (SHAP)
* Key-Sport Distribution by Country (Map)
* Training/Test Error Curves
* Model Comparison Table (MAE, RMSE, R², CPI)

All figures were generated using real model outputs and stored datasets.

## 🛠 Requirements

Main packages include:

* pandas
* numpy
* scikit-learn
* matplotlib
* shap
* seaborn
* optuna
* geopandas
* plotly

## 🙌 Acknowledgements

Special thanks to my teammate for their significant contribution in **data collection, cleaning, and preprocessing**, which laid the foundation for this project.


