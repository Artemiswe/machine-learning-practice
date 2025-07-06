
# Structured Data Modeling with Ensemble Learning

This repository presents two real-world applications of ensemble learning techniques on structured datasets:

- 🥇 **Olympic Medal Prediction** based on national-level historical sports data
- ⚰️ **Mortality Rate Modeling** across urban and rural regions in China

Both tasks leverage models such as **Gradient Boosting**, **Random Forest**, and **XGBoost**. The focus is not only on predictive performance, but also on model interpretability (via SHAP, partial dependence) and uncertainty evaluation.


## 🔍 Motivation

In scenarios involving moderately sized structured datasets with complex feature dependencies, ensemble models are known to provide a good balance between accuracy and robustness. This project investigates their performance across **multi-output regression tasks**, while also assessing their reliability and interpretability.


## 🧪 Tasks Overview

### 🥇 Medal Prediction

Predict the gold, silver, bronze, and total medal counts of each country based on historical Olympic data. Key features include:

- Athlete numbers
- Host country indicator
- Past medal history
- Event-wise per-capita performance (48 sports)

Additional modules include:

- SHAP-based feature attribution
- Country-wise sports dominance mapping
- CPI score for model comparison

### ⚰️ Mortality Rate Modeling

Forecast age-standardized mortality rates (ASMR) for **urban and rural regions** in China in 2010 and 2020 using socioeconomic and environmental indicators.

Highlights include:

- VIF-based multicollinearity reduction
- BayesSearchCV hyperparameter optimization
- Feature importance and partial dependence plots
- Multi-model performance variance analysis

## 📁 Project Structure

```bash
machine-learning-practice/
├── data/                        # Medal prediction datasets
│   ├── AllAverage.csv
│   ├── GoldAverage.csv
│   ├── SilverAverage.csv
│   ├── BronzeAverage.csv
│   ├── All_Predict.csv
│   └── ... (more medal data)
├── src/                         # Medal prediction scripts
│   ├── main.py
│   ├── drawmap.py
│   ├── draw_impo.py
│   ├── draw_ratio_error.py
│   └── err.py
├── mortality/                  # Mortality prediction task
│   ├── predict.py                 # Full mortality modeling and analysis pipeline
│   ├── data1.csv               # Urban mortality dataset
│   └── data2.csv               # Rural mortality dataset
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
````



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

### 3. Run Medal Prediction Task

```bash
python src/main.py
```

### 4. Run Mortality Modeling Task

```bash
cd mortality
python predict.py 
```



## 📊 Sample Outputs

* Medal Prediction:

  * SHAP Feature Importance Bar Charts
  * Sports Dominance Map by Country
  * Error/R² Trend Charts
  * CPI-Based Model Comparison

* Mortality Modeling:

  * VIF Tables for Urban/Rural
  * Feature Importance Rankings (by model)
  * Partial Dependence Curves
  * Model Ranking with Variance-Aware Score



## 🛠 Requirements

Main Python libraries used:

* `pandas`, `numpy`
* `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
* `optuna`, `scikit-optimize`
* `shap`, `seaborn`, `matplotlib`
* `geopandas`, `plotly`



## 🙌 Acknowledgements

**Special thanks to my teammates for their valuable help in data collection and preprocessing**.

All modeling design, algorithm implementation, evaluation, and visualization—across both the **medal prediction** and **mortality modeling** tasks—were independently completed by the author.
