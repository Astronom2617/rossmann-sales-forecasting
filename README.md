# Rossmann Store Sales Forecasting

Time series forecasting for Rossmann drugstore chain sales.
Baseline → SARIMA → XGBoost comparison.

## Problem
Rossmann operates 1,115 drug stores across Germany.
Goal: predict daily sales for each store based on historical data.

## Dataset
Kaggle — [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- `train.csv` — 1M+ rows of historical sales (2013–2015)
- `store.csv` — store metadata (type, assortment, competition)

## Approach
**Baseline → Evaluation → Improvement**

1. EDA — seasonality, promo impact, store type analysis
2. Baseline — mean per store, then mean per store+day
3. SARIMA — classical time series model
4. XGBoost — gradient boosting with feature engineering

## Results

| Model | RMSE | RMSPE |
|---|---|---|
| Mean per store | €3267 | - |
| Mean per store+day | €2014 | - |
| SARIMA (1,0,1)(1,0,1,7) | €2140 | - |
| XGBoost | €1632 | **0.2871** |

RMSPE (Root Mean Squared Percentage Error) is the official Kaggle metric for this competition.
Calculated on open stores only (Sales > 0), which is consistent with the competition evaluation.

**XGBoost wins** — 50% improvement over naive baseline.

## Key Findings
- Strong weekly seasonality (Monday peak, Sunday near-zero due to German law)
- Promotions nearly double average sales (~€7,991 vs ~€4,406)
- Store type B outperforms others by ~2x (€10,058 vs ~€5,700)
- Simple rule-based baseline (store+day mean) outperforms SARIMA — 
  SARIMA ignores external features like promotions and store type
- XGBoost benefits from lag features (lag_7, lag_30) and all store metadata

## Features Used (XGBoost)
`Store`, `DayOfWeek`, `Promo`, `SchoolHoliday`, `CompetitionDistance`,
`month`, `year`, `day_of_week`, `lag_7`, `lag_30`

## Lessons Learned
- Simple rule-based baseline (store+day mean) outperforms SARIMA — external features matter more than autoregression
- XGBoost benefits most from lag features and store metadata
- RMSPE penalizes relative errors — a €100 error on a €200 sale is worse than on a €2000 sale

## Stack
Python, Pandas, NumPy, Matplotlib, Statsmodels, XGBoost
