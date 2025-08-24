# Credit Risk – Monte Carlo Simulation & ML

This project applies **Monte Carlo simulation** to model credit-risk drivers and estimate default likelihood, then trains a **RandomForest** baseline for prediction. It includes data cleaning, exploratory analysis, simulated scenarios, and evaluation metrics.

## Key Features
- Data prep: missing values, outlier handling, categorical encoding.
- EDA: histograms, bar charts, scatter plots, correlation heatmap.
- Monte Carlo: draws for age, income, employment, loan attributes; scenario-based default scoring.
- ML baseline: RandomForestClassifier with standardized features and a classification report.

## Tech Stack
Python · pandas · numpy · matplotlib · seaborn · scikit-learn

## Data
Uses the public **credit_risk_dataset.csv** (Kaggle).  
> Place the file at `data/credit_risk_dataset.csv` or update the path in the notebook/script.

## Getting Started
```bash
git clone https://github.com/KaoutharJribi/credit-risk-monte-carlo.git
cd credit-risk-monte-carlo
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
