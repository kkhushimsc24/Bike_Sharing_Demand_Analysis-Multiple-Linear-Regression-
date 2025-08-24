# Shared Bikes Demand Analysis (Multiple Linear Regression)

## Objective
Build a regression model to understand & predict bike demand and identify key drivers.

## Dataset
- `synthetic_bike_data.csv` (structured like Kaggle bike sharing data). Replace with Kaggle dataset for real-world results.

## Steps
1. EDA & preprocessing: scaling (MinMaxScaler), one-hot encoding for categorical variables.
2. Modeling: Linear Regression pipeline with train/test split.
3. Diagnostics: Variance Inflation Factor (VIF) for multicollinearity; Breusch–Pagan test for heteroscedasticity.

## How to Run
```bash
pip install -r requirements.txt
python bikes_analysis.py
```

## Expected Output
- R² & MSE on test split
- VIF table
- Breusch–Pagan p-value

## Notes for Interview
- Explain why linear regression fits a continuous target.
- Discuss impact of variables (e.g., temp positive, humidity negative).
- Mention assumptions checked (multicollinearity, homoscedasticity).
