# Shared Bikes Demand Analysis (Multiple Linear Regression)
# Author: Khushi Upmanyu
# How to run:
#   pip install -r requirements.txt
#   python bikes_analysis.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Load data
data = pd.read_csv('synthetic_bike_data.csv')

# Separate features and target
X = data.drop('count', axis=1)
y = data['count']

numeric_features = ['temp', 'humidity', 'windspeed']
categorical_features = ['season', 'workingday']

numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R2 Score: {r2:.3f}')
print(f'MSE: {mse:.2f}')

# VIF & Breusch-Pagan using statsmodels
X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded = sm.add_constant(X_encoded)
model = sm.OLS(y, X_encoded).fit()

vif_data = pd.DataFrame()
vif_data['Feature'] = X_encoded.columns
vif_data['VIF'] = [variance_inflation_factor(X_encoded.values, i) for i in range(X_encoded.shape[1])]
print('\nVariance Inflation Factors (VIF):')
print(vif_data)

bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f'\nBreusch-Pagan p-value: {bp_test[1]:.4f}')
