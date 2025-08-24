
# 🚲 Shared Bikes Demand Analysis (Multiple Linear Regression)

## 🎯 Objective
Build a regression model to understand & predict bike demand and identify key drivers.

---

## 📂 Dataset
- `synthetic_bike_data.csv` (structured like Kaggle Bike Sharing dataset).  
- For real-world replication, replace with the **Kaggle [Bike Sharing Demand Dataset](https://www.kaggle.com/c/bike-sharing-demand)**.  

---

## ⚙️ Steps

### 🔍 EDA & Preprocessing
- Exploratory Data Analysis (EDA)  
- Scaling with **MinMaxScaler**  
- One-hot encoding for categorical variables  

### 🧮 Modeling
- **Multiple Linear Regression** pipeline  
- Train/Test split for evaluation  

### 📊 Diagnostics
- **Variance Inflation Factor (VIF)** → multicollinearity check  
- **Breusch–Pagan Test** → heteroscedasticity check  

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python bikes_analysis.py
