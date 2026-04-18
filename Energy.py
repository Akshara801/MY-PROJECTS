# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 2. GENERATE SYNTHETIC DATASET
# ============================================

np.random.seed(42)

# Create datetime range
date_range = pd.date_range(start='2023-01-01', periods=1000, freq='H')

# Create base dataframe
data = pd.DataFrame({
    'datetime': date_range,
    'temperature': np.random.normal(30, 5, 1000),
    'humidity': np.random.normal(60, 10, 1000)
})

# Create realistic energy consumption
data['energy_consumption'] = (
    50 +
    (data['temperature'] * 2) +
    (data['humidity'] * 0.5) +
    np.random.normal(0, 5, 1000)
)

# ============================================
# 3. DATA CLEANING
# ============================================

# Check missing values
print("Missing values:\n", data.isnull().sum())

# Drop missing values if any
data = data.dropna()

# ============================================
# 4. FEATURE ENGINEERING
# ============================================

data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month

# ============================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

plt.figure(figsize=(10, 5))
plt.plot(data['energy_consumption'])
plt.title("Energy Consumption Over Time")
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption")
plt.show()

# ============================================
# 6. PREPARE DATA FOR MODEL
# ============================================

features = ['temperature', 'humidity', 'hour', 'day', 'month']
X = data[features]
y = data['energy_consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 7. TRAIN MODEL (RANDOM FOREST)
# ============================================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============================================
# 8. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 9. MODEL EVALUATION
# ============================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R² Score:", r2)

# ============================================
# 10. VISUALIZATION (ACTUAL vs PREDICTED)
# ============================================

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Sample Index")
plt.ylabel("Energy")
plt.show()

# ============================================
# 11. FEATURE IMPORTANCE (IMPORTANT FOR INDUSTRY)
# ============================================

importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(8, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# ============================================
# 12. PREDICT NEW DATA (REAL USE CASE)
# ============================================

# Example: Predict energy for a new situation
new_data = pd.DataFrame({
    'temperature': [35],
    'humidity': [70],
    'hour': [14],
    'day': [15],
    'month': [6]
})

prediction = model.predict(new_data)

print("\nPredicted Energy Consumption for new data:", prediction[0])