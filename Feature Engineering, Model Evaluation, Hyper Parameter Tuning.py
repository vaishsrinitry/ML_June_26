# House Price Prediction Internship Project
# Submitted By Vaishnavi S

# Week 1 Task - Linear Regression & EDA

# Import Libraries


# For Built in Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For Model and Data Set
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

# Display Dataset
print(df.head())

# EDA (Exploratory Data Analysis)


# Dataset Information
print(df.info())

# Statistical Summary
print(df.describe())

# Check Missing Values
print(df.isnull().sum())

# Histogram of House Prices
plt.hist(df["Price"], bins=20)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot for the given data
plt.scatter(df["MedInc"], df["Price"])
plt.title("Median Income vs House Price")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.show()


# Data Preparation

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully")

# Display Coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)
# End of Task 1 - EDA & Regression Model (Week 1);


# Week 2 Task - Feature Engineering, Model Evaluation & Hyperparameter Tuning

# Import Additional Libraries for Week 2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Feature Engineering


# Fix skewed features using log transformations
X_train['Log_Population'] = np.log1p(X_train['Population'])
X_test['Log_Population'] = np.log1p(X_test['Population'])

X_train['Log_AveRooms'] = np.log1p(X_train['AveRooms'])
X_test['Log_AveRooms'] = np.log1p(X_test['AveRooms'])

# Drop old raw skewed columns to avoid redundancy
X_train_engineered = X_train.drop(["Population", "AveRooms"], axis=1)
X_test_engineered = X_test.drop(["Population", "AveRooms"], axis=1)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_engineered)
X_test_scaled = scaler.transform(X_test_engineered)

print("Feature Engineering and Scaling Completed")


# Hyperparameter Tuning & Model Training


# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Define Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Grid Search Cross Validation (Updated n_jobs to 1 to fix memory error)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=1)
grid_search.fit(X_train_scaled, y_train)

# Select Best Model
best_rf_model = grid_search.best_estimator_

print("\nHyperparameter Tuning Completed")
print("Best Parameters Found:", grid_search.best_params_)


# Model Evaluation


# Predict on Test Data
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Performance Metrics Calculation
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
# Display Evaluation Metrics
print("\nRandom Forest Performance Metrics:", flush=True)
print("Mean Absolute Error (MAE) :", mae_rf, flush=True)
print("Root Mean Squared Error (RMSE):", rmse_rf, flush=True)
print("R-squared (R2 Score)          :", r2_rf, flush=True)


# Residual Plot Visualization


# Calculate Errors
residuals = y_test - y_pred_rf

# Scatter plot of Residuals
plt.scatter(y_pred_rf, residuals, alpha=0.3, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot - Week 2 Random Forest")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Errors)")
plt.show()

# End of Task 2 - Feature Engineering, Evaluation & Tuning (Week 2);