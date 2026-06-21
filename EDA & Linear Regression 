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