# House Pricing EDA (Exploratory Data Analysis) - Python

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
sns.set(style="whitegrid")

# Load dataset (Update column names as needed)
df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\Karthik Prj\\eda\\house_price_data.csv')

# Display basic dataset info
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Get basic statistics of numeric columns
print("\nBasic statistical summary of the dataset:")
print(df.describe())

# Identify categorical columns
categorical_features = df.select_dtypes(include=['object']).columns
print("\nCategorical features:", list(categorical_features))

# Handle missing values
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
for col in categorical_features:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical columns to numeric using Label Encoding
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Check for correlation between numeric features
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Visualize the distribution of the target variable 'Price (USD)'
plt.figure(figsize=(8, 6))
sns.histplot(df['Price (USD)'], kde=True, bins=30)
plt.title('Price (USD) Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()

# Train-test split
X = df.drop(columns=['Price (USD)', 'House ID'])  # Features (removing ID)
y = df['Price (USD)']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display train-test split info
print(f"\nTraining Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

# Train a simple Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance Evaluation:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")
