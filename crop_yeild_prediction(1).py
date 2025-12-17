# Crop Yield Prediction using Machine Learning
# Author: Gouthami
# Description: End-to-end ML pipeline for crop yield prediction
import matplotlib
matplotlib.use('TkAgg')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# 1. Create Dataset
# -----------------------------
np.random.seed(42)

rows = 6000

df = pd.DataFrame({
    "Rainfall": np.random.randint(400, 1500, rows),
    "Temperature": np.random.randint(20, 38, rows),
    "Humidity": np.random.randint(40, 85, rows),
    "Soil_pH": np.random.uniform(6.0, 7.8, rows),
    "Area": np.random.uniform(1.0, 7.0, rows)
})

df["Yield"] = (
    0.015 * df["Rainfall"] +
    0.9 * df["Temperature"] +
    0.7 * df["Humidity"] +
    2.5 * df["Area"] -
    1.0 * abs(7 - df["Soil_pH"]) +
    np.random.normal(0, 3, rows)
)

# Save dataset
df.to_csv("crop_yield_dataset.csv", index=False)
print("Dataset created and saved as crop_yield_dataset.csv")


# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nDataset Statistics:")
print(df.describe())


# -----------------------------
# 3. Outlier Detection
# -----------------------------
plt.figure(figsize=(10,5))
df.boxplot()
plt.xticks(rotation=45)
plt.title("Outlier Detection Using Boxplot")
plt.show()


# -----------------------------
# 4. Histograms
# -----------------------------
features = ['Yield', 'Rainfall', 'Temperature']

for feature in features:
    plt.figure(figsize=(7,4))
    plt.hist(df[feature], bins=40, rwidth=0.8)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"{feature} Distribution")
    plt.show()


# -----------------------------
# 5. Scatter Plots
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['Rainfall'], df['Yield'])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Crop Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['Area'], df['Yield'])
plt.xlabel("Area")
plt.ylabel("Yield")
plt.title("Area vs Crop Yield")
plt.show()


# -----------------------------
# 6. Correlation Matrix & Heatmap
# -----------------------------
corr = df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()


# -----------------------------
# 7. Machine Learning Model
# -----------------------------
X = df[['Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'Area']]
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# -----------------------------
# 8. Accuracy & Metrics
# -----------------------------
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("R² Accuracy (%):", round(accuracy, 2))
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))


# -----------------------------
# 9. Actual vs Predicted Plot
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.show()


# -----------------------------
# 10. Sample Prediction
# -----------------------------
sample_df = pd.DataFrame(
    [[900, 32, 65, 7.0, 4.0]],
    columns=['Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'Area']
)

prediction = model.predict(sample_df)
print("\nPredicted Crop Yield for Sample Input:", round(prediction[0], 2))
