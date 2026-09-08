# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 20-08-2026



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read CSV file
df = pd.read_csv(r"C:\Users\admin\Downloads\bmw (1).csv")

# Given data
print("GIVEN DATA")
print(df.head())

# Select price column
data = df["price"].dropna().reset_index(drop=True)

# Augmented Dickey-Fuller Test
result = adfuller(data)

print("\nAUGMENTED DICKEY-FULLER TEST")
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("The data is stationary")
else:
    print("The data is not stationary")

# Split data
train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

# AR model with 13 lags
model = AutoReg(train, lags=13, trend="c")
model_fit = model.fit()

# PACF and ACF
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plot_pacf(data, lags=20, ax=ax[0])
ax[0].set_title("PACF")

plot_acf(data, lags=20, ax=ax[1])
ax[1].set_title("ACF")

plt.tight_layout()
plt.show()

# Predictions
predictions = model_fit.predict(
    start=len(train),
    end=len(data) - 1
)

print("\nPREDICTION")
print(predictions)

# MSE
mse = np.mean((test.values - predictions) ** 2)

print("\nFINAL PREDICTION")
print("Mean Squared Error:", mse)

# Plot actual vs predicted
plt.figure(figsize=(12, 5))

plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, predictions, label="Predicted")

plt.title("AR(13) - Actual vs Predicted")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()
plt.show()
```
### OUTPUT:

GIVEN DATA

<img width="642" height="329" alt="image" src="https://github.com/user-attachments/assets/10735646-421d-4a93-80f8-81eff413f052" />


PACF - ACF

<img width="1000" height="661" alt="image" src="https://github.com/user-attachments/assets/e419e624-1472-441f-ba15-f1f9895022e3" />



PREDICTION

<img width="436" height="265" alt="image" src="https://github.com/user-attachments/assets/3766ffcf-5387-4801-ba93-46a14bd06174" />


FINIAL PREDICTION

<img width="1031" height="501" alt="image" src="https://github.com/user-attachments/assets/2b740a84-9d73-4bda-bec8-30b6ff56f34d" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
