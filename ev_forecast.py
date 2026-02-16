# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:12:33 2026

@author: jonat
"""

# -*- coding: utf-8 -*-
"""
EV Sales Forecasting using ARIMA
Author: Jonathan
Created: Feb 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

file_path = r"C:/Users/jonat/OneDrive/Desktop/EV_Data/IEA Global EV Data 2024.csv"
df = pd.read_csv(file_path)

# Filter relevant data
df_sales = df[
    (df["parameter"] == "EV sales") &
    (df["unit"] == "Vehicles") &
    (df["category"] == "Historical") &
    (df["region"] == "World")
]

# Aggregate annual sales
annual_sales = df_sales.groupby("year")["value"].sum()

# Convert year index to datetime
annual_sales.index = pd.to_datetime(annual_sales.index, format="%Y")

# Set yearly frequency to avoid warnings
annual_sales = annual_sales.asfreq("YS")

sales_series = annual_sales

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

train = sales_series[:'2021']
test = sales_series['2022':]

# -------------------------------------------------
# FIT ARIMA MODEL (VALIDATION)
# -------------------------------------------------

model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

forecast_test = model_fit.forecast(steps=len(test))

# -------------------------------------------------
# EVALUATE MODEL
# -------------------------------------------------

mae = mean_absolute_error(test, forecast_test)
mape = (abs((test - forecast_test) / test)).mean() * 100

print("Mean Absolute Error:", round(mae, 2))
print("MAPE:", round(mape, 2), "%")

# -------------------------------------------------
# PLOT VALIDATION RESULTS
# -------------------------------------------------

plt.figure()
plt.plot(train, label="Train Data")
plt.plot(test, label="Actual Test Data")
plt.plot(test.index, forecast_test, label="Forecast")
plt.legend()
plt.title("ARIMA Model Validation (Global EV Sales)")
plt.xlabel("Year")
plt.ylabel("Vehicles Sold")
plt.show()

# -------------------------------------------------
# FINAL MODEL (FULL DATA)
# -------------------------------------------------

model_full = ARIMA(sales_series, order=(1, 1, 1))
model_full_fit = model_full.fit()

forecast_future = model_full_fit.forecast(steps=5)

future_dates = pd.date_range(
    start=sales_series.index[-1] + pd.DateOffset(years=1),
    periods=5,
    freq="YS"
)

# -------------------------------------------------
# PLOT FINAL FORECAST
# -------------------------------------------------

plt.figure()
plt.plot(sales_series, label="Historical Sales")
plt.plot(future_dates, forecast_future, label="5-Year Forecast")
plt.legend()
plt.title("5-Year Forecast of Global EV Sales")
plt.xlabel("Year")
plt.ylabel("Vehicles Sold")
plt.show()
