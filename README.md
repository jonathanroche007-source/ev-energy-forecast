# Global EV Sales Forecasting (ARIMA)

## Project Overview

This project forecasts global electric vehicle (EV) sales using historical data from the IEA Global EV Dataset (2024).

The objective is to:

- Clean and aggregate historical EV sales data
- Build a time series forecasting model (ARIMA)
- Validate model performance using MAE and MAPE
- Generate a 5-year future forecast

---

## Dataset

Source: IEA Global EV Data 2024  
Filtered for:
- Region: World
- Parameter: EV sales
- Unit: Vehicles
- Category: Historical

---

## Methodology

1. Data filtering and aggregation
2. Time series transformation (year → datetime index)
3. Train/Test split (pre-2022 training)
4. ARIMA(1,1,1) model fitting
5. Forecast validation
6. 5-year future projection

---

## Model Performance

- Mean Absolute Error (MAE): ~1,688,918 vehicles
- Mean Absolute Percentage Error (MAPE): ~13.11%

The model captures strong upward EV growth trends with reasonable forecasting accuracy.

---

## Technologies Used

- Python
- pandas
- matplotlib
- statsmodels
- scikit-learn

---

## Future Improvements

- SARIMA seasonal modeling
- Random Forest regression comparison
- Prophet model comparison
- Hyperparameter optimization
- Confidence interval visualization

---

## How to Run This Project

1. Clone the repository:
   git clone https://github.com/jonathanroche007-source/ev-energy-forecast.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the script:
   python ev_forecast.py


## Author

Jonathan Roche  
Energy & Data Analytics
