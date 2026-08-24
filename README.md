# Weather-Analytics-Forecasting
Weather Analytics and Temperature Forecasting using Python and Random Forest
# 🌤️ Weather Analytics & Forecasting

A data analytics and machine learning project that analyzes historical weather data and predicts future temperature using Random Forest Regression.

## 📌 Project Overview

This project analyzes historical weather parameters including:

- Temperature
- Humidity
- Rainfall
- Wind Speed
- Atmospheric Pressure
- Rain Probability
- Weather Type

The project also provides a 7-day temperature forecast using a Random Forest machine learning model.

## 🎯 Objectives

- Analyze historical weather patterns
- Calculate statistical measures
- Visualize temperature and rainfall trends
- Perform monthly weather analysis
- Implement a 7-day moving average
- Build machine learning models
- Compare Linear Regression and Random Forest
- Forecast future temperature
- Create an interactive Streamlit dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Google Colab

## 🤖 Machine Learning

Two models were tested:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.46°C | 1.78°C | 0.24 |
| Random Forest | 1.40°C | 1.73°C | 0.28 |

Random Forest was selected as the final model because it achieved the best performance.

## 📊 Dataset

The dataset contains:

- 365 records
- 8 columns
- No missing values

## 🔮 Forecasting

The Random Forest model uses:

- Humidity
- Rainfall
- Wind Speed
- Pressure
- Rain Probability
- Seasonal features
- Temperature lag features

The application generates a 7-day temperature forecast.

## 🚀 How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
