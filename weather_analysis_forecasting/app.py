import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Analytics & Forecasting",
    page_icon="🌤️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

  df = pd.read_csv("weather_analytics_dataset.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


df = load_data()


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "weather_random_forest_model.pkl"
    )

    return model


model = load_model()


# ============================================================
# DATA PREPARATION
# ============================================================

df["Temperature_7Day_MA"] = (
    df["Temperature_C"]
    .rolling(window=7)
    .mean()
)

df["Month"] = df["Date"].dt.month

df["Month_Name"] = (
    df["Date"].dt.strftime("%B")
)


month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


# ============================================================
# TITLE
# ============================================================

st.title("🌤️ Weather Analytics & Forecasting")

st.markdown(
    """
    **Historical Weather Analysis + Machine Learning Forecasting**

    This dashboard analyzes historical temperature, rainfall,
    humidity, wind speed and weather conditions and uses a
    Random Forest model to forecast temperature.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Dashboard")

st.sidebar.info(
    """
    **Project Features**

    • Historical weather analysis

    • Statistical analysis

    • Monthly trends

    • 7-day moving average

    • Random Forest forecasting

    • Model evaluation
    """
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

avg_temperature = (
    df["Temperature_C"].mean()
)

max_temperature = (
    df["Temperature_C"].max()
)

min_temperature = (
    df["Temperature_C"].min()
)

total_rainfall = (
    df["Rainfall_mm"].sum()
)

avg_humidity = (
    df["Humidity_%"].mean()
)

avg_wind_speed = (
    df["Wind_Speed_kmh"].mean()
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Weather Overview")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "🌡️ Avg Temperature",
    f"{avg_temperature:.2f} °C"
)

col2.metric(
    "🔥 Maximum",
    f"{max_temperature:.2f} °C"
)

col3.metric(
    "❄️ Minimum",
    f"{min_temperature:.2f} °C"
)

col4.metric(
    "🌧️ Total Rainfall",
    f"{total_rainfall:.2f} mm"
)

col5.metric(
    "💧 Avg Humidity",
    f"{avg_humidity:.2f}%"
)

col6.metric(
    "💨 Avg Wind",
    f"{avg_wind_speed:.2f} km/h"
)


st.divider()


# ============================================================
# TEMPERATURE TREND
# ============================================================

st.subheader("📈 Temperature Trend")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df["Date"],
    df["Temperature_C"],
    label="Daily Temperature",
    alpha=0.5
)

ax.plot(
    df["Date"],
    df["Temperature_7Day_MA"],
    label="7-Day Moving Average",
    linewidth=2
)

ax.set_title(
    "Daily Temperature and 7-Day Moving Average"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Temperature (°C)"
)

ax.legend()

ax.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# RAINFALL + HUMIDITY
# ============================================================

col1, col2 = st.columns(2)


# ---------------- RAINFALL ----------------

with col1:

    st.subheader("🌧️ Rainfall Trend")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["Date"],
        df["Rainfall_mm"]
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Rainfall (mm)"
    )

    ax.set_title(
        "Daily Rainfall"
    )

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)


# ---------------- HUMIDITY ----------------

with col2:

    st.subheader("💧 Humidity Trend")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["Date"],
        df["Humidity_%"]
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Humidity (%)"
    )

    ax.set_title(
        "Daily Humidity"
    )

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# WIND SPEED
# ============================================================

st.subheader("💨 Wind Speed Trend")

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    df["Date"],
    df["Wind_Speed_kmh"]
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Wind Speed (km/h)"
)

ax.set_title(
    "Daily Wind Speed"
)

ax.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

st.subheader("📅 Monthly Weather Analysis")

monthly_data = (
    df.groupby("Month_Name")
    .agg({
        "Temperature_C": "mean",
        "Rainfall_mm": "sum",
        "Humidity_%": "mean",
        "Wind_Speed_kmh": "mean"
    })
    .reindex(month_order)
)


monthly_data.columns = [
    "Average Temperature (°C)",
    "Total Rainfall (mm)",
    "Average Humidity (%)",
    "Average Wind Speed (km/h)"
]


st.dataframe(
    monthly_data.round(2),
    use_container_width=True
)


# ============================================================
# MONTHLY TEMPERATURE
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "🌡️ Monthly Temperature"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        monthly_data.index,
        monthly_data[
            "Average Temperature (°C)"
        ]
    )

    ax.set_xlabel("Month")

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(axis="y")

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# MONTHLY RAINFALL
# ============================================================

with col2:

    st.subheader(
        "🌧️ Monthly Rainfall"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        monthly_data.index,
        monthly_data[
            "Total Rainfall (mm)"
        ]
    )

    ax.set_xlabel("Month")

    ax.set_ylabel(
        "Rainfall (mm)"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(axis="y")

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# WEATHER TYPE DISTRIBUTION
# ============================================================

st.subheader(
    "☁️ Weather Type Distribution"
)

weather_counts = (
    df["Weather_Type"]
    .value_counts()
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.pie(
    weather_counts.values,
    labels=weather_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title(
    "Weather Condition Distribution"
)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# RANDOM FOREST MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader(
    "🤖 Machine Learning Model"
)

st.markdown(
    """
    ### Random Forest Regression

    The model uses:

    • Humidity

    • Rainfall

    • Wind Speed

    • Atmospheric Pressure

    • Rain Probability

    • Seasonal features

    • Previous temperature values
    """
)


# ============================================================
# MODEL PERFORMANCE VALUES
# ============================================================

model_mae = 1.399721

model_rmse = 1.730093

model_r2 = 0.277075


col1, col2, col3 = st.columns(3)


col1.metric(
    "MAE",
    f"{model_mae:.2f} °C"
)

col2.metric(
    "RMSE",
    f"{model_rmse:.2f} °C"
)

col3.metric(
    "R² Score",
    f"{model_r2:.2f}"
)


st.info(
    "Random Forest performed better than Linear Regression "
    "on the test set."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Feature Importance"
)


features = [
    "Humidity_%",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Pressure_hPa",
    "Rain_Probability_%",
    "Day_sin",
    "Day_cos",
    "Temp_Lag_1",
    "Temp_Lag_2",
    "Temp_Lag_3",
    "Temp_Lag_7"
]


feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=True
    )
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

ax.set_xlabel(
    "Importance"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Random Forest Feature Importance"
)

ax.grid(axis="x")

st.pyplot(fig)

plt.close(fig)


# ============================================================
# ML FORECAST FUNCTION
# ============================================================

def generate_ml_forecast(
    data,
    trained_model,
    days=7
):

    history = data.copy()

    temperature_history = list(
        history["Temperature_C"].values
    )

    last_date = (
        history["Date"].max()
    )

    # Latest weather values
    last_humidity = (
        history["Humidity_%"].iloc[-1]
    )

    last_rainfall = (
        history["Rainfall_mm"].iloc[-1]
    )

    last_wind = (
        history["Wind_Speed_kmh"].iloc[-1]
    )

    last_pressure = (
        history["Pressure_hPa"].iloc[-1]
    )

    last_rain_probability = (
        history["Rain_Probability_%"].iloc[-1]
    )

    future_dates = pd.date_range(
        start=last_date +
        pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    predictions = []

    for future_date in future_dates:

        day_of_year = (
            future_date.dayofyear
        )

        # Seasonal features
        day_sin = np.sin(
            2 * np.pi *
            day_of_year / 365
        )

        day_cos = np.cos(
            2 * np.pi *
            day_of_year / 365
        )

        # Lag features
        temp_lag_1 = (
            temperature_history[-1]
        )

        temp_lag_2 = (
            temperature_history[-2]
        )

        temp_lag_3 = (
            temperature_history[-3]
        )

        temp_lag_7 = (
            temperature_history[-7]
        )

        future_input = pd.DataFrame({

            "Humidity_%": [
                last_humidity
            ],

            "Rainfall_mm": [
                last_rainfall
            ],

            "Wind_Speed_kmh": [
                last_wind
            ],

            "Pressure_hPa": [
                last_pressure
            ],

            "Rain_Probability_%": [
                last_rain_probability
            ],

            "Day_sin": [
                day_sin
            ],

            "Day_cos": [
                day_cos
            ],

            "Temp_Lag_1": [
                temp_lag_1
            ],

            "Temp_Lag_2": [
                temp_lag_2
            ],

            "Temp_Lag_3": [
                temp_lag_3
            ],

            "Temp_Lag_7": [
                temp_lag_7
            ]
        })

        prediction = (
            trained_model
            .predict(future_input)[0]
        )

        predictions.append(
            prediction
        )

        # Add prediction to history
        temperature_history.append(
            prediction
        )

    forecast_df = pd.DataFrame({

        "Date": future_dates,

        "Predicted Temperature (°C)":
            predictions

    })

    return forecast_df


# ============================================================
# GENERATE FORECAST
# ============================================================

st.subheader(
    "🔮 Random Forest 7-Day Forecast"
)


ml_forecast = generate_ml_forecast(
    df,
    model,
    days=7
)


st.dataframe(
    ml_forecast.round(2),
    use_container_width=True
)


# ============================================================
# FORECAST GRAPH
# ============================================================

st.subheader(
    "📈 Historical Temperature + ML Forecast"
)


recent_data = df.tail(30)


fig, ax = plt.subplots(
    figsize=(14, 6)
)


ax.plot(
    recent_data["Date"],
    recent_data["Temperature_C"],
    label="Historical Temperature",
    linewidth=2
)


ax.plot(
    ml_forecast["Date"],
    ml_forecast[
        "Predicted Temperature (°C)"
    ],
    label="Random Forest Forecast",
    marker="o",
    linewidth=2
)


ax.set_title(
    "Random Forest 7-Day Temperature Forecast"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Temperature (°C)"
)

ax.legend()

ax.grid(True)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# DOWNLOAD FORECAST
# ============================================================

forecast_csv = (
    ml_forecast
    .to_csv(index=False)
)


st.download_button(
    label="📥 Download ML Forecast",
    data=forecast_csv,
    file_name=
        "random_forest_weather_forecast.csv",
    mime="text/csv"
)


# ============================================================
# EXTREME WEATHER INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🌡️ Weather Records"
)


hottest_day = df.loc[
    df["Temperature_C"].idxmax()
]

coldest_day = df.loc[
    df["Temperature_C"].idxmin()
]

rainiest_day = df.loc[
    df["Rainfall_mm"].idxmax()
]


col1, col2, col3 = st.columns(3)


col1.metric(
    "🔥 Hottest Day",
    hottest_day["Date"].strftime(
        "%d %B %Y"
    ),
    f"{hottest_day['Temperature_C']:.1f} °C"
)


col2.metric(
    "❄️ Coldest Day",
    coldest_day["Date"].strftime(
        "%d %B %Y"
    ),
    f"{coldest_day['Temperature_C']:.1f} °C"
)


col3.metric(
    "🌧️ Rainiest Day",
    rainiest_day["Date"].strftime(
        "%d %B %Y"
    ),
    f"{rainiest_day['Rainfall_mm']:.1f} mm"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Weather Analytics & Forecasting | "
    "Python • Pandas • NumPy • Scikit-learn • "
    "Random Forest • Streamlit"
)
