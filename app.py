import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Load the model
model = load_model(r"C:\Users\patel\Music\project\ML Model\Stock Prediction Model.keras")
st.set_page_config(page_title="Stock Market Predictor", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Stock Market Predictor</h1>", unsafe_allow_html=True)

st.sidebar.header("Navigation")
selected_option = st.sidebar.selectbox("Choose a section:", ["Stock Data", "Moving Averages", "Predictions", "Next Day Prediction"])

# Load stock symbols from CSV
try:
    stocks_df = pd.read_csv("Stock_Symbols.csv")
    stock_symbols = stocks_df['Symbol'].dropna().astype(str).tolist()
except FileNotFoundError:
    st.error("Stock symbols CSV file not found.")
    stock_symbols = []

# Dropdown to select stock
stock = st.sidebar.selectbox('Select a stock', options=stock_symbols)

#start and end dates
start = '2012-01-01'
end = '2024-10-30'

# Fetch stock data
try:
    data = yf.download(stock, start=start, end=end)
    if data.empty:
        st.error(f"No data available for {stock}. Please select another stock.")
except Exception as e:
    st.error(f"Failed to retrieve data for {stock}: {e}")

# Display stock data
if selected_option == "Stock Data":
    st.subheader(f'Stock Data for {stock}')
    st.markdown("<div style='display: flex; justify-content: center; margin-top: 20px;'>", unsafe_allow_html=True)
    st.write(data)
    st.markdown("</div>", unsafe_allow_html=True)

# Moving Averages section
if selected_option == "Moving Averages":
    ma_option = st.sidebar.selectbox("Select Moving Average", ["MA50", "MA100", "MA200"])

    # Calculate moving averages
    ma_50_days = data['Close'].rolling(50).mean()
    ma_100_days = data['Close'].rolling(100).mean()
    ma_200_days = data['Close'].rolling(200).mean()

    # Plot based on selected moving average
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price', line=dict(color='green')))

    if ma_option == "MA50":
        fig.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='firebrick')))
    elif ma_option == "MA100":
        fig.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='royalblue')))
    elif ma_option == "MA200":
        fig.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='purple')))

    fig.update_layout(title=f"Price and {ma_option} for {stock}", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Preparing data for training and testing
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

scale = 1 / scaler.scale_

# Preparing input data for prediction globally
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])
x, y = np.array(x), np.array(y)
y = y * scale

# Prediction section
if selected_option == "Predictions":
    predict = model.predict(x)
    predict = predict * scale

    # Interactive plot for Original vs Predicted Prices
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y.flatten(), mode='lines', name='Actual Price', line=dict(color='green')))
    fig_pred.add_trace(go.Scatter(y=predict.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))

    fig_pred.update_layout(title="Original Price vs Predicted Price", xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Display the last ten days of actual vs predicted prices in a table format
    st.subheader('Actual vs Predicted Prices (Last 10 Days)')
    st.markdown("<div style='display: flex; justify-content: center; margin-top: 20px;'>", unsafe_allow_html=True)
    # results_df = pd.DataFrame({'Actual Price': y.flatten(), 'Predicted Price': predict.flatten()}).tail(10)
    results_df = pd.DataFrame({'Actual Price': y.flatten(), 'Predicted Price': predict.flatten()}).tail(10)
    results_df.index = range(1, len(results_df) + 1)

    st.table(results_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # mape = mean_absolute_percentage_error(y, predict)
    # accuracy = 100 - mape * 100
    # st.write(f"Prediction Accuracy: {accuracy:.2f}%")

# Next Day Prediction section
if selected_option == "Next Day Prediction":
    last_100_days_scaled = data_test_scale[-100:]
    last_100_days_scaled = np.expand_dims(last_100_days_scaled, axis=0)
    next_day_prediction_scaled = model.predict(last_100_days_scaled)
    next_day_prediction = next_day_prediction_scaled * scale

    # st.subheader('Predicted Price for Next Day')
    st.markdown(
        f"<div style='text-align: center; margin-top: 20px; font-size: 24px;'><strong>The predicted stock price for the next day is: ${next_day_prediction[0][0]:.2f}</strong></div>",
        unsafe_allow_html=True
    )
