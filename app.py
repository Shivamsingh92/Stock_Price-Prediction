import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_option('deprecation.showPyplotGlobalUse', False)

# Fetch stock data
def fetch_stock_data(ticker):
    df = yf.download(ticker, period='1y')
    return df[['Close']]

# Preprocess data
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i - time_step:i])
        y.append(data_scaled[i])

    return np.array(X), np.array(y), scaler

# Split data
def split_data(X, y, split_ratio=0.8):
    split = int(len(X) * split_ratio)
    return X[:split], X[split:], y[:split], y[split:]

# Model builder
def build_model(model_type, input_shape, units=50, dropout_rate=0.2, dense_units=50):
    model = Sequential()
    if model_type == "RNN":
        model.add(SimpleRNN(units, return_sequences=False, input_shape=input_shape))
    elif model_type == "LSTM":
        model.add(LSTM(units, return_sequences=False, input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(units, return_sequences=False, input_shape=input_shape))
    elif model_type == "1D-CNN":
        model.add(Conv1D(filters=units, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
    if model_type in ["RNN", "LSTM", "GRU"]:
        model.add(Dropout(dropout_rate))

    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Predict future with weekend skip
def get_next_trading_days(start_date, days):
    trading_days = []
    current = start_date
    while len(trading_days) < days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday to Friday only
            trading_days.append(current)
    return trading_days

def predict_future(model, last_sequence, scaler, days=7):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Run pipeline
def run_model_pipeline(model_name, df):
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model(model_name, (X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    last_sequence = X[-1]
    future_preds = predict_future(model, last_sequence, scaler)

    return {
        "MSE": mse,
        "MAE": mae,
        "Future": future_preds,
        "Model": model,
        "Scaler": scaler,
        "Actual": y_test_inv,
        "Predicted": y_pred_inv,
        "Last Sequence": last_sequence
    }

# Streamlit UI
st.title("📈 Stock Price Prediction App")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")

if st.button("Predict"):
    df = fetch_stock_data(ticker)
    results = {}
    for model_name in ["RNN", "LSTM", "GRU", "1D-CNN"]:
        with st.spinner(f"Training {model_name} model..."):
            results[model_name] = run_model_pipeline(model_name, df)

    # Select best model based on MSE
    best_model_name = min(results, key=lambda name: results[name]["MSE"])
    st.success(f"✅ Best model selected: {best_model_name}")

    result = results[best_model_name]

    # Plot actual vs predicted
    st.subheader("📉 Actual vs Predicted Plot")
    plt.figure(figsize=(10, 5))
    plt.plot(result["Actual"], label="Actual")
    plt.plot(result["Predicted"], label="Predicted")
    plt.legend()
    st.pyplot()

    # Plot future prediction
    st.subheader("📊 Future 7 Trading Days Prediction")

    last_date = df.index[-1].to_pydatetime()
    future_dates = get_next_trading_days(last_date, 7)

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": result["Future"]
    })

    st.dataframe(future_df.set_index("Date"))

    plt.figure(figsize=(10, 5))
    plt.plot(future_df["Date"], future_df["Predicted Price"], marker='o', linestyle='--', color='green')
    plt.title(f"{ticker} - Future 7 Trading Days Forecast using {best_model_name}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    st.pyplot()
