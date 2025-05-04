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

# -----------------------------
# Core functions
# -----------------------------

@st.cache_data
def fetch_stock_data(ticker):
    df = yf.download(ticker, period='1y')
    return df[['Close']]

def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i - time_step:i])
        y.append(data_scaled[i])

    return np.array(X), np.array(y), scaler

def split_data(X, y, split_ratio=0.8):
    split = int(len(X) * split_ratio)
    return X[:split], X[split:], y[:split], y[split:]

def build_model(model_type, input_shape, units=50, dropout_rate=0.2, dense_units=50):
    model = Sequential()

    if model_type == "RNN":
        model.add(SimpleRNN(units, input_shape=input_shape))
    elif model_type == "LSTM":
        model.add(LSTM(units, input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(units, input_shape=input_shape))
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

def predict_future(model, last_sequence, scaler, days=7):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def run_model_pipeline(model_name, df):
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model(model_name, (X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()

    last_sequence = X[-1]
    future = predict_future(model, last_sequence, scaler)

    return {"MSE": mean_squared_error(y_test_inv, y_pred_inv),
            "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
            "Future 7 Days": future,
            "Actual": y_test_inv, "Pred": y_pred_inv}

# Helper function to get next business days (skip Sat-Sun)
def get_next_business_days(start_date, n_days):
    days = []
    current = start_date
    while len(days) < n_days:
        current += pd.Timedelta(days=1)
        if current.weekday() < 5:  # Monday–Friday
            days.append(current)
    return pd.to_datetime(days)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📈 Smart Stock Price Predictor (Weekends Skipped)")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL")

if st.button("Predict"):
    df = fetch_stock_data(ticker)

    if df.empty:
        st.error("Invalid ticker or no data found.")
    else:
        st.success(f"Fetched data for {ticker}")

        models = ["RNN", "LSTM", "GRU", "1D-CNN"]
        results = {}

        with st.spinner("Training models..."):
            for model_name in models:
                results[model_name] = run_model_pipeline(model_name, df)

        best_model = min(results, key=lambda m: results[m]["MSE"])
        st.subheader(f"✅ Best Model: {best_model} (MSE: {results[best_model]['MSE']:.4f})")

        # Show actual vs predicted
        st.subheader("📊 Actual vs Predicted Prices (Including Future Prediction)")

        # Combine actual and predicted values
        all_actual = np.concatenate([np.array(df['Close'].values[-60:]), results[best_model]["Future 7 Days"]])
        all_predicted = np.concatenate([results[best_model]["Pred"], results[best_model]["Future 7 Days"]])

        # Generate the dates for actual and predicted data
        all_dates = pd.to_datetime(df.index[-60:].append(get_next_business_days(df.index[-1], len(results[best_model]["Future 7 Days"]))))
        
        df_actual_predicted = pd.DataFrame({
            'Actual': all_actual,
            'Predicted': all_predicted
        }, index=all_dates)

        st.line_chart(df_actual_predicted)

        # Future prediction (weekends skipped)
        st.subheader("📅 Future 7-Day Stock Price Prediction (Weekends Skipped)")
        future_days = results[best_model]["Future 7 Days"]
        future_dates = get_next_business_days(df.index[-1], len(future_days))
        future_df = pd.DataFrame(future_days, index=future_dates, columns=["Predicted Price"])
        st.write(future_df)

        st.bar_chart(future_df)
