import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Split data
def split_data(X, y, split_ratio=0.8):
    split = int(len(X) * split_ratio)
    return X[:split], X[split:], y[:split], y[split:]


# Build model
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


# Predict future 7 weekdays (skip weekends)
def get_next_7_weekdays(start_date):
    dates = []
    while len(dates) < 7:
        start_date += timedelta(days=1)
        if start_date.weekday() < 5:  # 0-4 are Mon-Fri
            dates.append(start_date)
    return dates


def predict_future(model, last_sequence, scaler, days=7):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


# Run full pipeline
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
    future = predict_future(model, last_sequence, scaler)

    return {
        "MSE": mse,
        "MAE": mae,
        "Future 7 Days": future,
        "actual": y_test_inv,
        "predictions": y_pred_inv,
        "model": model,
        "scaler": scaler,
        "last_sequence": last_sequence
    }


# Streamlit App
def main():
    st.title("📈 Stock Price Prediction (Next 7 Days excluding weekends)")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL")

    if st.button("Predict"):
        df = fetch_stock_data(ticker)

        results = {}
        for model_name in ["RNN", "LSTM", "GRU", "1D-CNN"]:
            results[model_name] = run_model_pipeline(model_name, df)

        best_model = min(results, key=lambda model: results[model]["MSE"])
        st.success(f"✅ Best model based on MSE: {best_model}")

        future_preds = results[best_model]["Future 7 Days"]
        start_date = df.index[-1].date()
        future_dates = get_next_7_weekdays(start_date)

        # Convert future dates to datetime for plotting
        future_dates = [datetime.strptime(str(date), '%Y-%m-%d').date() for date in future_dates]

        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})

        st.subheader("📅 Future 7-Day Stock Price Prediction")
        st.dataframe(future_df)

        # Plot actual vs predicted
        st.subheader("📊 Actual vs Predicted (Test Data)")
        y_test = results[best_model]["actual"]
        y_pred = results[best_model]["predictions"]
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label="Actual", color="blue")
        plt.plot(y_pred, label="Predicted", color="red", linestyle='--')
        plt.title(f"{best_model} - Actual vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)

        # Plot future predictions
        st.subheader("📉 Next 7 Business Days Forecast")
        plt.figure(figsize=(10, 4))
        plt.plot(future_dates, future_preds, marker='o', linestyle='-', color='purple')
        plt.title(f"Future 7-Day Forecast ({best_model})")
        plt.xlabel("Date")
        plt.ylabel("Predicted Stock Price")
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gcf().autofmt_xdate()
        st.pyplot(plt)


if __name__ == "__main__":
    main()
