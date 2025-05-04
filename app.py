import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from datetime import timedelta

st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction Using RNN, LSTM, GRU, and 1D-CNN")

@st.cache_data
def load_data():
    df = pd.read_csv("stock_data.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['Close']])
    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == "RNN":
        model.add(SimpleRNN(50, return_sequences=False, input_shape=input_shape))
    elif model_type == "LSTM":
        model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(50, return_sequences=False, input_shape=input_shape))
    elif model_type == "1D-CNN":
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type):
    model = build_model(model_type, (X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    return model, predictions

def forecast_next_7_days(model, last_sequence, scaler, model_type):
    future_predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(7):
        pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        future_predictions.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], [[pred[0, 0]]], axis=0)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

def get_next_7_weekdays(start_date):
    next_days = []
    current_day = start_date + timedelta(days=1)
    while len(next_days) < 7:
        if current_day.weekday() < 5:  # Monday=0, Sunday=6
            next_days.append(current_day)
        current_day += timedelta(days=1)
    return next_days

df = load_data()
st.write("### 📅 Raw Stock Data", df.tail())

X, y, scaler = preprocess_data(df)
X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

results = {}

for model_type in ["RNN", "LSTM", "GRU", "1D-CNN"]:
    model, y_pred_scaled = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).flatten()

    last_sequence = X[-1]
    future_preds = forecast_next_7_days(model, last_sequence, scaler, model_type)

    results[model_type] = {
        "actual": y_test_inv,
        "predictions": y_pred_inv,
        "Future 7 Days": future_preds
    }

best_model = st.selectbox("🔍 Select Model to Visualize", list(results.keys()))

# 📈 Combined Plot: Actual, Predicted, and Future Forecast
st.subheader("📈 Combined Plot: Actual, Predicted & Future 7-Day Forecast")

y_test = results[best_model]["actual"]
y_pred = results[best_model]["predictions"]
future_preds = results[best_model]["Future 7 Days"]
start_date = df.index[-1].date()
future_dates = get_next_7_weekdays(start_date)

# Dates
test_dates = pd.date_range(end=start_date, periods=len(y_test)).tolist()
full_dates = test_dates + future_dates
combined_preds = list(y_pred) + list(future_preds)

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test, label="Actual (Test)", color="blue")
plt.plot(full_dates, combined_preds, label="Predicted", color="orange", linestyle='--', marker='o')
plt.axvline(x=test_dates[-1], color='gray', linestyle=':', label="Forecast Start")
plt.title(f"{best_model} - Combined Prediction (Test + Next 7 Days)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.gcf().autofmt_xdate()
st.pyplot(plt)

# 📋 Show Forecast Table
st.subheader("📋 Predicted Prices for Next 7 Business Days")
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_preds
})
future_df["Date"] = future_df["Date"].dt.strftime('%Y-%m-%d')
st.dataframe(future_df)
