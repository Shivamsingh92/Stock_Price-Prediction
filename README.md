# 📈 Stock Price Prediction using Deep Learning (RNN, LSTM, GRU, 1D-CNN)
App_link- https://stockprice-prediction-pejixxvw6iuqkt75a4e2ad.streamlit.app/

## 🧠 Problem Statement

Stock prices are inherently volatile and influenced by various market forces. Predicting future stock prices accurately is a complex time-series problem. This project aims to forecast the next 7 days of a stock's closing prices using past 1-year historical data.

## 🎯 Objective

- Build and compare four deep learning models: RNN, LSTM, GRU, and 1D-CNN.
- Predict the next 7-day closing prices for a given stock ticker.
- Update the dataset daily using the latest available stock data from Yahoo Finance.
- Visualize predictions with respect to actual data.
- Deploy the project via a Streamlit web app.

---

## 🚶‍♂️ Steps to Approach the Problem

### 1. Data Collection
- Use the `yfinance` library to download 1-year daily stock data for any user-specified ticker.
- Focus on the 'Close' price column.

### 2. Data Preprocessing
- Normalize the 'Close' prices using MinMaxScaler to scale between 0 and 1.
- Create a sliding window dataset (e.g., 60 previous days → 1 future day).
- Split the dataset into training and test sets (e.g., 80:20).

### 3. Model Building
Implement and train the following models:
- 🔁 RNN: Basic Recurrent Neural Network model.
- 🔁 LSTM: Long Short-Term Memory model for capturing long-term dependencies.
- 🔁 GRU: Gated Recurrent Unit model for efficient training with performance similar to LSTM.
- 📏 1D-CNN: Convolutional model applied on time series for trend detection.

### 4. Prediction and Evaluation
- Predict the closing prices for the next 7 days using each trained model.
- Inverse transform the normalized predictions to original scale.
- Calculate evaluation metrics: MSE, RMSE, MAE.

### 5. Visualization
- Plot actual vs predicted values for each model.
- Use matplotlib or seaborn for graphical insights.

### 6. Streamlit Web App
- Build a web app where users can:
  - Enter a stock ticker symbol.
  - Select the model type (RNN, LSTM, GRU, or 1D-CNN).
  - View predictions for the next 7 days along with a chart.

### 7. Automation
- On app load, auto-update data using `yfinance` to ensure predictions are based on the most recent data.

---

## 🛠 Technologies Used

- Python
- yfinance
- NumPy, Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn
- Streamlit

---


