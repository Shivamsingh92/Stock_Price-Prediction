import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp { background-color: #0d0f14; color: #e8eaf0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2230;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 16px;
}

/* Headline */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #00e5a0;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6c7a99;
    margin-bottom: 2rem;
}

/* Section header */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00e5a0;
    margin-bottom: 0.5rem;
}

/* Tag badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
    margin-top: 4px;
}
.badge-green  { background: #00e5a020; color: #00e5a0; border: 1px solid #00e5a040; }
.badge-yellow { background: #f5c51820; color: #f5c518; border: 1px solid #f5c51840; }
.badge-red    { background: #ff4d4d20; color: #ff6b6b; border: 1px solid #ff4d4d40; }
.badge-blue   { background: #4d9fff20; color: #7bb8ff; border: 1px solid #4d9fff40; }

/* Divider */
.thin-divider { border-top: 1px solid #1e2230; margin: 1.5rem 0; }

/* Future price row */
.future-card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.future-day  { font-family: 'Space Mono', monospace; color: #6c7a99; font-size: 0.82rem; }
.future-price { font-family: 'Space Mono', monospace; color: #00e5a0; font-size: 1.1rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Models ─────────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, dropout=0.2, dense_units=50):
        super().__init__()
        self.rnn      = nn.RNN(input_size, hidden_size, batch_first=True)
        self.drop1    = nn.Dropout(dropout)
        self.fc1      = nn.Linear(hidden_size, dense_units)
        self.relu     = nn.ReLU()
        self.drop2    = nn.Dropout(dropout)
        self.fc2      = nn.Linear(dense_units, 1)
    def forward(self, x):
        _, h = self.rnn(x)
        out = self.drop1(h.squeeze(0))
        return self.fc2(self.drop2(self.relu(self.fc1(out))))

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, dropout=0.2, dense_units=50):
        super().__init__()
        self.lstm  = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden_size, dense_units)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dense_units, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.drop1(h.squeeze(0))
        return self.fc2(self.drop2(self.relu(self.fc1(out))))

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, dropout=0.2, dense_units=50):
        super().__init__()
        self.gru   = nn.GRU(input_size, hidden_size, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden_size, dense_units)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dense_units, 1)
    def forward(self, x):
        _, h = self.gru(x)
        out = self.drop1(h.squeeze(0))
        return self.fc2(self.drop2(self.relu(self.fc1(out))))

class CNN1DModel(nn.Module):
    def __init__(self, time_step=60, input_size=1, filters=50, kernel_size=3,
                 dropout=0.2, dense_units=50):
        super().__init__()
        self.conv   = nn.Conv1d(input_size, filters, kernel_size)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool1d(2)
        conv_len    = (time_step - kernel_size + 1) // 2
        self.fc1    = nn.Linear(filters * conv_len, dense_units)
        self.drop   = nn.Dropout(dropout)
        self.fc2    = nn.Linear(dense_units, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv(x))).flatten(1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))

MODEL_REGISTRY = {"RNN": RNNModel, "LSTM": LSTMModel, "GRU": GRUModel, "1D-CNN": CNN1DModel}

# ── Pipeline helpers ────────────────────────────────────────────────────────
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_step, len(scaled)):
        X.append(scaled[i - time_step:i])
        y.append(scaled[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

def split_data(X, y, ratio=0.8):
    s = int(len(X) * ratio)
    return X[:s], X[s:], y[:s], y[s:]

def build_model(name, time_step, units, dropout, dense_units):
    cls = MODEL_REGISTRY[name]
    m = cls(time_step=time_step, filters=units, dropout=dropout, dense_units=dense_units) \
        if name == "1D-CNN" else \
        cls(hidden_size=units, dropout=dropout, dense_units=dense_units)
    return m.to(DEVICE)

def train_model(model, X_train, y_train, epochs, batch_size, lr, progress_bar, status_text, label):
    X_t = torch.from_numpy(X_train).to(DEVICE)
    y_t = torch.from_numpy(y_train).to(DEVICE)
    loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        progress_bar.progress(epoch / epochs)
        status_text.text(f"{label} — epoch {epoch}/{epochs}")

@torch.no_grad()
def predict(model, X_np):
    model.eval()
    return model(torch.from_numpy(X_np).to(DEVICE)).cpu().numpy()

@torch.no_grad()
def predict_future(model, last_seq, scaler, days=7):
    model.eval()
    seq, preds = last_seq.copy(), []
    for _ in range(days):
        x = torch.from_numpy(seq[np.newaxis]).to(DEVICE)
        p = model(x).item()
        preds.append(p)
        seq = np.append(seq[1:], [[p]], axis=0)
    return scaler.inverse_transform(
        np.array(preds, dtype=np.float32).reshape(-1, 1)
    ).flatten()

# ── Plot helpers ────────────────────────────────────────────────────────────
def make_pred_figure(model_name, actual, predicted):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#0d0f14")
    ax.plot(actual,    color="#4d9fff", linewidth=1.5, label="Actual")
    ax.plot(predicted, color="#00e5a0", linewidth=1.5, linestyle="--", label="Predicted")
    ax.set_title(f"{model_name}", color="#e8eaf0", fontsize=13, pad=10)
    ax.set_xlabel("Time", color="#6c7a99")
    ax.set_ylabel("Price ($)", color="#6c7a99")
    ax.tick_params(colors="#6c7a99")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2230")
    ax.legend(facecolor="#13161e", edgecolor="#1e2230", labelcolor="#e8eaf0")
    ax.grid(color="#1e2230", linewidth=0.5)
    fig.tight_layout()
    return fig

def make_bar_figure(results):
    names = list(results.keys())
    mses  = [results[m]["MSE"] for m in names]
    maes  = [results[m]["MAE"] for m in names]
    x     = np.arange(len(names))
    w     = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#0d0f14")
    ax.bar(x - w/2, mses, w, label="MSE", color="#4d9fff", alpha=0.85)
    ax.bar(x + w/2, maes, w, label="MAE", color="#00e5a0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Model Loss Comparison", color="#e8eaf0", fontsize=13)
    ax.tick_params(colors="#6c7a99")
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2230")
    ax.legend(facecolor="#13161e", edgecolor="#1e2230", labelcolor="#e8eaf0")
    ax.grid(axis="y", color="#1e2230", linewidth=0.5)
    fig.tight_layout()
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Configuration</p>', unsafe_allow_html=True)

    ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").upper()

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Models to Run</p>', unsafe_allow_html=True)
    run_rnn  = st.checkbox("RNN",    value=True)
    run_lstm = st.checkbox("LSTM",   value=True)
    run_gru  = st.checkbox("GRU",    value=True)
    run_cnn  = st.checkbox("1D-CNN", value=True)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Hyperparameters</p>', unsafe_allow_html=True)
    epochs      = st.slider("Epochs",      5, 50, 10)
    batch_size  = st.slider("Batch Size",  16, 128, 32, step=16)
    units       = st.slider("Hidden Units", 32, 256, 50, step=16)
    dropout     = st.slider("Dropout",     0.0, 0.5, 0.2, step=0.05)
    time_step   = st.slider("Look-back Window", 20, 120, 60, step=10)
    lr          = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    future_days = st.slider("Future Forecast Days", 3, 30, 7)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

# ── Main area ──────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">📈 Stock Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">PyTorch · RNN / LSTM / GRU / 1D-CNN · 7-day forecast</p>',
            unsafe_allow_html=True)

if not run_btn:
    # Landing state — show a price chart only
    st.markdown('<p class="section-label">Price History Preview</p>', unsafe_allow_html=True)
    try:
        preview = yf.download(ticker, period="6mo", progress=False)[["Close"]]
        if not preview.empty:
            fig, ax = plt.subplots(figsize=(12, 3))
            fig.patch.set_facecolor("#0d0f14"); ax.set_facecolor("#0d0f14")
            ax.plot(preview.index, preview["Close"], color="#4d9fff", linewidth=1.5)
            ax.fill_between(preview.index, preview["Close"].values.flatten(),
                            alpha=0.08, color="#4d9fff")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.tick_params(colors="#6c7a99")
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2230")
            ax.grid(color="#1e2230", linewidth=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception:
        st.info("Enter a valid ticker in the sidebar to preview price history.")

    st.markdown("""
    <div style='margin-top:2rem;color:#6c7a99;font-size:0.9rem;'>
    Configure your models and hyperparameters in the sidebar, then hit
    <strong style='color:#00e5a0'>Run Prediction</strong> to start training.
    </div>
    """, unsafe_allow_html=True)

else:
    selected_models = [m for m, flag in
                       [("RNN", run_rnn), ("LSTM", run_lstm),
                        ("GRU", run_gru), ("1D-CNN", run_cnn)] if flag]

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        st.stop()

    # ── Fetch data ──────────────────────────────────────────────────────────
    with st.spinner(f"Downloading {ticker} data…"):
        try:
            df = yf.download(ticker, period="1y", progress=False)[["Close"]]
            if df.empty:
                st.error(f"No data found for **{ticker}**. Check the ticker symbol.")
                st.stop()
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    st.success(f"Downloaded **{len(df)}** trading days for **{ticker}**")

    # ── Pre-process ─────────────────────────────────────────────────────────
    X, y, scaler = preprocess_data(df.values, time_step)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Train ───────────────────────────────────────────────────────────────
    results = {}
    train_section = st.empty()

    for idx, model_name in enumerate(selected_models):
        with train_section.container():
            st.markdown(f'<p class="section-label">Training {model_name} '
                        f'({idx+1}/{len(selected_models)})</p>',
                        unsafe_allow_html=True)
            prog  = st.progress(0)
            stxt  = st.empty()

        model = build_model(model_name, time_step, units, dropout, dense_units=50)
        train_model(model, X_train, y_train, epochs, batch_size, lr, prog, stxt, model_name)

        y_pred      = predict(model, X_test)
        y_test_inv  = scaler.inverse_transform(y_test).flatten()
        y_pred_inv  = scaler.inverse_transform(y_pred).flatten()
        future_vals = predict_future(model, X[-1], scaler, future_days)

        results[model_name] = {
            "MSE": mean_squared_error(y_test_inv, y_pred_inv),
            "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
            "actual": y_test_inv,
            "predictions": y_pred_inv,
            "future": future_vals,
        }

    train_section.empty()

    # ── Metrics row ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.metric(label=f"{name} — MSE", value=f"{res['MSE']:.2f}")
            st.metric(label=f"{name} — MAE", value=f"{res['MAE']:.2f}")

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    # ── Bar chart ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Loss Comparison</p>', unsafe_allow_html=True)
    fig_bar = make_bar_figure(results)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    # ── Per-model prediction charts ─────────────────────────────────────────
    st.markdown('<p class="section-label">Actual vs Predicted</p>', unsafe_allow_html=True)
    for name, res in results.items():
        fig = make_pred_figure(name, res["actual"], res["predictions"])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    # ── Future forecast ─────────────────────────────────────────────────────
    best = min(results, key=lambda m: results[m]["MSE"])
    st.markdown(f'<p class="section-label">🔮 {future_days}-Day Forecast · Best Model: {best}</p>',
                unsafe_allow_html=True)

    import datetime
    today = datetime.date.today()
    future_prices = results[best]["future"]
    last_actual   = results[best]["actual"][-1]

    cols2 = st.columns([1, 1])
    with cols2[0]:
        for i, price in enumerate(future_prices):
            day = today + datetime.timedelta(days=i + 1)
            delta = price - last_actual
            sign  = "▲" if delta >= 0 else "▼"
            color = "#00e5a0" if delta >= 0 else "#ff6b6b"
            st.markdown(
                f"""<div class="future-card">
                      <span class="future-day">{day.strftime('%a, %b %d')}</span>
                      <span class="future-price">${price:.2f}
                        <span style='font-size:0.75rem;color:{color}'>{sign} {abs(delta):.2f}</span>
                      </span>
                    </div>""",
                unsafe_allow_html=True,
            )

    with cols2[1]:
        fig_f, ax_f = plt.subplots(figsize=(6, 4))
        fig_f.patch.set_facecolor("#0d0f14"); ax_f.set_facecolor("#0d0f14")
        tail = results[best]["actual"][-30:]
        ax_f.plot(range(len(tail)), tail, color="#4d9fff", linewidth=1.5, label="Recent Actual")
        start = len(tail) - 1
        ax_f.plot(range(start, start + future_days + 1),
                  [tail[-1]] + list(future_prices),
                  color="#00e5a0", linewidth=2, linestyle="--", marker="o",
                  markersize=4, label=f"Forecast ({best})")
        ax_f.axvline(start, color="#f5c518", linewidth=1, linestyle=":", alpha=0.7)
        ax_f.tick_params(colors="#6c7a99")
        for sp in ax_f.spines.values(): sp.set_edgecolor("#1e2230")
        ax_f.grid(color="#1e2230", linewidth=0.4)
        ax_f.legend(facecolor="#13161e", edgecolor="#1e2230", labelcolor="#e8eaf0")
        fig_f.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#6c7a99;font-size:0.8rem;'>Device: <code>{DEVICE}</code> &nbsp;|&nbsp; "
        f"Models run: {', '.join(selected_models)} &nbsp;|&nbsp; "
        f"Best by MSE: <strong style='color:#00e5a0'>{best}</strong></p>",
        unsafe_allow_html=True,
    )
