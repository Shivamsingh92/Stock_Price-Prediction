import torch
import torch.nn as nn
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, model_type="LSTM"):
        super().__init__()
        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif model_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        self.model_type = model_type

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class CNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.flatten_size = hidden_size * 30  # 60 // 2 = 30
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_model(model_type, input_shape):
    if model_type == "1D-CNN":
        return CNNModel()
    else:
        return RNNModel(model_type=model_type)


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    model.train()
    X = torch.FloatTensor(X_train)
    y = torch.FloatTensor(y_train)
    for _ in range(epochs):
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb.squeeze())
            loss.backward()
            optimizer.step()
    return model


def predict_model(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()


def predict_future(model, last_sequence, scaler, days=7):
    predictions = []
    current_seq = last_sequence.copy()
    model.eval()
    for _ in range(days):
        inp = torch.FloatTensor(current_seq.reshape(1, *current_seq.shape))
        with torch.no_grad():
            pred = model(inp).numpy()[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()
