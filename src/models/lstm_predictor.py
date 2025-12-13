import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # predict next price direction

class LSTMPredictor:
    _device_printed = False  # Class-level flag to print device only once

    def __init__(self, verbose=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Only print device once per session, or if verbose
        if verbose or not LSTMPredictor._device_printed:
            if verbose:
                print(f"LSTM using device: {self.device}")
            LSTMPredictor._device_printed = True
        self.model = LSTMModel().to(self.device)
        self.scaler = MinMaxScaler()

    def prepare_data(self, prices: np.ndarray, seq_len=60):
        scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i])  # next price
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, prices: np.ndarray, epochs=50, seq_len=60, verbose=False):
        X, y = self.prepare_data(prices, seq_len)
        X, y = X.to(self.device), y.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        torch.save(self.model.state_dict(), "models/lstm_xrp.pth")

    def predict_signal(self, recent_prices: np.ndarray, seq_len=60) -> bool:
        if len(recent_prices) < seq_len:
            return False
        self.model.eval()
        scaled = self.scaler.transform(recent_prices[-seq_len:].reshape(-1, 1))
        # Shape: (1, seq_len, 1) - batch, sequence, features
        seq = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(seq).cpu().numpy()
            next_price = self.scaler.inverse_transform(pred)[0][0]

        return next_price > recent_prices[-1]  # True = bullish signal
