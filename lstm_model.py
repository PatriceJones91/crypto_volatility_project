from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    Builds sequences for LSTM:
      X_seq: shape (seq_len, n_features)
      y: scalar target (next-day volatility)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = int(seq_len)

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of rows")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if len(self.X) <= self.seq_len:
            raise ValueError("Not enough rows to build sequences. Increase data or reduce seq_len.")

    def __len__(self):
        # last sequence ends at index len-2 because y is aligned with X row
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        x_seq = self.X[idx: idx + self.seq_len]                 # (seq_len, n_features)
        y_t = self.y[idx + self.seq_len]                        # predict target at next index after window
        return torch.from_numpy(x_seq), torch.tensor(y_t)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # last time step
        pred = self.head(last)
        return pred.squeeze(-1)


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * len(xb)
            n += len(xb)

        train_loss = running / max(n, 1)

        # Validation
        model.eval()
        running_v = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running_v += loss.item() * len(xb)
                nv += len(xb)

        val_loss = running_v / max(nv, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | train MSE: {train_loss:.8f} | val MSE: {val_loss:.8f}")

    return train_losses, val_losses


def predict_lstm(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb).detach().cpu().numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)