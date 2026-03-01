from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.preprocessing import (
    load_data,
    add_returns,
    add_realized_volatility,
    add_ewma_volatility,
    add_target,
)
from src.metrics import rmse, mae
from src.baseline_models import baseline_hist_vol, baseline_ewma_vol
from src.lstm_model import SequenceDataset, LSTMRegressor, train_lstm, predict_lstm


def print_block(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, "data", "btc_kaggle.csv")
    results_csv = os.path.join(root, "results_week3_lstm.csv")
    loss_png = os.path.join(root, "week3_lstm_loss.png")

    # Settings
    VOL_WINDOW = 14
    EWMA_LAMBDA = 0.94
    seq_len = 14
    batch_size = 32
    epochs = 30
    lr = 1e-3
    hidden_size = 64
    num_layers = 2
    dropout = 0.2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_block("WEEK 3: LSTM VOLATILITY FORECASTING")
    print("Device:", device)

    # Load + preprocess
    df = load_data(data_path)
    print(f"Loaded rows: {len(df):,}")

    df = add_returns(df)
    df = add_realized_volatility(df, window=VOL_WINDOW)
    df = add_ewma_volatility(df, lam=EWMA_LAMBDA)
    df = add_target(df)

    df_model = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df_model.columns if c not in ["Date", "target_vol"]]

    # Time split
    split_idx = int(len(df_model) * 0.8)
    df_train = df_model.iloc[:split_idx].copy()
    df_test = df_model.iloc[split_idx:].copy()

    X_train = df_train[feature_cols]
    y_train = df_train["target_vol"]
    X_test = df_test[feature_cols]
    y_test = df_test["target_vol"]

    print_block("TRAIN/TEST SPLIT")
    print(f"Train rows: {len(df_train):,} | Test rows: {len(df_test):,}")

    # Baselines
    y_true = y_test.to_numpy(dtype=float)
    pred_hist = baseline_hist_vol(df_test)
    pred_ewma = baseline_ewma_vol(df_test)

    m_hist = (rmse(y_true, pred_hist), mae(y_true, pred_hist))
    m_ewma = (rmse(y_true, pred_ewma), mae(y_true, pred_ewma))

    print_block("BASELINES")
    print(f"Historical | RMSE={m_hist[0]:.10f} | MAE={m_hist[1]:.10f}")
    print(f"EWMA       | RMSE={m_ewma[0]:.10f} | MAE={m_ewma[1]:.10f}")

    # Build sequences
    Xtr = X_train.to_numpy(dtype=float)
    ytr = y_train.to_numpy(dtype=float)
    Xte = X_test.to_numpy(dtype=float)
    yte = y_test.to_numpy(dtype=float)

    val_frac = 0.15
    split = int(len(Xtr) * (1 - val_frac))

    train_ds = SequenceDataset(Xtr[:split], ytr[:split], seq_len=seq_len)
    val_ds = SequenceDataset(Xtr[split:], ytr[split:], seq_len=seq_len)
    test_ds = SequenceDataset(Xte, yte, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Train LSTM
    print_block("TRAINING LSTM")

    model = LSTMRegressor(
        n_features=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    train_losses, val_losses = train_lstm(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    # Predict
    y_test_seq_true = yte[seq_len:]
    pred_lstm = predict_lstm(model.to(device), test_loader, device=device)

    m_lstm = (rmse(y_test_seq_true, pred_lstm), mae(y_test_seq_true, pred_lstm))

    print_block("LSTM RESULTS")
    print(f"LSTM | RMSE={m_lstm[0]:.10f} | MAE={m_lstm[1]:.10f}")

    # Save results
    results = pd.DataFrame({
        "y_true": y_test_seq_true,
        "pred_hist": pred_hist[seq_len:],
        "pred_ewma": pred_ewma[seq_len:],
        "pred_lstm": pred_lstm,
    })

    results.to_csv(results_csv, index=False)

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.savefig(loss_png)

    print_block("SAVED FILES")
    print(results_csv)
    print(loss_png)


if __name__ == "__main__":
    main()