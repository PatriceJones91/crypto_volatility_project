from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    load_data,
    validate_columns,
    add_returns,
    add_realized_volatility,
    add_ewma_volatility,
    add_target,
)
from src.baseline_models import baseline_hist_vol, baseline_ewma_vol
from src.rf_model import build_features
from src.lstm_model import SequenceDataset, LSTMRegressor, train_lstm, predict_lstm
from src.metrics import rmse, mae, save_metrics_table
from src.context_features import merge_context_features


def ensure_output_dirs(root: str) -> None:
    os.makedirs(os.path.join(root, "outputs", "charts"), exist_ok=True)
    os.makedirs(os.path.join(root, "outputs", "tables"), exist_ok=True)
    os.makedirs(os.path.join(root, "outputs", "exports"), exist_ok=True)


def print_block(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, "data", "raw", "btc_kaggle.csv")
    results_csv = os.path.join(root, "outputs", "exports", "results_week3_lstm.csv")
    loss_png = os.path.join(root, "outputs", "charts", "week3_lstm_loss.png")
    metrics_csv = os.path.join(root, "outputs", "tables", "metrics_week3.csv")

    ensure_output_dirs(root)

    vol_window = 14
    ewma_lambda = 0.94
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

    df = load_data(data_path)
    validate_columns(df, ["Date", "Open", "High", "Low", "Close", "Volume"])
    print(f"Loaded rows: {len(df):,}")

    df = add_returns(df)
    df = add_realized_volatility(df, window=vol_window)
    df = add_ewma_volatility(df, lam=ewma_lambda)
    df = add_target(df)

    df = merge_context_features(df, context_df=None)

    df_model = build_features(df, window=vol_window)
    feature_cols = [c for c in df_model.columns if c not in ["Date", "target_vol"]]

    split_idx = int(len(df_model) * 0.8)
    df_train = df_model.iloc[:split_idx].copy()
    df_test = df_model.iloc[split_idx:].copy()

    X_train = df_train[feature_cols]
    y_train = df_train["target_vol"]
    X_test = df_test[feature_cols]
    y_test = df_test["target_vol"]

    print_block("TRAIN / TEST SPLIT")
    print(f"Train rows: {len(df_train):,} | Test rows: {len(df_test):,}")

    y_true = y_test.to_numpy(dtype=float)
    pred_hist = baseline_hist_vol(df_test)
    pred_ewma = baseline_ewma_vol(df_test)

    print_block("BASELINES")
    print(f"Historical Volatility | RMSE={rmse(y_true, pred_hist):.10f} | MAE={mae(y_true, pred_hist):.10f}")
    print(f"EWMA                  | RMSE={rmse(y_true, pred_ewma):.10f} | MAE={mae(y_true, pred_ewma):.10f}")

    Xtr = X_train.to_numpy(dtype=float)
    ytr = y_train.to_numpy(dtype=float)
    Xte = X_test.to_numpy(dtype=float)
    yte = y_test.to_numpy(dtype=float)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    val_frac = 0.15
    train_cut = int(len(Xtr) * (1 - val_frac))

    train_ds = SequenceDataset(Xtr[:train_cut], ytr[:train_cut], seq_len=seq_len)
    val_ds = SequenceDataset(Xtr[train_cut:], ytr[train_cut:], seq_len=seq_len)
    test_ds = SequenceDataset(Xte, yte, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print_block("TRAINING LSTM")

    model = LSTMRegressor(
        n_features=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    train_losses, val_losses = train_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    pred_lstm = predict_lstm(model.to(device), test_loader, device=device)
    y_test_seq_true = yte[seq_len - 1:]

    min_len = min(len(y_test_seq_true), len(pred_lstm))
    y_test_seq_true = y_test_seq_true[:min_len]
    pred_lstm = pred_lstm[:min_len]

    print_block("LSTM RESULTS")
    print(f"LSTM | RMSE={rmse(y_test_seq_true, pred_lstm):.10f} | MAE={mae(y_test_seq_true, pred_lstm):.10f}")

    results = pd.DataFrame(
        {
            "Date": df_test["Date"].iloc[seq_len - 1:].reset_index(drop=True)[:min_len],
            "y_true": y_test_seq_true,
            "pred_hist": pred_hist[seq_len - 1:][:min_len],
            "pred_ewma": pred_ewma[seq_len - 1:][:min_len],
            "pred_lstm": pred_lstm,
        }
    )
    results.to_csv(results_csv, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Historical Volatility",
                "RMSE": rmse(y_test_seq_true, pred_hist[seq_len - 1:][:min_len]),
                "MAE": mae(y_test_seq_true, pred_hist[seq_len - 1:][:min_len]),
            },
            {
                "Model": "EWMA",
                "RMSE": rmse(y_test_seq_true, pred_ewma[seq_len - 1:][:min_len]),
                "MAE": mae(y_test_seq_true, pred_ewma[seq_len - 1:][:min_len]),
            },
            {
                "Model": "LSTM",
                "RMSE": rmse(y_test_seq_true, pred_lstm),
                "MAE": mae(y_test_seq_true, pred_lstm),
            },
        ]
    ).sort_values("RMSE").reset_index(drop=True)
    save_metrics_table(metrics_df, metrics_csv)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png, dpi=150)
    plt.close()

    print_block("SAVED FILES")
    print(results_csv)
    print(metrics_csv)
    print(loss_png)


if __name__ == "__main__":
    main()