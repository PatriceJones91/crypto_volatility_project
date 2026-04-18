import os
import pandas as pd

from src.preprocessing import (
    load_data,
    validate_columns,
    add_returns,
    add_realized_volatility,
    add_ewma_volatility,
    add_target,
)
from src.baseline_models import baseline_hist_vol, baseline_ewma_vol
from src.rf_model import build_features, train_random_forest
from src.metrics import rmse, mae, save_metrics_table
from src.context_features import merge_context_features


def ensure_output_dirs(root: str) -> None:
    os.makedirs(os.path.join(root, "outputs", "charts"), exist_ok=True)
    os.makedirs(os.path.join(root, "outputs", "tables"), exist_ok=True)
    os.makedirs(os.path.join(root, "outputs", "exports"), exist_ok=True)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root, "data", "raw", "btc_kaggle.csv")

    ensure_output_dirs(root)

    # 1) Load + validate
    df = load_data(csv_path)
    validate_columns(df, ["Date", "Open", "High", "Low", "Close", "Volume"])
    print(f"Loaded {len(df):,} rows")

    # 2) Preprocess
    df = add_returns(df)
    df = add_realized_volatility(df, window=14)
    df = add_ewma_volatility(df, lam=0.94)
    df = add_target(df)

    # 3) Optional context integration
    # Replace None with a real context DataFrame later if you add one
    df = merge_context_features(df, context_df=None)

    # 4) Feature build
    df_model = build_features(df, window=14)

    # 5) Chronological split
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx].copy()
    test_df = df_model.iloc[split_idx:].copy()

    feature_cols = [c for c in df_model.columns if c not in ["Date", "target_vol"]]
    y_true = test_df["target_vol"].to_numpy()

    # 6) Baselines
    pred_hist = baseline_hist_vol(test_df)
    pred_ewma = baseline_ewma_vol(test_df)

    # 7) Random Forest
    rf_model = train_random_forest(train_df, feature_cols)
    pred_rf = rf_model.predict(test_df[feature_cols])

    # 8) Metrics
    metrics_rows = [
        {
            "Model": "Historical Volatility",
            "RMSE": rmse(y_true, pred_hist),
            "MAE": mae(y_true, pred_hist),
        },
        {
            "Model": "EWMA",
            "RMSE": rmse(y_true, pred_ewma),
            "MAE": mae(y_true, pred_ewma),
        },
        {
            "Model": "Random Forest",
            "RMSE": rmse(y_true, pred_rf),
            "MAE": mae(y_true, pred_rf),
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)

    print("\nMODEL COMPARISON")
    print(metrics_df.to_string(index=False))

    # 9) Save exports
    results = pd.DataFrame(
        {
            "Date": test_df["Date"],
            "y_true_vol": y_true,
            "pred_hist": pred_hist,
            "pred_ewma": pred_ewma,
            "pred_rf": pred_rf,
        }
    )

    results_path = os.path.join(root, "outputs", "exports", "results_week2.csv")
    metrics_path = os.path.join(root, "outputs", "tables", "metrics_week2.csv")

    results.to_csv(results_path, index=False)
    save_metrics_table(metrics_df, metrics_path)

    print(f"\nSaved results: {results_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
