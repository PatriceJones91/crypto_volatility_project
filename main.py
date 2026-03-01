import os
import pandas as pd

from src.preprocessing import (
    load_data, add_returns, add_realized_volatility, add_ewma_volatility, add_target
)
from src.baseline_models import baseline_hist_vol, baseline_ewma_vol
from src.ml_models import build_features, train_random_forest
from src.metrics import rmse, mae

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root, "data", "btc_kaggle.csv")

    # 1) Load
    df = load_data(csv_path)
    print("Loaded:", len(df), "rows")

    # 2) Compute returns + volatility + target
    df = add_returns(df)
    df = add_realized_volatility(df, window=14)
    df = add_ewma_volatility(df, lam=0.94)
    df = add_target(df)

    # 3) Build model features
    df_model = build_features(df, window=14)

    # 4) Time split (no shuffle)
    split = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split].copy()
    test_df  = df_model.iloc[split:].copy()

    feature_cols = [c for c in df_model.columns if c not in ["Date", "target_vol"]]

    y_true = test_df["target_vol"].to_numpy()

    # 5) Baselines
    pred_hist = baseline_hist_vol(test_df)
    pred_ewma = baseline_ewma_vol(test_df)

    print("\nBASELINES")
    print("Historical Vol -> RMSE:", rmse(y_true, pred_hist), " MAE:", mae(y_true, pred_hist))
    print("EWMA Vol       -> RMSE:", rmse(y_true, pred_ewma), " MAE:", mae(y_true, pred_ewma))

    # 6) Random Forest
    model = train_random_forest(train_df, feature_cols)
    pred_rf = model.predict(test_df[feature_cols])

    print("\nRANDOM FOREST")
    print("RF -> RMSE:", rmse(y_true, pred_rf), " MAE:", mae(y_true, pred_rf))

    # 7) Save results for slides/video
    results = pd.DataFrame({
        "Date": test_df["Date"],
        "y_true_vol": y_true,
        "pred_hist": pred_hist,
        "pred_ewma": pred_ewma,
        "pred_rf": pred_rf
    })
    out_path = os.path.join(root, "results_week2.csv")
    results.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
