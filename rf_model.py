import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def build_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()

    out["ret_abs"] = out["log_return"].abs()
    out["ret_sq"] = out["log_return"] ** 2

    out["roll_mean_ret"] = out["log_return"].rolling(window).mean()
    out["roll_std_ret"] = out["log_return"].rolling(window).std()

    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_return"] = (out["Close"] - out["Open"]) / out["Open"]

    out["log_volume"] = (out["Volume"] + 1).apply(lambda x: __import__("math").log(x))

    feature_cols = [
        "log_return", "ret_abs", "ret_sq",
        "realized_vol", "ewma_vol",
        "roll_mean_ret", "roll_std_ret",
        "hl_range", "oc_return", "log_volume"
    ]

    return out[["Date"] + feature_cols + ["target_vol"]].dropna().reset_index(drop=True)

def train_random_forest(train_df: pd.DataFrame, feature_cols: list[str]) -> RandomForestRegressor:
    X_train = train_df[feature_cols]
    y_train = train_df["target_vol"]

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
