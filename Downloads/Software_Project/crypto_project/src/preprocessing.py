import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"]).diff()
    return out


def add_realized_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["realized_vol"] = np.sqrt((out["log_return"] ** 2).rolling(window).mean())
    return out


def add_ewma_volatility(df: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    out = df.copy()
    r2 = out["log_return"] ** 2
    ewma_var = np.full(len(out), np.nan)

    valid_r2 = r2.dropna()
    if valid_r2.empty:
        out["ewma_vol"] = np.nan
        return out

    seed = valid_r2.iloc[: min(30, len(valid_r2))].mean()
    start_idx = r2.first_valid_index()

    if start_idx is None:
        out["ewma_vol"] = np.nan
        return out

    ewma_var[start_idx] = seed

    for t in range(start_idx + 1, len(out)):
        prev_var = ewma_var[t - 1]
        prev_r2 = r2.iloc[t - 1]
        if np.isnan(prev_var) or np.isnan(prev_r2):
            ewma_var[t] = prev_var
        else:
            ewma_var[t] = lam * prev_var + (1 - lam) * float(prev_r2)

    out["ewma_vol"] = np.sqrt(ewma_var)
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_vol"] = out["realized_vol"].shift(-1)
    return out
