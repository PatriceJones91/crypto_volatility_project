import numpy as np
import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"]).diff()
    return out

def add_realized_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["realized_vol"] = out["log_return"].rolling(window).std()
    return out

def add_ewma_volatility(df: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    out = df.copy()
    r2 = out["log_return"] ** 2
    ewma_var = np.full(len(out), np.nan)

    # seed variance using first 30 squared returns
    seed = r2.dropna().iloc[:30].mean()
    start_idx = r2.first_valid_index()
    if start_idx is None:
        out["ewma_vol"] = np.nan
        return out

    ewma_var[start_idx] = seed

    for t in range(start_idx + 1, len(out)):
        if np.isnan(ewma_var[t - 1]) or np.isnan(r2.iloc[t - 1]):
            ewma_var[t] = ewma_var[t - 1]
        else:
            ewma_var[t] = lam * ewma_var[t - 1] + (1 - lam) * float(r2.iloc[t - 1])

    out["ewma_vol"] = np.sqrt(ewma_var)
    return out

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_vol"] = out["realized_vol"].shift(-1)  # tomorrow's vol
    return out
