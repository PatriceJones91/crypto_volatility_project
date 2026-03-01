import numpy as np
import pandas as pd

def baseline_hist_vol(df: pd.DataFrame) -> np.ndarray:
    return df["realized_vol"].to_numpy(dtype=float)

def baseline_ewma_vol(df: pd.DataFrame) -> np.ndarray:
    return df["ewma_vol"].to_numpy(dtype=float)
