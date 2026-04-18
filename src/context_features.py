import pandas as pd


def merge_context_features(df: pd.DataFrame, context_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge contextual variables into the engineered Bitcoin dataset.
    If no contextual data is provided, return the original dataset unchanged.
    """
    if context_df is None:
        return df

    if "Date" not in context_df.columns:
        raise ValueError("Context DataFrame must contain a 'Date' column.")

    context_df = context_df.copy()
    context_df["Date"] = pd.to_datetime(context_df["Date"])

    merged = df.merge(context_df, on="Date", how="left")
    return merged
