def merge_context_features(df, context_df=None):
    """
    Merge contextual variables into the engineered Bitcoin dataset.
    If no contextual data is provided, return the original dataset.
    """
    if context_df is None:
        return df
    return df.merge(context_df, on="Date", how="left")
