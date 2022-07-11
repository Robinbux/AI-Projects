from typing import List
import pandas as pd


def add_suffix(df: pd.DataFrame, suffix: str, cols_to_exclude: List[str] = []) -> pd.DataFrame:
    return df.rename(columns={col: col+suffix for col in df.columns if col not in cols_to_exclude})
