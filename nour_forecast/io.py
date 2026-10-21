"""Provided by DSC. Do not change: the grader overwrites this file."""
from pathlib import Path

import pandas as pd


def load_sales(path: str | Path) -> pd.DataFrame:
    """branch_id, item_id (string), date (datetime64, a Cairo calendar day, no time zone),
    units_sold (int64), stock_out (bool). Only days with a sale or a stock-out have a row."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.astype({"branch_id": "string", "item_id": "string", "units_sold": "int64", "stock_out": "bool"})
