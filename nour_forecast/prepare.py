"""W4: from sparse sales rows to complete daily series and Egyptian weeks. Do not change signatures."""
import pandas as pd


def to_daily_series(sales: pd.DataFrame, branch_id: str, item_id: str, start=None, end=None) -> pd.DataFrame:
    """One branch-item pair. Index: every calendar day from start (default: first row's date) to end
    (default: last row's date), daily frequency, named 'date'. Columns: units (int64, sum of units_sold
    that day, 0 on days with no row) and stock_out (bool, True if any row that day says so)."""
    raise NotImplementedError


def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Egyptian weeks, Saturday to Friday, indexed by the Saturday ('week_start').
    Columns: units (sum), stock_out_days (count of True), days (days present). Only complete weeks
    (days == 7) are returned. All columns int64."""
    raise NotImplementedError
