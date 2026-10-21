"""W4: a rolling-origin backtest that never leaks the future. Do not change the signature."""
import pandas as pd


def backtest(daily: pd.DataFrame, model, h: int = 14, n_origins: int = 4, step: int = 7, m: int = 7) -> pd.DataFrame:
    """daily: output of to_daily_series. model(train: pd.Series, h: int) -> pd.Series of h forecasts.
    With T = len(daily), round i (0..n_origins-1) trains on positions [0, e_i) of daily['units'] (as float)
    and tests on [e_i, e_i + h), where e_i = T - h - step * (n_origins - 1 - i). So the last test window
    ends on the last day. The model receives only the training Series.
    Days in the test window with stock_out True are excluded from wape, mase and bias.
    mase uses the full training Series as y_train, with season m.
    Returns one row per round with columns origin (first test date), wape, mase, bias, n_scored.
    ValueError if h, n_origins or step < 1, or if e_0 < 2 * m."""
    raise NotImplementedError
