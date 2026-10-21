"""W4: the two baseline forecasts. Do not change signatures."""
import pandas as pd


def seasonal_naive(y: pd.Series, h: int, m: int = 7) -> pd.Series:
    """Forecast i (0-based) equals y's value at position len(y) - m + (i % m). Float values.
    Index: the h days after y's last date. ValueError if len(y) < m."""
    raise NotImplementedError


def ets(y: pd.Series, h: int, m: int = 7) -> pd.Series:
    """statsmodels ExponentialSmoothing(y, trend=None, seasonal='add', seasonal_periods=m,
    initialization_method='estimated').fit().forecast(h), clipped at 0. If len(y) < 2 * m, return
    seasonal_naive(y, h, m) instead. Same index rule as seasonal_naive."""
    raise NotImplementedError
