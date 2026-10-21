import math

import numpy as np
import pandas as pd
import pytest

from nour_forecast.backtest import backtest
from nour_forecast.baselines import ets, seasonal_naive
from nour_forecast.metrics import bias, mase, wape
from nour_forecast.prepare import to_daily_series, weekly


def daily(units, start="2026-01-03", stock_out=None):
    idx = pd.date_range(start, periods=len(units), freq="D", name="date")
    so = stock_out if stock_out is not None else [False] * len(units)
    return pd.DataFrame({"units": np.asarray(units, dtype="int64"), "stock_out": so}, index=idx)


def test_wape_and_bias():
    assert wape([10, 0, 10], [8, 1, 12]) == pytest.approx(5 / 20)
    assert bias([10, 0, 10], [8, 1, 12]) == pytest.approx(1 / 20)
    assert math.isnan(wape([0, 0], [1, 1]))


def test_mase_uses_the_seasonal_naive_scale():
    train = [1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9]          # |y_t - y_{t-7}| = 2 every day
    assert mase([10, 10], [12, 7], train, m=7) == pytest.approx(2.5 / 2)


def test_seasonal_naive_repeats_last_week_and_continues_the_dates():
    y = daily(list(range(1, 15)))["units"]
    fc = seasonal_naive(y, h=10)
    assert fc.tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]
    assert fc.index[0] == pd.Timestamp("2026-01-17")


def test_daily_series_fills_missing_days_with_zero():
    sales = pd.DataFrame({"branch_id": ["BR-001"] * 2, "item_id": ["ITM-0001"] * 2,
                          "date": pd.to_datetime(["2026-03-01", "2026-03-04"]),
                          "units_sold": [5, 2], "stock_out": [False, True]})
    d = to_daily_series(sales, "BR-001", "ITM-0001")
    assert d["units"].tolist() == [5, 0, 0, 2] and d["stock_out"].tolist() == [False, False, False, True]


def test_weekly_weeks_start_on_saturday():
    w = weekly(daily([1] * 21, start="2026-01-03"))          # 3 Jan 2026 is a Saturday
    assert list(w.index) == list(pd.to_datetime(["2026-01-03", "2026-01-10", "2026-01-17"]))
    assert w["units"].tolist() == [7, 7, 7]


def test_backtest_shape_and_last_origin():
    d = daily(np.tile([5, 6, 7, 8, 9, 3, 10], 12))
    bt = backtest(d, seasonal_naive, h=14, n_origins=4, step=7)
    assert list(bt.columns) == ["origin", "wape", "mase", "bias", "n_scored"]
    assert bt["origin"].iloc[-1] == d.index[-14]
    assert np.allclose(bt["wape"], 0)                          # a perfectly weekly series


def test_ets_is_non_negative_and_dated():
    y = daily(np.tile([0, 0, 1, 0, 0, 5, 0], 8))["units"]
    fc = ets(y, h=7)
    assert (fc >= 0).all() and fc.index[0] == y.index[-1] + pd.Timedelta(days=1)
