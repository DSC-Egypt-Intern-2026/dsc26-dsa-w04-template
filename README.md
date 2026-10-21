# W4 · Nour: a demand baseline and an honest backtest

DSC Summer Internship 2026 · Data Science & Analytics track · Task after session W4 (Metrics, dashboards and forecasting basics)

**Repository name:** `dsc26-dsa-w04` (exactly this, public, in your own GitHub account)
**Language:** Python 3.12 · pandas 2.3.3 · statsmodels 0.15.0 · PyArrow 25.0.1 · pytest 9.1.1 (pinned in `requirements.txt`)
**Expected time:** 3–4 hours

## Scenario

Dr. Mona has agreed that the forecasting model must beat "a sensible baseline" before Nour pays for anything fancier. Your squad needs that baseline and the backtest harness every later model will go through. You turn Nour's sparse sales rows into complete daily series, aggregate them into Egyptian weeks, build a seasonal-naive and an ETS forecast, and write a rolling-origin backtest that never leaks the future and never scores a day when the shelf was empty.

## Set up

```bash
uv venv --python 3.12
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt   # the pinned versions; do not add packages
pytest -q                            # public tests: they fail until you build the task
```

## Files

```
nour_forecast/
  io.py          load_sales(path). Provided by DSC. Do not change (the grader overwrites it).
  metrics.py     YOU WRITE: wape, bias, mase
  prepare.py     YOU WRITE: to_daily_series, weekly
  baselines.py   YOU WRITE: seasonal_naive, ets
  backtest.py    YOU WRITE: backtest
data/sales_sample.parquet   14,807 rows: one year (4 Oct 2025 – 2 Oct 2026), 6 branches x 8 items
notebooks/explore.ipynb     an empty notebook with the imports, to use or ignore
tests/test_public.py        public tests
AI_USAGE.md                 fill in before you submit
```

`load_sales` returns `branch_id`, `item_id` (string), `date` (datetime64, a Cairo calendar day, no time zone), `units_sold` (int64) and `stock_out` (bool). Rows exist only for days with a sale or a stock-out. Fridays are low, Saturdays high; some items sell less than one a day; stock-out runs of 2–6 days are planted. The data is synthetic and generated from a seed; the hidden tests use other seeds. You may add helper modules inside `nour_forecast/` and your own tests under `tests/`.

## What to build

`nour_forecast/metrics.py` (all three accept lists, NumPy arrays or Series):

```python
def wape(y_true, y_pred) -> float:
    """sum(|y_true - y_pred|) / sum(|y_true|). nan if the denominator is 0 or the input is empty."""

def bias(y_true, y_pred) -> float:
    """sum(y_pred - y_true) / sum(y_true). Positive means over-forecasting. nan if sum(y_true) is 0."""

def mase(y_true, y_pred, y_train, m: int = 7) -> float:
    """mean(|y_true - y_pred|) / mean(|y_train[t] - y_train[t-m]|) over t = m..len(y_train)-1.
    nan if len(y_train) <= m, if y_true is empty, or if the scale is 0."""
```

`nour_forecast/prepare.py`:

```python
def to_daily_series(sales: pd.DataFrame, branch_id: str, item_id: str, start=None, end=None) -> pd.DataFrame:
    """One branch-item pair. Index: every calendar day from start (default: first row's date) to end
    (default: last row's date), daily frequency, named 'date'. Columns: units (int64, sum of units_sold
    that day, 0 on days with no row) and stock_out (bool, True if any row that day says so)."""

def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Egyptian weeks, Saturday to Friday, indexed by the Saturday ('week_start').
    Columns: units (sum), stock_out_days (count of True), days (days present). Only complete weeks
    (days == 7) are returned. All columns int64."""
```

`nour_forecast/baselines.py`:

```python
def seasonal_naive(y: pd.Series, h: int, m: int = 7) -> pd.Series:
    """Forecast i (0-based) equals y's value at position len(y) - m + (i % m). Float values.
    Index: the h days after y's last date. ValueError if len(y) < m."""

def ets(y: pd.Series, h: int, m: int = 7) -> pd.Series:
    """statsmodels ExponentialSmoothing(y, trend=None, seasonal='add', seasonal_periods=m,
    initialization_method='estimated').fit().forecast(h), clipped at 0. If len(y) < 2 * m, return
    seasonal_naive(y, h, m) instead. Same index rule as seasonal_naive."""
```

`nour_forecast/backtest.py`:

```python
def backtest(daily: pd.DataFrame, model, h: int = 14, n_origins: int = 4, step: int = 7, m: int = 7) -> pd.DataFrame:
    """daily: output of to_daily_series. model(train: pd.Series, h: int) -> pd.Series of h forecasts.
    With T = len(daily), round i (0..n_origins-1) trains on positions [0, e_i) of daily['units'] (as float)
    and tests on [e_i, e_i + h), where e_i = T - h - step * (n_origins - 1 - i). So the last test window
    ends on the last day. The model receives only the training Series.
    Days in the test window with stock_out True are excluded from wape, mase and bias.
    mase uses the full training Series as y_train, with season m.
    Returns one row per round with columns origin (first test date), wape, mase, bias, n_scored.
    ValueError if h, n_origins or step < 1, or if e_0 < 2 * m."""
```

Details the hidden tests rely on:

- `seasonal_naive` returns a float Series (`dtype float64`), indexed by the `h` calendar days after `y`'s last date, and works for any `m >= 1`.
- `ets` must use exactly the statsmodels call above (it is compared with that model to a small tolerance). If `y`'s index has no frequency set, give it a daily one before fitting. Ignore statsmodels' convergence warnings; do not change the model to silence them.
- `to_daily_series` ignores rows of other branches and items, sums several rows on the same day, and honours explicit `start` and `end` (days outside them are dropped, days inside with no row get 0 units and `stock_out` False).
- `weekly` labels each week with its Saturday; the index is named `week_start`; weeks with fewer than 7 days present (at either end) are dropped.
- `backtest`: the model is called with the training slice of `daily["units"]` as a **float** Series with its dates as index, starting on the first day of `daily` and ending the day before the round's origin; never the `stock_out` column, never a test day. Rows of the result are in round order (i = 0 first); `origin` is a Timestamp; `n_scored` is an int. `e_0` exactly equal to `2 * m` is allowed.

## Public tests

`tests/test_public.py` has seven tests. Run them with `pytest -q` from the repository root. They fail on the starter and pass on a correct solution. Passing them is necessary, not sufficient.

## How grading works

1. Push to `dsc26-dsa-w04`, then press **Submit for grading** on DSC Hex. The grader pins your latest commit and clones it.
2. If `AI_USAGE.md` is missing or lacks any of its three headings, the grader stops and reports **0**.
3. DSC's copy of `nour_forecast/io.py` replaces yours. Any `conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg` or `[tool.pytest]` settings at the repository root are removed.
4. The hidden tests run with `pytest` on Python 3.12 with the pinned packages, network off, 60 s per test and 3 minutes for the whole run. They generate fresh branches and items from seeds that are not in the starter, hand your backtest a "spy" model that inspects what it is given, and use Hypothesis for random weekly patterns.
5. You get a score from 0 to 100 and, per hidden test, its public name, pass or fail, and a short hint. A test ID passes only if every case under it passes. You never see the hidden test code.
6. You get 5 grader runs per day. Run the public tests locally first.

| ID | Public name | What it checks | Weight |
|---|---|---|---|
| H01 | metrics: WAPE and bias definitions | Arrays and Series mixed; sign of bias; NaN on zero denominators and empty input | 8 |
| H02 | metrics: MASE scale and season | `m=1` vs `m=7` on the same data; too-short training data, empty input and a zero scale give NaN | 8 |
| H03 | seasonal naive: values, dates, short input | Known values for `m=7` and `m=1`; index continues the calendar; `ValueError` when shorter than a season | 8 |
| H04 | ETS: follows the specified model | Matches the specified statsmodels model on 120 days of a seeded series to a small tolerance; falls back to seasonal naive under 14 days | 8 |
| H05 | daily series: gaps, duplicates, other pairs | Two rows on one day summed, other branches and items ignored, explicit and default start and end, daily frequency, dtypes | 8 |
| H06 | weekly: Saturday weeks, partial weeks dropped | A series starting on a Thursday and ending on a Saturday; stock-out days counted | 8 |
| H07 | backtest: origins in the right places | `h=10, n_origins=5, step=3`: the five origins at the specified positions | 10 |
| H08 | backtest: no peeking at the future | A spy model records every training Series: it always ends the day before its origin and is a float Series | 10 |
| H09 | backtest: empty-shelf days are not scored | A window with three stock-out days: `n_scored` is 4 and the error is 0 | 10 |
| H10 | backtest: the numbers end to end | Two seeded Nour fixtures: every round's WAPE, MASE and bias equal an independent computation | 10 |
| H11 | backtest: refuses impossible setups | Too-short series, `h=0`, `n_origins=0` and `step=0` raise `ValueError`; the shortest valid series runs | 5 |
| H12 | property: a perfectly weekly series scores zero | 40 random weekly patterns repeated 6–12 weeks: WAPE and bias 0; MASE 0 or NaN | 7 |
| | | **Total** | **100** |

Late policy: ×1.0 within 72 hours of the session starting, ×0.6 for the following 7 days, ×0.5 after that until Thu 3 Dec 23:59. W5 reopens W4 at ×1.0. Your best attempt counts.

## AI_USAGE.md

AI tools are allowed. Keep the three headings `## Tools`, `## What AI got wrong` and `## In my own words`. The grader does not score the content, but it reports 0 if the file or a heading is missing. Under `## In my own words`, answer in one or two sentences each, without AI:

1. Why does your backtest skip stock-out days when scoring, and what would happen to the reported accuracy if it didn't?
2. On your sample run, did ETS beat the seasonal naive for the slowest items? What would you recommend to Dr. Mona for those items?
