"""W4: forecast error metrics. All three accept lists, NumPy arrays or Series. Do not change signatures."""


def wape(y_true, y_pred) -> float:
    """sum(|y_true - y_pred|) / sum(|y_true|). nan if the denominator is 0 or the input is empty."""
    raise NotImplementedError


def bias(y_true, y_pred) -> float:
    """sum(y_pred - y_true) / sum(y_true). Positive means over-forecasting. nan if sum(y_true) is 0."""
    raise NotImplementedError


def mase(y_true, y_pred, y_train, m: int = 7) -> float:
    """mean(|y_true - y_pred|) / mean(|y_train[t] - y_train[t-m]|) over t = m..len(y_train)-1.
    nan if len(y_train) <= m, if y_true is empty, or if the scale is 0."""
    raise NotImplementedError
