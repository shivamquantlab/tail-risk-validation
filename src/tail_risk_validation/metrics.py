"""Historical VaR/ES and one-step walk-forward forecasts."""

import numpy as np


def _alpha_value(alpha: float) -> float:
    if isinstance(alpha, (bool, np.bool_)):
        raise TypeError("alpha must be a numeric confidence level")
    try:
        value = float(alpha)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("alpha must be between 0.5 and 1") from error
    if not np.isfinite(value) or not 0.5 < value < 1.0:
        raise ValueError("alpha must be between 0.5 and 1")
    return value


def _returns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ValueError("returns must be a finite vector with at least 2 observations")
    if (result <= -1).any():
        raise ValueError("simple returns must be greater than -100%")
    return result


def historical_var_es(returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Return positive loss VaR and empirical tail-mean ES at confidence alpha."""
    values = _returns(returns)
    alpha = _alpha_value(alpha)
    losses = -values
    var = float(np.quantile(losses, alpha, method="linear"))
    es = float(losses[losses >= var].mean())
    return var, es


def rolling_var_es(
    returns: np.ndarray,
    *,
    lookback: int = 250,
    alpha: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast risk for return t using only returns strictly before t."""
    values = _returns(returns)
    if (
        isinstance(lookback, (bool, np.bool_))
        or not isinstance(lookback, (int, np.integer))
        or lookback < 2
        or lookback >= values.size
    ):
        raise ValueError("lookback must be an integer of at least 2 and shorter than returns")
    alpha = _alpha_value(alpha)
    var = np.full(values.size, np.nan)
    es = np.full(values.size, np.nan)
    for index in range(lookback, values.size):
        var[index], es[index] = historical_var_es(values[index - lookback:index], alpha)
    return var, es


def kupiec_pof_lr(exceptions: np.ndarray, alpha: float = 0.95) -> float:
    """Unconditional-coverage likelihood-ratio statistic (no p-value approximation)."""
    raw_hits = np.asarray(exceptions)
    if raw_hits.ndim != 1 or raw_hits.size == 0:
        raise ValueError("exceptions must be a non-empty vector")
    if np.issubdtype(raw_hits.dtype, np.bool_):
        hits = raw_hits
    elif np.issubdtype(raw_hits.dtype, np.number):
        if not np.isfinite(raw_hits).all() or not np.isin(raw_hits, [0, 1]).all():
            raise ValueError("exceptions must contain only boolean or binary values")
        hits = raw_hits.astype(bool)
    else:
        raise ValueError("exceptions must contain only boolean or binary values")
    alpha = _alpha_value(alpha)
    expected = 1.0 - alpha
    observed = float(hits.mean())

    def bernoulli_log_likelihood(probability: float) -> float:
        success = int(hits.sum())
        failure = hits.size - success
        success_term = 0.0 if success == 0 else success * np.log(probability)
        failure_term = 0.0 if failure == 0 else failure * np.log1p(-probability)
        return float(success_term + failure_term)

    restricted = bernoulli_log_likelihood(expected)
    unrestricted = bernoulli_log_likelihood(observed)
    return float(max(0.0, -2.0 * (restricted - unrestricted)))

