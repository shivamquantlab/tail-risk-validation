import numpy as np
import pytest

from tail_risk_validation.metrics import (
    historical_var_es,
    kupiec_pof_lr,
    rolling_var_es,
)


def test_historical_var_and_expected_shortfall_use_loss_tail():
    var, es = historical_var_es(np.array([-0.10, -0.05, 0.0, 0.02, 0.03]), alpha=0.6)
    assert var == pytest.approx(0.02)
    assert es == pytest.approx(0.075)


def test_rolling_forecast_does_not_use_current_or_future_return():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.5])
    var, _ = rolling_var_es(returns, lookback=4, alpha=0.75)
    altered = returns.copy()
    altered[4] = 0.9
    var_altered, _ = rolling_var_es(altered, lookback=4, alpha=0.75)
    assert var[4] == pytest.approx(var_altered[4])
    assert np.isnan(var[:4]).all()


def test_kupiec_statistic_is_zero_for_expected_empirical_rate():
    hits = np.array([True, False] * 2 + [False] * 16)
    assert kupiec_pof_lr(hits, alpha=0.9) == pytest.approx(0.0)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        historical_var_es(np.array([-0.1, 0.1]), alpha=1.0)

