import math

import pandas as pd

from src.backtest import run_monthly_backtest
from src.signals import build_equal_weight_portfolios, momentum_signal


def test_momentum_signal_skips_current_month():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    returns = pd.DataFrame({"A": [0.10, 0.20, 0.30, 0.40, 0.50]}, index=idx)

    signal = momentum_signal(returns, lookback=3, skip=1)

    expected = (1.10 * 1.20 * 1.30) - 1
    assert math.isclose(signal.loc[idx[3], "A"], expected, rel_tol=1e-12)


def test_equal_weight_portfolio_selects_top_quantile():
    idx = pd.date_range("2020-01-31", periods=1, freq="ME")
    signals = pd.DataFrame(
        {"A": [3.0], "B": [2.0], "C": [1.0], "D": [0.0]},
        index=idx,
    )

    weights = build_equal_weight_portfolios(signals, top_quantile=0.5, min_assets=1)

    assert math.isclose(weights.loc[idx[0], "A"], 0.5)
    assert math.isclose(weights.loc[idx[0], "B"], 0.5)
    assert math.isclose(weights.loc[idx[0], "C"], 0.0)


def test_backtest_uses_weights_before_next_month_return():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.10, -0.20],
            "B": [0.0, 0.00, 0.10],
        },
        index=idx,
    )
    signals = pd.DataFrame(
        {
            "A": [2.0, 1.0, 1.0],
            "B": [1.0, 2.0, 2.0],
        },
        index=idx,
    )

    result = run_monthly_backtest(
        returns,
        signals,
        top_quantile=0.5,
        cost_bps=0.0,
        min_assets=1,
    )

    assert math.isclose(result.returns.loc[idx[1]], 0.10)
    assert math.isclose(result.returns.loc[idx[2]], 0.10)

