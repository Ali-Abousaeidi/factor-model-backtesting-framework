import math

import pandas as pd

from src.metrics import cagr, drawdown, max_drawdown, performance_table, sharpe_ratio


def test_cagr_for_flat_monthly_return_series():
    returns = pd.Series([0.01] * 24)

    result = cagr(returns)

    assert math.isclose(result, (1.01**12) - 1, rel_tol=1e-12)


def test_drawdown_and_max_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])

    dd = drawdown(returns)

    assert math.isclose(dd.iloc[1], -0.20, rel_tol=1e-12)
    assert math.isclose(max_drawdown(returns), -0.20, rel_tol=1e-12)


def test_performance_table_has_strategy_and_benchmark_columns():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    strategy = pd.Series([0.01, 0.015, 0.005, 0.012] * 6, index=idx)
    benchmark = pd.Series([0.005, 0.007, 0.003, 0.006] * 6, index=idx)
    rf = pd.Series([0.001] * 24, index=idx)

    table = performance_table(strategy, benchmark, rf=rf)

    assert "Strategy" in table.columns
    assert "Benchmark" in table.columns
    assert math.isnan(table.loc["Information Ratio", "Benchmark"])
    assert sharpe_ratio(strategy, rf=rf) > 0
