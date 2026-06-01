import pandas as pd

from src.diagnostics import holdings_diagnostics, portfolio_diagnostics, regime_performance


def test_holdings_diagnostics_ranks_recurring_names():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    weights = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.0, 0.5],
            "B": [0.5, 0.0, 1.0, 0.5],
            "C": [0.0, 0.5, 0.0, 0.0],
        },
        index=idx,
    )

    table = holdings_diagnostics(weights, top_n=2)

    assert table.iloc[0]["Months Held"] == 3
    assert len(table) == 2


def test_portfolio_diagnostics_builds_monthly_panel():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    weights = pd.DataFrame({"A": [1.0, 0.5], "B": [0.0, 0.5]}, index=idx)
    turnover = pd.Series([0.5, 0.25], index=idx)
    costs = pd.Series([0.0005, 0.00025], index=idx)

    table = portfolio_diagnostics(weights, turnover, costs)

    assert table.loc[idx[0], "Holdings"] == 1
    assert table.loc[idx[1], "Holdings"] == 2
    assert "Transaction Cost" in table.columns


def test_regime_performance_splits_benchmark_direction():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    strategy = pd.Series([0.02, -0.01, 0.03, -0.02] * 3, index=idx)
    benchmark = pd.Series([0.01, -0.02, 0.02, -0.01] * 3, index=idx)

    table = regime_performance(strategy, benchmark)

    assert "Benchmark Up Months" in table.columns
    assert "Benchmark Down Months" in table.columns
    assert table.loc["Months", "Benchmark Up Months"] == 6
