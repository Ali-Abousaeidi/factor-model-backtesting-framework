"""Monthly momentum backtest with turnover and transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .signals import build_equal_weight_portfolios, momentum_signal


@dataclass(frozen=True)
class BacktestResult:
    """Outputs from the monthly rebalancing loop."""

    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame
    holdings: pd.DataFrame


def run_monthly_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    top_quantile: float = 0.20,
    cost_bps: float = 10.0,
    min_assets: int = 5,
) -> BacktestResult:
    """Run a no-look-ahead monthly backtest.

    Weights formed at month-end t earn returns in month t+1. Transaction costs
    are charged from the t rebalance and subtracted from the next month's
    realized return.
    """

    returns = returns.sort_index().apply(pd.to_numeric, errors="coerce")
    signals = signals.reindex(index=returns.index, columns=returns.columns)
    weights = build_equal_weight_portfolios(
        signals,
        top_quantile=top_quantile,
        min_assets=min_assets,
    ).reindex(index=returns.index, columns=returns.columns, fill_value=0.0)

    net_records: list[tuple[pd.Timestamp, float]] = []
    gross_records: list[tuple[pd.Timestamp, float]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    cost_records: list[tuple[pd.Timestamp, float]] = []

    previous_weights = pd.Series(0.0, index=returns.columns, dtype=float)
    dates = list(returns.index)
    for idx in range(len(dates) - 1):
        rebalance_date = dates[idx]
        realized_date = dates[idx + 1]
        target_weights = weights.loc[rebalance_date].fillna(0.0)
        next_returns = returns.loc[realized_date].fillna(0.0)

        turnover = float(0.5 * (target_weights - previous_weights).abs().sum())
        cost = float(turnover * cost_bps / 10000.0)
        gross_return = float((target_weights * next_returns).sum())
        net_return = gross_return - cost

        net_records.append((realized_date, net_return))
        gross_records.append((realized_date, gross_return))
        turnover_records.append((realized_date, turnover))
        cost_records.append((realized_date, cost))
        previous_weights = target_weights

    net = pd.Series(dict(net_records), name="Strategy Net Return").sort_index()
    gross = pd.Series(dict(gross_records), name="Strategy Gross Return").sort_index()
    turnover_series = pd.Series(dict(turnover_records), name="Turnover").sort_index()
    cost_series = pd.Series(dict(cost_records), name="Transaction Cost").sort_index()
    holdings = weights.gt(0.0)

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        turnover=turnover_series,
        costs=cost_series,
        weights=weights,
        holdings=holdings,
    )


def run_momentum_backtest(
    returns: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    top_quantile: float = 0.20,
    cost_bps: float = 10.0,
    min_assets: int = 5,
) -> BacktestResult:
    """Build momentum signals and run the monthly strategy."""

    signals = momentum_signal(returns, lookback=lookback, skip=skip)
    return run_monthly_backtest(
        returns=returns,
        signals=signals,
        top_quantile=top_quantile,
        cost_bps=cost_bps,
        min_assets=min_assets,
    )


def main() -> None:
    from .pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()

