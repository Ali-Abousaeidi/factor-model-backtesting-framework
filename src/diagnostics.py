"""Diagnostics for portfolio holdings, regimes, and robustness analysis."""

from __future__ import annotations

import pandas as pd

from .backtest import run_momentum_backtest
from .metrics import performance_summary
from .regressions import run_factor_regression


def holdings_diagnostics(weights: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Summarize portfolio concentration and recurring holdings."""

    active_weights = weights.loc[weights.sum(axis=1) > 0].copy()
    if active_weights.empty:
        return pd.DataFrame()

    selected = active_weights.gt(0)
    months_held = selected.sum(axis=0).sort_values(ascending=False)
    avg_weight = active_weights.where(selected).mean(axis=0).reindex(months_held.index)
    share_months = months_held / len(active_weights)
    table = pd.DataFrame(
        {
            "Months Held": months_held,
            "Share of Active Months": share_months,
            "Average Active Weight": avg_weight,
        }
    )
    return table.head(top_n)


def portfolio_diagnostics(
    weights: pd.DataFrame,
    turnover: pd.Series,
    costs: pd.Series,
) -> pd.DataFrame:
    """Create a monthly diagnostics panel for the strategy."""

    active_weights = weights.reindex(turnover.index).fillna(0.0)
    holdings_count = active_weights.gt(0.0).sum(axis=1).rename("Holdings")
    max_weight = active_weights.max(axis=1).rename("Max Weight")
    return pd.concat(
        [
            holdings_count,
            max_weight,
            turnover.rename("Turnover"),
            costs.rename("Transaction Cost"),
        ],
        axis=1,
    ).dropna(how="all")


def regime_performance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series | None = None,
) -> pd.DataFrame:
    """Compare strategy behavior in benchmark up and down months."""

    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    regimes = {
        "Benchmark Up Months": aligned.loc[aligned["benchmark"] >= 0],
        "Benchmark Down Months": aligned.loc[aligned["benchmark"] < 0],
    }
    rows: dict[str, dict[str, float]] = {}
    for name, frame in regimes.items():
        if len(frame) < 6:
            rows[name] = {}
            continue
        rows[name] = performance_summary(
            frame["strategy"],
            benchmark_returns=frame["benchmark"],
            rf=rf,
        )
        rows[name]["Months"] = float(len(frame))
        rows[name]["Hit Rate"] = float((frame["strategy"] > frame["benchmark"]).mean())
    return pd.DataFrame(rows)


def robustness_grid(
    universe_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    factors: pd.DataFrame,
    lookbacks: tuple[int, ...] = (6, 9, 12, 15),
    costs_bps: tuple[float, ...] = (5.0, 10.0, 20.0),
    top_quantiles: tuple[float, ...] = (0.10, 0.20, 0.30),
    skip: int = 1,
    min_assets: int = 5,
    hac_lags: int = 6,
) -> pd.DataFrame:
    """Run a parameter grid over lookback, cost, and portfolio breadth."""

    rows: list[dict[str, float]] = []
    for lookback in lookbacks:
        for top_quantile in top_quantiles:
            for cost_bps in costs_bps:
                result = run_momentum_backtest(
                    universe_returns,
                    lookback=lookback,
                    skip=skip,
                    top_quantile=top_quantile,
                    cost_bps=cost_bps,
                    min_assets=min_assets,
                )
                active_dates = result.turnover[result.turnover > 0].index
                if active_dates.empty:
                    continue
                first_active = active_dates.min()
                strategy = result.returns.loc[first_active:]
                aligned = pd.concat(
                    [
                        strategy.rename("strategy"),
                        benchmark_returns.rename("benchmark"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                if len(aligned) < 24:
                    continue

                summary = performance_summary(
                    aligned["strategy"],
                    benchmark_returns=aligned["benchmark"],
                    rf=factors["RF"],
                    turnover=result.turnover.reindex(aligned.index),
                    costs=result.costs.reindex(aligned.index),
                )
                carhart = run_factor_regression(
                    aligned["strategy"],
                    factors,
                    "Carhart4",
                    hac_lags=hac_lags,
                )
                rows.append(
                    {
                        "Lookback": float(lookback),
                        "Top Quantile": float(top_quantile),
                        "Cost bps": float(cost_bps),
                        "CAGR": summary["CAGR"],
                        "Sharpe": summary["Sharpe"],
                        "Sortino": summary["Sortino"],
                        "Max Drawdown": summary["Max Drawdown"],
                        "Information Ratio": summary["Information Ratio"],
                        "Annualized Turnover": summary["Annualized Turnover"],
                        "Carhart Alpha Monthly": carhart.alpha_monthly,
                        "Carhart Alpha t-stat": carhart.alpha_tstat,
                    }
                )
    return pd.DataFrame(rows)
