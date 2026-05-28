"""Performance and risk metrics for monthly return series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 12


def _as_series(values: pd.Series | pd.DataFrame, name: str = "returns") -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected a Series or one-column DataFrame.")
        values = values.iloc[:, 0]
    series = values.copy()
    series.name = series.name or name
    return pd.to_numeric(series, errors="coerce").dropna()


def _align_rf(returns: pd.Series, rf: pd.Series | None) -> pd.Series:
    if rf is None:
        return returns
    aligned = pd.concat([returns, rf.rename("rf")], axis=1, join="inner").dropna()
    return aligned.iloc[:, 0] - aligned["rf"]


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate from periodic returns."""

    returns = _as_series(returns)
    if returns.empty:
        return float("nan")
    terminal_value = float((1.0 + returns).prod())
    if terminal_value <= 0:
        return -1.0
    years = len(returns) / periods_per_year
    return terminal_value ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized sample volatility."""

    returns = _as_series(returns)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf: pd.Series | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio using periodic excess returns."""

    returns = _as_series(returns)
    excess = _align_rf(returns, rf)
    if len(excess) < 2:
        return float("nan")
    denom = excess.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(math.sqrt(periods_per_year) * excess.mean() / denom)


def sortino_ratio(
    returns: pd.Series,
    rf: pd.Series | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio using downside deviation."""

    returns = _as_series(returns)
    excess = _align_rf(returns, rf)
    if excess.empty:
        return float("nan")
    downside = excess.clip(upper=0.0)
    downside_dev = math.sqrt(float((downside**2).mean()))
    if downside_dev == 0 or np.isnan(downside_dev):
        return float("nan")
    return float(math.sqrt(periods_per_year) * excess.mean() / downside_dev)


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown series from periodic returns."""

    returns = _as_series(returns)
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough drawdown."""

    dd = drawdown(returns)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by absolute max drawdown."""

    max_dd = max_drawdown(returns)
    if max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return float(cagr(returns) / abs(max_dd))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized active return divided by tracking error."""

    returns = _as_series(returns, "returns")
    benchmark_returns = _as_series(benchmark_returns, "benchmark")
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active.std(ddof=1)
    if tracking_error == 0 or np.isnan(tracking_error):
        return float("nan")
    return float(math.sqrt(periods_per_year) * active.mean() / tracking_error)


def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    rf: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict[str, float]:
    """Compute the standard project metric set for one return series."""

    returns = _as_series(returns)
    summary = {
        "CAGR": cagr(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns, rf=rf),
        "Sortino": sortino_ratio(returns, rf=rf),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
    }
    if benchmark_returns is not None:
        summary["Information Ratio"] = information_ratio(returns, benchmark_returns)
    if turnover is not None:
        turnover = _as_series(turnover, "turnover")
        summary["Average Monthly Turnover"] = float(turnover.mean())
        summary["Annualized Turnover"] = float(turnover.mean() * PERIODS_PER_YEAR)
    if costs is not None:
        costs = _as_series(costs, "costs")
        summary["Average Monthly Cost"] = float(costs.mean())
        summary["Annualized Cost Drag"] = float(costs.mean() * PERIODS_PER_YEAR)
    return summary


def performance_table(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> pd.DataFrame:
    """Side-by-side strategy and benchmark metric table."""

    strategy_returns = _as_series(strategy_returns, "strategy")
    benchmark_returns = _as_series(benchmark_returns, "benchmark")
    strategy = performance_summary(
        strategy_returns,
        benchmark_returns=benchmark_returns,
        rf=rf,
        turnover=turnover,
        costs=costs,
    )
    benchmark = performance_summary(benchmark_returns, rf=rf)
    benchmark["Information Ratio"] = float("nan")
    benchmark["Average Monthly Turnover"] = float("nan")
    benchmark["Annualized Turnover"] = float("nan")
    benchmark["Average Monthly Cost"] = float("nan")
    benchmark["Annualized Cost Drag"] = float("nan")
    return pd.DataFrame({"Strategy": strategy, "Benchmark": benchmark})

