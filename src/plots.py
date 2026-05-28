"""Project plots saved for reports and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import drawdown


sns.set_theme(style="whitegrid", context="talk")


def _ensure_parent(out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Growth of one dollar from return series."""

    return (1.0 + returns.fillna(0.0)).cumprod()


def plot_equity_curve(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    out_path: Path | str,
) -> Path:
    """Save strategy vs benchmark cumulative return plot."""

    path = _ensure_parent(out_path)
    frame = pd.concat(
        [
            cumulative_returns(strategy_returns).rename("Strategy"),
            cumulative_returns(benchmark_returns).rename("Benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    fig, ax = plt.subplots(figsize=(12, 7))
    frame.plot(ax=ax, linewidth=2.0)
    ax.set_title("Momentum Strategy vs Benchmark")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_drawdowns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    out_path: Path | str,
) -> Path:
    """Save strategy and benchmark drawdown plot."""

    path = _ensure_parent(out_path)
    frame = pd.concat(
        [
            drawdown(strategy_returns).rename("Strategy"),
            drawdown(benchmark_returns).rename("Benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    frame.plot(ax=ax, linewidth=1.8)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_factor_cumulative(
    factors: pd.DataFrame,
    out_path: Path | str,
) -> Path:
    """Save cumulative factor return plot."""

    path = _ensure_parent(out_path)
    factor_cols = [col for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if col in factors]
    cumulative = cumulative_returns(factors[factor_cols])

    fig, ax = plt.subplots(figsize=(12, 7))
    cumulative.plot(ax=ax, linewidth=1.8)
    ax.set_title("Fama-French Factor Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_factor_loadings(
    regression_table: pd.DataFrame,
    model: str,
    out_path: Path | str,
) -> Path:
    """Save a bar chart of factor loadings for one model row."""

    path = _ensure_parent(out_path)
    if model not in regression_table.index:
        raise ValueError(f"{model} not found in regression table.")
    factor_cols = [col for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if col in regression_table]
    loadings = regression_table.loc[model, factor_cols].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c78a8" if value >= 0 else "#f58518" for value in loadings]
    loadings.plot(kind="bar", ax=ax, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"{model} Factor Loadings")
    ax.set_ylabel("Loading")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

