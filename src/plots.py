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


def plot_rolling_factor_exposures(
    rolling_table: pd.DataFrame,
    out_path: Path | str,
) -> Path:
    """Save rolling alpha, t-stat, and factor exposure charts."""

    path = _ensure_parent(out_path)
    factor_cols = [
        col
        for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
        if col in rolling_table.columns and rolling_table[col].notna().any()
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    rolling_table["Alpha Annualized"].plot(ax=axes[0], color="#4c78a8", linewidth=1.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Rolling Annualized Alpha")
    axes[0].set_ylabel("Alpha")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    rolling_table["Alpha t-stat"].plot(ax=axes[1], color="#f58518", linewidth=1.8)
    axes[1].axhline(2.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(-2.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Rolling Alpha t-stat")
    axes[1].set_ylabel("t-stat")

    rolling_table[factor_cols].plot(ax=axes[2], linewidth=1.5)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Rolling Factor Loadings")
    axes[2].set_ylabel("Loading")
    axes[2].set_xlabel("")
    axes[2].legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_turnover_and_holdings(
    diagnostics: pd.DataFrame,
    out_path: Path | str,
) -> Path:
    """Save monthly turnover, holdings count, and transaction-cost diagnostics."""

    path = _ensure_parent(out_path)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    diagnostics["Turnover"].plot(ax=axes[0], color="#4c78a8", linewidth=1.4)
    axes[0].set_title("Monthly Turnover")
    axes[0].set_ylabel("Turnover")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    diagnostics["Holdings"].plot(ax=axes[1], color="#54a24b", linewidth=1.4)
    axes[1].set_title("Number of Holdings")
    axes[1].set_ylabel("Count")

    diagnostics["Transaction Cost"].plot(ax=axes[2], color="#e45756", linewidth=1.4)
    axes[2].set_title("Monthly Transaction Cost Drag")
    axes[2].set_ylabel("Cost")
    axes[2].set_xlabel("")
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.2%}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_gross_vs_net(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    out_path: Path | str,
) -> Path:
    """Save cumulative gross and net return comparison."""

    path = _ensure_parent(out_path)
    frame = pd.concat(
        [
            cumulative_returns(gross_returns).rename("Gross"),
            cumulative_returns(net_returns).rename("Net"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    frame.plot(ax=ax, linewidth=2.0)
    ax.set_title("Gross vs Net Strategy Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_sensitivity_heatmap(
    sensitivity: pd.DataFrame,
    out_path: Path | str,
    metric: str = "Sharpe",
    cost_bps: float = 10.0,
) -> Path:
    """Save a lookback/breadth sensitivity heatmap for one cost assumption."""

    path = _ensure_parent(out_path)
    subset = sensitivity.loc[sensitivity["Cost bps"].eq(cost_bps)]
    if subset.empty:
        subset = sensitivity
    pivot = subset.pivot_table(
        index="Lookback",
        columns="Top Quantile",
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f"{metric} Sensitivity at {cost_bps:.0f} bps Cost")
    ax.set_xlabel("Top Quantile")
    ax.set_ylabel("Lookback Months")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
