"""Signal construction and portfolio weighting rules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def momentum_signal(
    monthly_returns: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional momentum signals.

    At date t the signal uses returns available up to t-skip. The default
    skips the most recent completed month and compounds the prior 12 monthly
    returns, which avoids using the next month's realized returns.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    if skip < 0:
        raise ValueError("skip must be non-negative.")

    returns = monthly_returns.sort_index().apply(pd.to_numeric, errors="coerce")
    min_periods = lookback if min_periods is None else min_periods
    shifted = returns.shift(skip)
    return shifted.rolling(lookback, min_periods=min_periods).apply(
        lambda window: np.prod(1.0 + window) - 1.0,
        raw=True,
    )


def equal_weight_top_quantile(
    signal_row: pd.Series,
    top_quantile: float = 0.20,
    min_assets: int = 5,
) -> pd.Series:
    """Convert one cross-section of scores into long-only equal weights."""

    if not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1].")
    if min_assets <= 0:
        raise ValueError("min_assets must be positive.")

    scores = pd.to_numeric(signal_row, errors="coerce").dropna()
    weights = pd.Series(0.0, index=signal_row.index, dtype=float)
    if len(scores) < min_assets:
        return weights

    n_select = max(1, math.ceil(len(scores) * top_quantile))
    winners = scores.sort_values(ascending=False).head(n_select).index
    weights.loc[winners] = 1.0 / len(winners)
    return weights


def build_equal_weight_portfolios(
    signals: pd.DataFrame,
    top_quantile: float = 0.20,
    min_assets: int = 5,
) -> pd.DataFrame:
    """Build a target-weight panel from a signal panel."""

    if signals.empty:
        return signals.copy()
    weights = signals.apply(
        equal_weight_top_quantile,
        axis=1,
        top_quantile=top_quantile,
        min_assets=min_assets,
    )
    return weights.reindex(columns=signals.columns).fillna(0.0)

