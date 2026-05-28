"""Factor regression models with Newey-West standard errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


MODEL_FACTORS: dict[str, list[str]] = {
    "CAPM": ["MKT_RF"],
    "FF3": ["MKT_RF", "SMB", "HML"],
    "Carhart4": ["MKT_RF", "SMB", "HML", "MOM"],
    "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
}

ALL_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]


@dataclass(frozen=True)
class FactorRegressionResult:
    """Clean container around one fitted factor model."""

    model: str
    nobs: int
    r_squared: float
    params: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series

    @property
    def alpha_monthly(self) -> float:
        return float(self.params.get("const", np.nan))

    @property
    def alpha_annualized(self) -> float:
        alpha = self.alpha_monthly
        if np.isnan(alpha):
            return float("nan")
        return float((1.0 + alpha) ** 12 - 1.0)

    @property
    def alpha_tstat(self) -> float:
        return float(self.tvalues.get("const", np.nan))

    def to_row(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "Model": self.model,
            "N": self.nobs,
            "R2": self.r_squared,
            "Alpha Monthly": self.alpha_monthly,
            "Alpha Annualized": self.alpha_annualized,
            "Alpha t-stat": self.alpha_tstat,
        }
        for factor in ALL_FACTORS:
            row[factor] = float(self.params.get(factor, np.nan))
            row[f"{factor} t-stat"] = float(self.tvalues.get(factor, np.nan))
        return row


def _model_name(model: str) -> str:
    lookup = {name.upper(): name for name in MODEL_FACTORS}
    key = model.upper()
    if key not in lookup:
        raise ValueError(f"Unknown model '{model}'. Valid models: {list(MODEL_FACTORS)}")
    return lookup[key]


def _as_return_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("returns must be a Series or one-column DataFrame.")
        returns = returns.iloc[:, 0]
    series = pd.to_numeric(returns, errors="coerce")
    series.name = series.name or "return"
    return series


def regression_frame(
    asset_returns: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    factor_columns: Iterable[str],
) -> pd.DataFrame:
    """Align raw asset returns with factors and create excess returns."""

    required = [*factor_columns, "RF"]
    missing = [col for col in required if col not in factors.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")

    returns = _as_return_series(asset_returns).rename("asset_return")
    frame = pd.concat([returns, factors[required]], axis=1, join="inner").dropna()
    frame["excess_return"] = frame["asset_return"] - frame["RF"]
    return frame


def run_factor_regression(
    asset_returns: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    model: str,
    hac_lags: int = 6,
) -> FactorRegressionResult:
    """Fit one factor model using HAC/Newey-West standard errors."""

    canonical_model = _model_name(model)
    factor_columns = MODEL_FACTORS[canonical_model]
    frame = regression_frame(asset_returns, factors, factor_columns)
    min_obs = len(factor_columns) + 3
    if len(frame) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} observations for {canonical_model}; got {len(frame)}."
        )

    y = frame["excess_return"]
    x = sm.add_constant(frame[factor_columns], has_constant="add")
    fitted = sm.OLS(y, x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(hac_lags)},
    )
    return FactorRegressionResult(
        model=canonical_model,
        nobs=int(fitted.nobs),
        r_squared=float(fitted.rsquared),
        params=fitted.params,
        tvalues=fitted.tvalues,
        pvalues=fitted.pvalues,
    )


def run_factor_suite(
    asset_returns: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    models: Iterable[str] = MODEL_FACTORS.keys(),
    hac_lags: int = 6,
) -> pd.DataFrame:
    """Run CAPM, FF3, Carhart4, and FF5 regressions."""

    rows = [
        run_factor_regression(asset_returns, factors, model, hac_lags=hac_lags).to_row()
        for model in models
    ]
    table = pd.DataFrame(rows).set_index("Model")
    return table

