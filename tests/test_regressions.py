import numpy as np
import pandas as pd

from src.regressions import run_factor_regression, run_factor_suite


def test_factor_regression_recovers_synthetic_beta():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    market = pd.Series(rng.normal(0.005, 0.04, len(idx)), index=idx)
    rf = pd.Series(0.001, index=idx)
    returns = rf + 0.002 + 1.5 * market + rng.normal(0.0, 0.001, len(idx))
    factors = pd.DataFrame(
        {
            "MKT_RF": market,
            "SMB": rng.normal(0.0, 0.02, len(idx)),
            "HML": rng.normal(0.0, 0.02, len(idx)),
            "RMW": rng.normal(0.0, 0.02, len(idx)),
            "CMA": rng.normal(0.0, 0.02, len(idx)),
            "MOM": rng.normal(0.0, 0.02, len(idx)),
            "RF": rf,
        },
        index=idx,
    )

    result = run_factor_regression(returns, factors, "CAPM", hac_lags=3)

    assert abs(result.params["MKT_RF"] - 1.5) < 0.02
    assert abs(result.alpha_monthly - 0.002) < 0.001


def test_factor_suite_returns_all_project_models():
    idx = pd.date_range("2010-01-31", periods=80, freq="ME")
    rf = pd.Series(0.001, index=idx)
    factors = pd.DataFrame(
        {
            "MKT_RF": np.linspace(-0.03, 0.04, len(idx)),
            "SMB": np.linspace(0.02, -0.01, len(idx)),
            "HML": np.linspace(-0.02, 0.03, len(idx)),
            "RMW": np.linspace(0.01, 0.02, len(idx)),
            "CMA": np.linspace(0.00, 0.01, len(idx)),
            "MOM": np.linspace(0.04, -0.03, len(idx)),
            "RF": rf,
        },
        index=idx,
    )
    returns = rf + 0.001 + factors["MKT_RF"]

    table = run_factor_suite(returns, factors)

    assert list(table.index) == ["CAPM", "FF3", "Carhart4", "FF5"]
    assert "Alpha t-stat" in table.columns
    assert "MKT_RF" in table.columns

