"""Streamlit dashboard for the generated factor backtest reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _load_csv(name: str, **kwargs) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists():
        st.warning(f"Missing {path}. Run `python -m src.pipeline` first.")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def main() -> None:
    st.set_page_config(
        page_title="Factor Model Backtest",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
    st.title("Factor Model & Equity Backtesting Framework")
    st.caption("Generated from the Python pipeline in this repository.")

    metrics = _load_csv("strategy_metrics.csv", index_col=0)
    strategy_attr = _load_csv("strategy_factor_attribution.csv", index_col=0)
    target_attr = _load_csv("target_attribution.csv", index_col=0)
    sensitivity = _load_csv("sensitivity.csv")
    top_holdings = _load_csv("holdings_diagnostics.csv", index_col=0)
    regime = _load_csv("regime_metrics.csv", index_col=0)

    if not metrics.empty:
        cols = st.columns(4)
        cols[0].metric("Strategy CAGR", f"{metrics.loc['CAGR', 'Strategy']:.2%}")
        cols[1].metric("Sharpe", f"{metrics.loc['Sharpe', 'Strategy']:.2f}")
        cols[2].metric("Max Drawdown", f"{metrics.loc['Max Drawdown', 'Strategy']:.2%}")
        cols[3].metric(
            "Carhart Alpha t-stat",
            f"{strategy_attr.loc['Carhart4', 'Alpha t-stat']:.2f}",
        )

    tab_overview, tab_attribution, tab_robustness, tab_diagnostics = st.tabs(
        ["Overview", "Attribution", "Robustness", "Diagnostics"]
    )

    with tab_overview:
        st.subheader("Performance")
        st.dataframe(metrics.style.format("{:.2%}"), use_container_width=True)
        equity = FIGURES_DIR / "equity_curve.png"
        drawdowns = FIGURES_DIR / "drawdowns.png"
        if equity.exists():
            st.image(str(equity), caption="Equity Curve")
        if drawdowns.exists():
            st.image(str(drawdowns), caption="Drawdowns")

    with tab_attribution:
        st.subheader("Strategy Factor Attribution")
        st.dataframe(strategy_attr, use_container_width=True)
        st.subheader("Target Factor Attribution")
        st.dataframe(target_attr, use_container_width=True)
        rolling = FIGURES_DIR / "rolling_strategy_carhart.png"
        if rolling.exists():
            st.image(str(rolling), caption="Rolling Strategy Carhart Attribution")

    with tab_robustness:
        st.subheader("Parameter Sensitivity")
        st.dataframe(sensitivity, use_container_width=True)
        heatmap = FIGURES_DIR / "sensitivity_heatmap.png"
        if heatmap.exists():
            st.image(str(heatmap), caption="Sensitivity Heatmap")
        st.subheader("Regime Metrics")
        st.dataframe(regime, use_container_width=True)

    with tab_diagnostics:
        st.subheader("Recurring Holdings")
        st.dataframe(top_holdings, use_container_width=True)
        turnover = FIGURES_DIR / "turnover_holdings.png"
        gross_net = FIGURES_DIR / "gross_vs_net.png"
        if turnover.exists():
            st.image(str(turnover), caption="Turnover and Holdings")
        if gross_net.exists():
            st.image(str(gross_net), caption="Gross vs Net Returns")


if __name__ == "__main__":
    main()
