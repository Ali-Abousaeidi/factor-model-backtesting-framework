"""End-to-end project runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_momentum_backtest
from .config import ProjectConfig, ensure_project_dirs
from .data import (
    fetch_fama_french_factors,
    filter_universe_by_history,
    load_monthly_returns,
)
from .diagnostics import (
    holdings_diagnostics,
    portfolio_diagnostics,
    regime_performance,
    robustness_grid,
)
from .metrics import performance_summary, performance_table
from .plots import (
    plot_drawdowns,
    plot_equity_curve,
    plot_factor_cumulative,
    plot_factor_loadings,
    plot_gross_vs_net,
    plot_rolling_factor_exposures,
    plot_sensitivity_heatmap,
    plot_turnover_and_holdings,
)
from .regressions import rolling_factor_regression, run_factor_suite


def _trim_to_active_backtest(result: BacktestResult) -> BacktestResult:
    active_dates = result.turnover[result.turnover > 0].index
    if active_dates.empty:
        raise RuntimeError("Backtest never opened a position.")
    first_active = active_dates.min()
    return BacktestResult(
        returns=result.returns.loc[first_active:],
        gross_returns=result.gross_returns.loc[first_active:],
        turnover=result.turnover.loc[first_active:],
        costs=result.costs.loc[first_active:],
        weights=result.weights,
        holdings=result.holdings,
    )


def _align_strategy_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise RuntimeError("Strategy and benchmark returns do not overlap.")
    return aligned["strategy"], aligned["benchmark"]


def _fmt_pct(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.2%}"


def _fmt_num(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.2f}"


def _markdown_metric_table(metrics_table: pd.DataFrame) -> str:
    rows = [
        ("CAGR", _fmt_pct),
        ("Annualized Volatility", _fmt_pct),
        ("Sharpe", _fmt_num),
        ("Sortino", _fmt_num),
        ("Max Drawdown", _fmt_pct),
        ("Calmar", _fmt_num),
        ("Information Ratio", _fmt_num),
        ("Annualized Turnover", _fmt_pct),
        ("Annualized Cost Drag", _fmt_pct),
    ]
    lines = [
        "| Metric | Strategy | Benchmark |",
        "|---|---:|---:|",
    ]
    for metric, formatter in rows:
        if metric not in metrics_table.index:
            continue
        strategy = formatter(float(metrics_table.loc[metric, "Strategy"]))
        benchmark = formatter(float(metrics_table.loc[metric, "Benchmark"]))
        lines.append(f"| {metric} | {strategy} | {benchmark} |")
    return "\n".join(lines)


def _write_summary(
    config: ProjectConfig,
    metrics_table: pd.DataFrame,
    target_attribution: pd.DataFrame,
    strategy_attribution: pd.DataFrame,
    strategy_returns: pd.Series,
    universe_size: int,
    out_path: Path,
) -> Path:
    period = f"{strategy_returns.index.min():%Y-%m} to {strategy_returns.index.max():%Y-%m}"
    carhart = strategy_attribution.loc["Carhart4"]
    survives = abs(float(carhart["Alpha t-stat"])) >= 2.0
    verdict = "survives" if survives else "does not survive"

    alpha_lines = []
    for model in ["CAPM", "FF3", "Carhart4", "FF5"]:
        row = strategy_attribution.loc[model]
        alpha_lines.append(
            f"- {model}: alpha {_fmt_pct(float(row['Alpha Monthly']))}/month "
            f"(t = {_fmt_num(float(row['Alpha t-stat']))})"
        )

    target_carhart = target_attribution.loc["Carhart4"]
    text = f"""# Generated Project Summary

Period: {period}

Attribution target: {config.attribution_target}
Benchmark: {config.benchmark}
Strategy: {config.lookback_months}-month momentum, skip {config.skip_months} month(s), long-only top {config.top_quantile:.0%}, equal-weighted
Universe: {universe_size} static liquid US large-cap tickers after data-availability filtering
Transaction cost: {config.transaction_cost_bps:.1f} bps per unit of one-way turnover

## Headline Performance

{_markdown_metric_table(metrics_table)}

## Alpha After Factors

{chr(10).join(alpha_lines)}

Conclusion: the strategy's Carhart alpha {verdict} the conventional |t| >= 2 hurdle.

## Attribution Target Snapshot

For {config.attribution_target}, the Carhart4 alpha is {_fmt_pct(float(target_carhart['Alpha Monthly']))}/month
with t-stat {_fmt_num(float(target_carhart['Alpha t-stat']))}; R2 is {_fmt_num(float(target_carhart['R2']))}.

## Output Files

- reports/strategy_metrics.csv
- reports/target_attribution.csv
- reports/strategy_factor_attribution.csv
- reports/backtest_returns.csv
- reports/sensitivity.csv
- reports/oos_metrics.csv
- reports/regime_metrics.csv
- reports/rolling_strategy_carhart.csv
- reports/rolling_target_carhart.csv
- reports/holdings_diagnostics.csv
- reports/portfolio_diagnostics.csv
- reports/figures/equity_curve.png
- reports/figures/drawdowns.png
- reports/figures/factor_cumulative_returns.png
- reports/figures/target_carhart_loadings.png
- reports/figures/rolling_strategy_carhart.png
- reports/figures/rolling_target_carhart.png
- reports/figures/turnover_holdings.png
- reports/figures/gross_vs_net.png
- reports/figures/sensitivity_heatmap.png
"""
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _write_research_report(
    config: ProjectConfig,
    metrics_table: pd.DataFrame,
    target_attribution: pd.DataFrame,
    strategy_attribution: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    top_holdings: pd.DataFrame,
    sensitivity: pd.DataFrame,
    strategy_returns: pd.Series,
    out_path: Path,
) -> Path:
    period = f"{strategy_returns.index.min():%Y-%m} to {strategy_returns.index.max():%Y-%m}"
    carhart = strategy_attribution.loc["Carhart4"]
    best_sensitivity = sensitivity.sort_values("Sharpe", ascending=False).head(5)

    text = f"""# Factor Model and Equity Backtesting Research Report

## Executive Summary

This project tests a monthly long-only momentum strategy and then asks whether
its performance is residual alpha or known factor exposure. The strategy earns
a headline Sharpe of {_fmt_num(float(metrics_table.loc['Sharpe', 'Strategy']))}
over {period}, but its Carhart 4-factor alpha is
{_fmt_pct(float(carhart['Alpha Monthly']))}/month with t-stat
{_fmt_num(float(carhart['Alpha t-stat']))}. Because the Carhart model includes
the published momentum factor, this is the central honesty test: the default
strategy does not clear the conventional |t| >= 2 hurdle after controlling for
momentum.

## Research Design

- Attribution target: {config.attribution_target}
- Strategy universe: static liquid US large-cap list after data filtering
- Benchmark: {config.benchmark}
- Signal: {config.lookback_months}-month momentum, skipping the latest {config.skip_months} month(s)
- Portfolio: long-only top {config.top_quantile:.0%}, equal-weighted
- Rebalance frequency: monthly
- Transaction cost: {config.transaction_cost_bps:.1f} bps per unit of one-way turnover
- Regression inference: Newey-West/HAC standard errors with {config.hac_lags} monthly lags

## Headline Performance

{_markdown_metric_table(metrics_table)}

## Strategy Factor Attribution

{strategy_attribution[['Alpha Monthly', 'Alpha t-stat', 'R2', 'MKT_RF', 'SMB', 'HML', 'MOM']].to_markdown()}

## Target Attribution

{target_attribution[['Alpha Monthly', 'Alpha t-stat', 'R2', 'MKT_RF', 'SMB', 'HML', 'MOM']].to_markdown()}

## In-Sample vs Out-of-Sample

{oos_metrics.to_markdown()}

## Market Regimes

{regime_metrics.to_markdown()}

## Most Recurring Holdings

{top_holdings.to_markdown()}

## Robustness Snapshot

Top five parameter combinations by Sharpe:

{best_sensitivity.to_markdown(index=False)}

## Caveats

- The universe is static, so survivorship bias likely inflates performance.
- Yahoo Finance data does not fully capture delisting returns.
- The transaction-cost model is simple and does not include market impact or taxes.
- Parameter sweeps are reported as robustness checks, not as permission to data-mine the best result.
- Strong raw performance is not the same as statistically significant alpha after published factors.

## Figures

- `reports/figures/equity_curve.png`
- `reports/figures/drawdowns.png`
- `reports/figures/rolling_strategy_carhart.png`
- `reports/figures/rolling_target_carhart.png`
- `reports/figures/turnover_holdings.png`
- `reports/figures/gross_vs_net.png`
- `reports/figures/sensitivity_heatmap.png`
"""
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _run_sensitivity(
    universe_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    factors: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    return robustness_grid(
        universe_returns=universe_returns,
        benchmark_returns=benchmark_returns,
        factors=factors,
        skip=config.skip_months,
        min_assets=config.min_assets,
        hac_lags=config.hac_lags,
    )


def _run_oos_split(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    factors: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    split = pd.Timestamp(config.oos_start).to_period("M").to_timestamp("M")
    slices = {
        "In-Sample": strategy_returns.loc[strategy_returns.index < split],
        "Out-of-Sample": strategy_returns.loc[strategy_returns.index >= split],
    }
    rows = {}
    for label, returns in slices.items():
        if len(returns) < 12:
            rows[label] = {}
            continue
        benchmark = benchmark_returns.reindex(returns.index).dropna()
        returns, benchmark = _align_strategy_benchmark(returns, benchmark)
        rows[label] = performance_summary(
            returns,
            benchmark_returns=benchmark,
            rf=factors["RF"],
        )
    return pd.DataFrame(rows)


def run_pipeline(
    config: ProjectConfig | None = None,
    force: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    """Run the full project and write reports/figures to disk."""

    config = ProjectConfig() if config is None else config
    ensure_project_dirs(config)

    factors = fetch_fama_french_factors(
        start=config.start,
        end=config.end,
        cache_dir=config.data_dir,
        force=force,
    )

    tickers = [config.attribution_target, config.benchmark, *config.universe]
    monthly_returns = load_monthly_returns(
        tickers,
        start=config.start,
        end=config.end,
        cache_dir=config.data_dir,
        force=force,
    )
    # Keep performance and factor attribution on the same fully published
    # monthly window. This also avoids treating the current partial month as a
    # completed monthly return when yfinance returns fresh daily prices.
    monthly_returns = monthly_returns.loc[monthly_returns.index <= factors.index.max()]

    missing = [
        ticker
        for ticker in [config.attribution_target, config.benchmark]
        if ticker not in monthly_returns.columns
    ]
    if missing:
        raise RuntimeError(f"Missing required ticker returns: {missing}")

    target_returns = monthly_returns[config.attribution_target].dropna()
    benchmark_returns = monthly_returns[config.benchmark].dropna()
    universe_columns = [
        ticker for ticker in config.universe if ticker in monthly_returns.columns
    ]
    min_observations = config.lookback_months + config.skip_months + 12
    universe_returns = filter_universe_by_history(
        monthly_returns[universe_columns],
        min_observations=min_observations,
    )

    target_attribution = run_factor_suite(
        target_returns,
        factors,
        hac_lags=config.hac_lags,
    )

    backtest = _trim_to_active_backtest(
        run_momentum_backtest(
            universe_returns,
            lookback=config.lookback_months,
            skip=config.skip_months,
            top_quantile=config.top_quantile,
            cost_bps=config.transaction_cost_bps,
            min_assets=config.min_assets,
        )
    )
    strategy_returns, benchmark_aligned = _align_strategy_benchmark(
        backtest.returns,
        benchmark_returns,
    )
    backtest_turnover = backtest.turnover.reindex(strategy_returns.index)
    backtest_costs = backtest.costs.reindex(strategy_returns.index)
    strategy_attribution = run_factor_suite(
        strategy_returns,
        factors,
        hac_lags=config.hac_lags,
    )
    rolling_strategy = rolling_factor_regression(
        strategy_returns,
        factors,
        model="Carhart4",
        window=36,
        hac_lags=config.hac_lags,
    )
    rolling_target = rolling_factor_regression(
        target_returns,
        factors,
        model="Carhart4",
        window=36,
        hac_lags=config.hac_lags,
    )
    metrics = performance_table(
        strategy_returns,
        benchmark_aligned,
        rf=factors["RF"],
        turnover=backtest_turnover,
        costs=backtest_costs,
    )
    sensitivity = _run_sensitivity(
        universe_returns,
        benchmark_returns,
        factors,
        config,
    )
    oos_metrics = _run_oos_split(
        strategy_returns,
        benchmark_aligned,
        factors,
        config,
    )
    portfolio_diag = portfolio_diagnostics(
        backtest.weights,
        backtest_turnover,
        backtest_costs,
    )
    top_holdings = holdings_diagnostics(backtest.weights)
    regime_metrics = regime_performance(
        strategy_returns,
        benchmark_aligned,
        rf=factors["RF"],
    )

    target_attribution_path = config.reports_dir / "target_attribution.csv"
    strategy_attribution_path = config.reports_dir / "strategy_factor_attribution.csv"
    metrics_path = config.reports_dir / "strategy_metrics.csv"
    backtest_returns_path = config.reports_dir / "backtest_returns.csv"
    sensitivity_path = config.reports_dir / "sensitivity.csv"
    oos_path = config.reports_dir / "oos_metrics.csv"
    rolling_strategy_path = config.reports_dir / "rolling_strategy_carhart.csv"
    rolling_target_path = config.reports_dir / "rolling_target_carhart.csv"
    portfolio_diag_path = config.reports_dir / "portfolio_diagnostics.csv"
    top_holdings_path = config.reports_dir / "holdings_diagnostics.csv"
    regime_path = config.reports_dir / "regime_metrics.csv"

    target_attribution.to_csv(target_attribution_path)
    strategy_attribution.to_csv(strategy_attribution_path)
    metrics.to_csv(metrics_path)
    sensitivity.to_csv(sensitivity_path, index=False)
    oos_metrics.to_csv(oos_path)
    rolling_strategy.to_csv(rolling_strategy_path, index_label="Date")
    rolling_target.to_csv(rolling_target_path, index_label="Date")
    portfolio_diag.to_csv(portfolio_diag_path, index_label="Date")
    top_holdings.to_csv(top_holdings_path)
    regime_metrics.to_csv(regime_path)

    backtest_frame = pd.concat(
        [
            strategy_returns.rename("strategy_net_return"),
            backtest.gross_returns.reindex(strategy_returns.index).rename("strategy_gross_return"),
            benchmark_aligned.rename("benchmark_return"),
            backtest_turnover.rename("turnover"),
            backtest_costs.rename("transaction_cost"),
        ],
        axis=1,
    )
    backtest_frame.to_csv(backtest_returns_path, index_label="Date")

    factor_plot = plot_factor_cumulative(
        factors,
        config.figures_dir / "factor_cumulative_returns.png",
    )
    equity_plot = plot_equity_curve(
        strategy_returns,
        benchmark_aligned,
        config.figures_dir / "equity_curve.png",
    )
    drawdown_plot = plot_drawdowns(
        strategy_returns,
        benchmark_aligned,
        config.figures_dir / "drawdowns.png",
    )
    loadings_plot = plot_factor_loadings(
        target_attribution,
        "Carhart4",
        config.figures_dir / "target_carhart_loadings.png",
    )
    rolling_strategy_plot = plot_rolling_factor_exposures(
        rolling_strategy,
        config.figures_dir / "rolling_strategy_carhart.png",
    )
    rolling_target_plot = plot_rolling_factor_exposures(
        rolling_target,
        config.figures_dir / "rolling_target_carhart.png",
    )
    turnover_plot = plot_turnover_and_holdings(
        portfolio_diag,
        config.figures_dir / "turnover_holdings.png",
    )
    gross_net_plot = plot_gross_vs_net(
        backtest.gross_returns.reindex(strategy_returns.index),
        strategy_returns,
        config.figures_dir / "gross_vs_net.png",
    )
    sensitivity_plot = plot_sensitivity_heatmap(
        sensitivity,
        config.figures_dir / "sensitivity_heatmap.png",
        metric="Sharpe",
        cost_bps=config.transaction_cost_bps,
    )

    summary_path = _write_summary(
        config=config,
        metrics_table=metrics,
        target_attribution=target_attribution,
        strategy_attribution=strategy_attribution,
        strategy_returns=strategy_returns,
        universe_size=universe_returns.shape[1],
        out_path=config.reports_dir / "summary.md",
    )
    research_report_path = _write_research_report(
        config=config,
        metrics_table=metrics,
        target_attribution=target_attribution,
        strategy_attribution=strategy_attribution,
        oos_metrics=oos_metrics,
        regime_metrics=regime_metrics,
        top_holdings=top_holdings,
        sensitivity=sensitivity,
        strategy_returns=strategy_returns,
        out_path=config.reports_dir / "research_report.md",
    )

    return {
        "factors": factors,
        "monthly_returns": monthly_returns,
        "universe_returns": universe_returns,
        "target_attribution": target_attribution,
        "strategy_attribution": strategy_attribution,
        "rolling_strategy": rolling_strategy,
        "rolling_target": rolling_target,
        "metrics": metrics,
        "sensitivity": sensitivity,
        "oos_metrics": oos_metrics,
        "portfolio_diagnostics": portfolio_diag,
        "top_holdings": top_holdings,
        "regime_metrics": regime_metrics,
        "target_attribution_path": target_attribution_path,
        "strategy_attribution_path": strategy_attribution_path,
        "metrics_path": metrics_path,
        "backtest_returns_path": backtest_returns_path,
        "sensitivity_path": sensitivity_path,
        "oos_path": oos_path,
        "rolling_strategy_path": rolling_strategy_path,
        "rolling_target_path": rolling_target_path,
        "portfolio_diag_path": portfolio_diag_path,
        "top_holdings_path": top_holdings_path,
        "regime_path": regime_path,
        "summary_path": summary_path,
        "research_report_path": research_report_path,
        "factor_plot": factor_plot,
        "equity_plot": equity_plot,
        "drawdown_plot": drawdown_plot,
        "loadings_plot": loadings_plot,
        "rolling_strategy_plot": rolling_strategy_plot,
        "rolling_target_plot": rolling_target_plot,
        "turnover_plot": turnover_plot,
        "gross_net_plot": gross_net_plot,
        "sensitivity_plot": sensitivity_plot,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the factor model backtest project.")
    parser.add_argument("--start", default=ProjectConfig.start)
    parser.add_argument("--end", default=ProjectConfig.end)
    parser.add_argument("--target", default=ProjectConfig.attribution_target)
    parser.add_argument("--benchmark", default=ProjectConfig.benchmark)
    parser.add_argument("--cost-bps", type=float, default=ProjectConfig.transaction_cost_bps)
    parser.add_argument("--lookback", type=int, default=ProjectConfig.lookback_months)
    parser.add_argument("--skip", type=int, default=ProjectConfig.skip_months)
    parser.add_argument("--force", action="store_true", help="Refresh cached data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = replace(
        ProjectConfig(),
        start=args.start,
        end=args.end,
        attribution_target=args.target.upper(),
        benchmark=args.benchmark.upper(),
        transaction_cost_bps=args.cost_bps,
        lookback_months=args.lookback,
        skip_months=args.skip,
    )
    outputs = run_pipeline(config=config, force=args.force)
    metrics: pd.DataFrame = outputs["metrics"]  # type: ignore[assignment]
    summary_path: Path = outputs["summary_path"]  # type: ignore[assignment]
    print(f"Wrote {summary_path}")
    print(
        "Strategy CAGR "
        f"{metrics.loc['CAGR', 'Strategy']:.2%}, "
        "Sharpe "
        f"{metrics.loc['Sharpe', 'Strategy']:.2f}, "
        "MaxDD "
        f"{metrics.loc['Max Drawdown', 'Strategy']:.2%}"
    )


if __name__ == "__main__":
    main()
