# Factor Model and Equity Backtesting Research Report

## Executive Summary

This project tests a monthly long-only momentum strategy and then asks whether
its performance is residual alpha or known factor exposure. The strategy earns
a headline Sharpe of 1.07
over 2011-03 to 2026-03, but its Carhart 4-factor alpha is
0.25%/month with t-stat
1.54. Because the Carhart model includes
the published momentum factor, this is the central honesty test: the default
strategy does not clear the conventional |t| >= 2 hurdle after controlling for
momentum.

## Research Design

- Attribution target: BRK-B
- Strategy universe: static liquid US large-cap list after data filtering
- Benchmark: SPY
- Signal: 12-month momentum, skipping the latest 1 month(s)
- Portfolio: long-only top 20%, equal-weighted
- Rebalance frequency: monthly
- Transaction cost: 10.0 bps per unit of one-way turnover
- Regression inference: Newey-West/HAC standard errors with 6 monthly lags

## Headline Performance

| Metric | Strategy | Benchmark |
|---|---:|---:|
| CAGR | 20.20% | 13.10% |
| Annualized Volatility | 17.39% | 14.07% |
| Sharpe | 1.07 | 0.85 |
| Sortino | 1.94 | 1.35 |
| Max Drawdown | -21.30% | -23.93% |
| Calmar | 0.95 | 0.55 |
| Information Ratio | 0.69 | n/a |
| Annualized Turnover | 256.63% | n/a |
| Annualized Cost Drag | 0.26% | n/a |

## Strategy Factor Attribution

| Model    |   Alpha Monthly |   Alpha t-stat |       R2 |   MKT_RF |        SMB |         HML |        MOM |
|:---------|----------------:|---------------:|---------:|---------:|-----------:|------------:|-----------:|
| CAPM     |      0.00568602 |        2.89158 | 0.689024 | 0.986501 | nan        | nan         | nan        |
| FF3      |      0.0048618  |        2.48938 | 0.713418 | 1.03262  |  -0.211648 |  -0.12478   | nan        |
| Carhart4 |      0.00251389 |        1.54463 | 0.831849 | 1.16688  |  -0.073797 |   0.0149407 |   0.553394 |
| FF5      |      0.00517378 |        2.65205 | 0.719049 | 1.0446   |  -0.270543 |  -0.132496  | nan        |

## Target Attribution

| Model    |   Alpha Monthly |   Alpha t-stat |       R2 |   MKT_RF |        SMB |        HML |         MOM |
|:---------|----------------:|---------------:|---------:|---------:|-----------:|-----------:|------------:|
| CAPM     |      0.00235589 |       1.05123  | 0.360618 | 0.67315  | nan        | nan        | nan         |
| FF3      |      0.00126189 |       0.69703  | 0.473599 | 0.759261 |  -0.470962 |   0.47389  | nan         |
| Carhart4 |      0.00141453 |       0.713748 | 0.474055 | 0.752757 |  -0.477801 |   0.466365 |  -0.0325472 |
| FF5      |      0.00100049 |       0.530265 | 0.476477 | 0.75721  |  -0.426774 |   0.458179 | nan         |

## In-Sample vs Out-of-Sample

|                       |   In-Sample |   Out-of-Sample |
|:----------------------|------------:|----------------:|
| CAGR                  |    0.207643 |        0.194023 |
| Annualized Volatility |    0.153876 |        0.199946 |
| Sharpe                |    1.27528  |        0.855316 |
| Sortino               |    2.24797  |        1.60279  |
| Max Drawdown          |   -0.213041 |       -0.175402 |
| Calmar                |    0.97466  |        1.10616  |
| Information Ratio     |    0.830164 |        0.51971  |

## Market Regimes

|                       |   Benchmark Up Months |   Benchmark Down Months |
|:----------------------|----------------------:|------------------------:|
| CAGR                  |             0.550023  |               -0.308742 |
| Annualized Volatility |             0.138187  |                0.130941 |
| Sharpe                |             3.19417   |               -2.80427  |
| Sortino               |            15.3678    |               -2.23094  |
| Max Drawdown          |            -0.0674779 |               -0.829522 |
| Calmar                |             8.15116   |               -0.372192 |
| Information Ratio     |             0.661885  |                0.741497 |
| Months                |           124         |               57        |
| Hit Rate              |             0.572581  |                0.631579 |

## Most Recurring Holdings

|      |   Months Held |   Share of Active Months |   Average Active Weight |
|:-----|--------------:|-------------------------:|------------------------:|
| NVDA |           109 |                 0.598901 |               0.0913261 |
| NFLX |           104 |                 0.571429 |               0.0926573 |
| AMZN |            75 |                 0.412088 |               0.0926061 |
| AMD  |            75 |                 0.412088 |               0.0912727 |
| AAPL |            69 |                 0.379121 |               0.0934124 |
| ISRG |            66 |                 0.362637 |               0.0924242 |
| META |            64 |                 0.351648 |               0.0917614 |
| ADBE |            64 |                 0.351648 |               0.0916193 |
| GE   |            53 |                 0.291209 |               0.0914237 |
| SBUX |            53 |                 0.291209 |               0.0948542 |
| MA   |            52 |                 0.285714 |               0.0940559 |
| CAT  |            52 |                 0.285714 |               0.0923077 |
| MSFT |            50 |                 0.274725 |               0.0910909 |
| UNH  |            47 |                 0.258242 |               0.0934236 |
| GS   |            46 |                 0.252747 |               0.0920949 |

## Robustness Snapshot

Top five parameter combinations by Sharpe:

|   Lookback |   Top Quantile |   Cost bps |     CAGR |   Sharpe |   Sortino |   Max Drawdown |   Information Ratio |   Annualized Turnover |   Carhart Alpha Monthly |   Carhart Alpha t-stat |
|-----------:|---------------:|-----------:|---------:|---------:|----------:|---------------:|--------------------:|----------------------:|------------------------:|-----------------------:|
|          6 |            0.2 |          5 | 0.245888 |  1.30452 |   2.53554 |      -0.240665 |            0.856467 |               3.74993 |              0.00635688 |                3.46966 |
|          6 |            0.2 |         10 | 0.243594 |  1.29348 |   2.50677 |      -0.241806 |            0.838855 |               3.74993 |              0.0061994  |                3.38365 |
|          6 |            0.2 |         20 | 0.239016 |  1.2714  |   2.44966 |      -0.244083 |            0.803606 |               3.74993 |              0.00588445 |                3.21157 |
|          6 |            0.3 |          5 | 0.212498 |  1.20918 |   2.21613 |      -0.241076 |            0.717628 |               3.1893  |              0.00395985 |                2.75817 |
|          6 |            0.3 |         10 | 0.210596 |  1.19925 |   2.19265 |      -0.242022 |            0.699002 |               3.1893  |              0.00382786 |                2.66568 |

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
