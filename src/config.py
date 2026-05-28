"""Project defaults and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


DEFAULT_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "JPM",
    "V",
    "JNJ",
    "WMT",
    "PG",
    "XOM",
    "HD",
    "MA",
    "UNH",
    "BAC",
    "KO",
    "PEP",
    "CVX",
    "MRK",
    "ABBV",
    "COST",
    "ADBE",
    "CRM",
    "CSCO",
    "ORCL",
    "INTC",
    "NFLX",
    "DIS",
    "MCD",
    "NKE",
    "AMD",
    "QCOM",
    "TXN",
    "IBM",
    "GE",
    "CAT",
    "GS",
    "HON",
    "AMGN",
    "LOW",
    "SBUX",
    "UPS",
    "LMT",
    "TMO",
    "DHR",
    "LIN",
    "ISRG",
    "BLK",
    "SPGI",
    "MDLZ",
)


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for the default project run.

    The defaults are deliberately modest so the project can be reproduced from
    free data sources in a few minutes. The README documents the survivorship
    bias introduced by using a static current large-cap universe.
    """

    start: str = "2010-01-01"
    end: str | None = None
    attribution_target: str = "BRK-B"
    benchmark: str = "SPY"
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    lookback_months: int = 12
    skip_months: int = 1
    top_quantile: float = 0.20
    transaction_cost_bps: float = 10.0
    hac_lags: int = 6
    min_assets: int = 5
    oos_start: str = "2020-01-01"
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR


def ensure_project_dirs(config: ProjectConfig = ProjectConfig()) -> None:
    """Create local cache/report folders."""

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

