"""Data acquisition and return preparation."""

from __future__ import annotations

import hashlib
import io
import re
import warnings
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as web

from .config import DATA_DIR


FRENCH_DATASETS = {
    "ff5": "F-F_Research_Data_5_Factors_2x3",
    "mom": "F-F_Momentum_Factor",
}

FRENCH_DIRECT_FILES = {
    "F-F_Research_Data_5_Factors_2x3": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "F-F_Momentum_Factor": "F-F_Momentum_Factor_CSV.zip",
}


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = str(item).strip().upper()
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


def _month_end_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp("M")
    if pd.api.types.is_integer_dtype(index):
        parsed = pd.to_datetime(index.astype(str), format="%Y%m")
        return parsed.to_period("M").to_timestamp("M")
    if pd.api.types.is_object_dtype(index):
        as_str = pd.Index(index).astype(str)
        if as_str.str.match(r"^\d{6}$").all():
            parsed = pd.to_datetime(as_str, format="%Y%m")
            return parsed.to_period("M").to_timestamp("M")
    parsed = pd.to_datetime(index)
    return parsed.to_period("M").to_timestamp("M")


def _standard_factor_name(name: object) -> str:
    text = str(name).strip().replace(" ", "")
    normalized = text.upper().replace("-", "_")
    mapping = {
        "MKT_RF": "MKT_RF",
        "MKTRF": "MKT_RF",
        "MKT-RF": "MKT_RF",
        "SMB": "SMB",
        "HML": "HML",
        "RMW": "RMW",
        "CMA": "CMA",
        "RF": "RF",
        "MOM": "MOM",
    }
    return mapping.get(normalized, normalized)


def _clean_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "Date" in frame.columns:
        frame = frame.set_index("Date")
    frame.index = _month_end_index(frame.index)
    frame.columns = [_standard_factor_name(col) for col in frame.columns]
    frame = frame.loc[:, ~pd.Index(frame.columns).str.startswith("UNNAMED")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all").sort_index()
    # Kenneth French files are in percent units. Convert to decimal returns.
    return frame / 100.0


def _filter_dates(
    frame: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    out = frame
    if start is not None:
        start_ts = pd.Timestamp(start).to_period("M").to_timestamp("M")
        out = out.loc[out.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end).to_period("M").to_timestamp("M")
        out = out.loc[out.index <= end_ts]
    return out


def _date_token(value: str | None) -> str:
    if value is None:
        return "none"
    return re.sub(r"[^0-9A-Za-z]+", "", str(value)).lower()


def _fetch_french_with_pdr(
    dataset: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        raw = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
    return _clean_factor_frame(raw)


def _fetch_french_direct(dataset: str) -> pd.DataFrame:
    filename = FRENCH_DIRECT_FILES[dataset]
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{filename}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        inner_name = archive.namelist()[0]
        text = archive.read(inner_name).decode("latin1")

    lines = text.splitlines()
    first_data_row = next(
        i for i, line in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", line)
    )
    header = lines[first_data_row - 1]
    data_lines: list[str] = []
    for line in lines[first_data_row:]:
        if re.match(r"^\s*\d{6}\s*,", line):
            data_lines.append(line)
        else:
            break

    csv_text = "\n".join([header, *data_lines])
    frame = pd.read_csv(io.StringIO(csv_text))
    frame = frame.rename(columns={frame.columns[0]: "Date"})
    return _clean_factor_frame(frame)


def fetch_fama_french_factors(
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path = DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch monthly FF5 plus momentum factors in decimal units."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"fama_french_monthly_factors_{_date_token(start)}_{_date_token(end)}.csv"
    )
    if cache_path.exists() and not force:
        factors = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return _filter_dates(factors, start=start, end=end)

    frames: dict[str, pd.DataFrame] = {}
    for key, dataset in FRENCH_DATASETS.items():
        try:
            frames[key] = _fetch_french_with_pdr(dataset, start=start, end=end)
        except Exception:
            frames[key] = _filter_dates(
                _fetch_french_direct(dataset),
                start=start,
                end=end,
            )

    factors = frames["ff5"].join(frames["mom"][["MOM"]], how="inner")
    factors = factors[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]]
    factors.to_csv(cache_path, index_label="Date")
    return factors


def _cache_key(tickers: list[str], start: str | None, end: str | None) -> str:
    payload = "|".join([*tickers, str(start), str(end)]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def download_adjusted_close(
    tickers: Iterable[str] | str,
    start: str,
    end: str | None = None,
    cache_dir: Path = DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices with yfinance and cache them locally."""

    if isinstance(tickers, str):
        ticker_list = _dedupe([tickers])
    else:
        ticker_list = _dedupe(tickers)
    if not ticker_list:
        raise ValueError("At least one ticker is required.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"prices_{_cache_key(ticker_list, start, end)}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        if "Close" in first_level:
            close = raw["Close"]
        elif "Adj Close" in first_level:
            close = raw["Adj Close"]
        else:
            raise RuntimeError(f"Could not find close columns in: {raw.columns}")
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[close_col]].rename(columns={close_col: ticker_list[0]})

    if isinstance(close, pd.Series):
        close = close.to_frame(ticker_list[0])
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.reindex(columns=[ticker for ticker in ticker_list if ticker in close.columns])
    close = close.dropna(how="all").sort_index()
    close.to_csv(cache_path, index_label="Date")
    return close


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily adjusted prices to month-end simple returns."""

    if prices.empty:
        raise ValueError("prices is empty.")
    monthly_prices = prices.sort_index().resample("ME").last()
    returns = monthly_prices.pct_change(fill_method=None)
    return returns.dropna(how="all")


def load_monthly_returns(
    tickers: Iterable[str] | str,
    start: str,
    end: str | None = None,
    cache_dir: Path = DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Download prices and return monthly return panel."""

    prices = download_adjusted_close(
        tickers,
        start=start,
        end=end,
        cache_dir=cache_dir,
        force=force,
    )
    return monthly_returns_from_prices(prices)


def filter_universe_by_history(
    returns: pd.DataFrame,
    min_observations: int,
) -> pd.DataFrame:
    """Keep assets with enough non-null monthly observations."""

    counts = returns.notna().sum(axis=0)
    kept = counts[counts >= min_observations].index
    if len(kept) == 0:
        raise RuntimeError("No universe assets have enough return history.")
    return returns.loc[:, kept]
