"""Strict loaders; no data downloads, fabricated chronology or silent imputation."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_series(series):
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Demand requires a DatetimeIndex.")
    if series.index.hasnans or series.index.has_duplicates:
        raise ValueError("Timestamps must be valid and unique.")
    series = series.sort_index().astype(float)
    if len(series) < 2:
        raise ValueError("At least two hourly observations are required.")
    if not series.index.equals(pd.date_range(series.index[0], series.index[-1], freq="h")):
        raise ValueError("Use a complete hourly index with NaN for missing observations.")
    finite = series.dropna()
    if not np.isfinite(finite).all() or (finite < 0).any():
        raise ValueError("Demand must be non-negative and finite, or NaN for missing values.")
    if finite.empty:
        raise ValueError("No usable demand observations.")
    return series.rename("demand_mw")


def load_eia(directory, authority, max_demand_mw=1_000_000):
    """Read legacy Date/Hour CSVs for one explicitly chosen authority.

    Only complete days with hour labels 1..24 are mapped to local interval-start
    labels. Other days are excluded in full, not coerced across a DST transition.
    These labels are timezone-naive and must not be advertised as UTC timestamps.
    """
    if not authority or not np.isfinite(max_demand_mw) or max_demand_mw <= 0:
        raise ValueError("Specify an authority and a positive finite demand bound.")
    files = sorted(Path(directory).glob("EIA*.csv"))
    if not files:
        raise FileNotFoundError("No EIA*.csv files found in the data directory.")
    chunks, total = [], 0
    columns = ["Balancing Authority", "Date", "Hour", "DemandMW"]
    for path in files:
        frame = pd.read_csv(path, usecols=columns)
        total += len(frame)
        chunks.append(frame.loc[frame[columns[0]].eq(authority)].copy())
    frame = pd.concat(chunks, ignore_index=True)
    if frame.empty:
        raise ValueError(f"Authority {authority!r} is absent from the supplied files.")
    frame["date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="raise")
    frame["Hour"] = pd.to_numeric(frame["Hour"], errors="raise")
    if frame["Hour"].isna().any() or not frame["Hour"].between(1, 25).all() or not frame["Hour"].mod(1).eq(0).all():
        raise ValueError("Hour labels must be integers between 1 and 25.")
    if frame.duplicated(["date", "Hour"]).any():
        raise ValueError("Duplicate authority/date/hour keys; check overlapping inputs.")
    bad_days = [date for date, group in frame.groupby("date")
                if set(group["Hour"]) != set(range(1, 25))]
    accepted = frame.loc[~frame["date"].isin(bad_days)].copy()
    if accepted.empty:
        raise ValueError("No complete 24-hour days remain.")
    demand = pd.to_numeric(accepted["DemandMW"], errors="raise")
    invalid = ~np.isfinite(demand) | demand.lt(0) | demand.gt(max_demand_mw)
    demand = demand.mask(invalid)
    timestamp = accepted["date"] + pd.to_timedelta(accepted["Hour"] - 1, unit="h")
    end_label = frame["date"].max().strftime("%Y-%m-%d 23:00:00")
    index = pd.date_range(frame["date"].min(), end_label, freq="h")
    series = pd.Series(demand.to_numpy(), index=timestamp).reindex(index)
    audit = {
        "input_rows_all_authorities": total,
        "authority": authority,
        "selected_authority_rows": len(frame),
        "excluded_non_24_hour_dates": [str(x.date()) for x in bad_days],
        "excluded_rows_non_24_hour_days": len(frame) - len(accepted),
        "invalid_or_missing_demand_on_accepted_days": int(invalid.sum()),
        "demand_upper_bound_mw": float(max_demand_mw),
        "hourly_grid_rows": len(series),
        "missing_grid_values": int(series.isna().sum()),
        "time_basis": "local interval-start labels; timezone not inferred",
        "input_sha256": {f.name: file_hash(f) for f in files},
    }
    return validate_series(series), audit


def synthetic_series(hours=24 * 120, seed=42):
    """Artificial, non-private demand for CI and a reproducible demonstration."""
    if hours < 720:
        raise ValueError("Synthetic demonstration needs at least 720 hours.")
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    values = 1000 + 120 * np.sin(2 * np.pi * t / 24) + 50 * np.cos(2 * np.pi * t / 168)
    values += rng.normal(0, 8, hours)
    index = pd.date_range("2021-01-01", periods=hours, freq="h")
    return pd.Series(values, index=index, name="demand_mw")
