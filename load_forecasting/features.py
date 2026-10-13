"""Past-only inputs and target-date partitioning."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import validate_series


@dataclass
class Samples:
    timestamps: pd.DatetimeIndex
    tabular: np.ndarray
    history: np.ndarray
    target: np.ndarray
    feature_names: list


def make_samples(series):
    """Predict observed demand y[t] using only y[t-168:t] and calendar[t].

    A full finite week is required for every model, ensuring common target rows.
    Targets are never imputed. Gaps invalidate affected windows.
    """
    series = validate_series(series)
    if len(series) <= 168:
        raise ValueError("More than 168 hourly rows are required.")
    values = series.to_numpy()
    windows = np.lib.stride_tricks.sliding_window_view(values, 169)
    valid = np.isfinite(windows).all(axis=1)
    window = windows[valid]
    if not len(window):
        raise ValueError("No complete 168-hour histories with observed targets.")
    history, target = window[:, :-1].copy(), window[:, -1].copy()
    dates = series.index[168:][valid]
    past24 = history[:, -24:]
    columns = [history[:, -1], history[:, -2], history[:, -24], history[:, -168],
               past24.mean(axis=1), past24.std(axis=1)]
    names = ["lag_1", "lag_2", "lag_24", "lag_168", "past_24_mean", "past_24_std"]
    for label, vals, period in [("hour", dates.hour, 24), ("weekday", dates.dayofweek, 7),
                                 ("month", dates.month - 1, 12)]:
        columns += [np.sin(2 * np.pi * vals / period), np.cos(2 * np.pi * vals / period)]
        names += [f"{label}_sin", f"{label}_cos"]
    return Samples(dates, np.column_stack(columns), history, target, names)


def split_masks(dates, validation_start, test_start):
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("Validation start must precede test start.")
    masks = {"train": dates < validation_start,
             "validation": (dates >= validation_start) & (dates < test_start),
             "test": dates >= test_start}
    if any(int(mask.sum()) < 2 for mask in masks.values()):
        raise ValueError("Each partition needs at least two valid target rows.")
    return masks


def fit_scalers(samples, train):
    """Fit all transformation statistics exclusively on training samples."""
    x_scaler = StandardScaler().fit(samples.tabular[train])
    y_scaler = StandardScaler().fit(samples.target[train, None])
    return x_scaler, y_scaler
