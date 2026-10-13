"""Regression tests for the scientific failure modes found in the source package."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_forecasting.data import load_eia, synthetic_series, validate_series
from load_forecasting.features import make_samples, split_masks, fit_scalers
from load_forecasting.run import benchmark, main


def write_eia(directory, days=12):
    rows = []
    for date in pd.date_range("2021-01-01", periods=days):
        for authority, offset in [("TEST", 100), ("OTHER", 10_000)]:
            for hour in range(1, 25):
                rows.append([authority, date.strftime("%m/%d/%Y"), hour, offset + hour])
    frame = pd.DataFrame(rows, columns=["Balancing Authority", "Date", "Hour", "DemandMW"])
    frame.to_csv(directory / "EIA_test.csv", index=False)
    return frame


def test_authorities_not_concatenated(tmp_path):
    write_eia(tmp_path)
    series, audit = load_eia(tmp_path, "TEST")
    assert len(series) == 12 * 24
    assert series.max() == 124
    assert audit["selected_authority_rows"] == 288


@pytest.mark.parametrize("extra_hour", [False, True])
def test_non_24_hour_day_is_excluded_without_time_compression(tmp_path, extra_hour):
    frame = write_eia(tmp_path)
    if extra_hour:
        frame.loc[len(frame)] = ["TEST", "01/02/2021", 25, 500]
    else:
        frame = frame.drop(frame[(frame.Date == "01/02/2021") & (frame.Hour == 24) & (frame["Balancing Authority"] == "TEST")].index)
    frame.to_csv(tmp_path / "EIA_test.csv", index=False)
    series, audit = load_eia(tmp_path, "TEST")
    assert series.loc["2021-01-02"].isna().all()
    assert len(series) == 288
    assert len(audit["excluded_non_24_hour_dates"]) == 1


def test_duplicate_keys_rejected(tmp_path):
    frame = write_eia(tmp_path)
    pd.concat([frame, frame.iloc[:1]]).to_csv(tmp_path / "EIA_test.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_eia(tmp_path, "TEST")


def test_bad_demand_is_not_clipped_or_imputed(tmp_path):
    frame = write_eia(tmp_path)
    frame.loc[0, "DemandMW"] = 2_147_480_000
    frame.to_csv(tmp_path / "EIA_test.csv", index=False)
    series, audit = load_eia(tmp_path, "TEST")
    assert np.isnan(series.iloc[0])
    assert audit["invalid_or_missing_demand_on_accepted_days"] == 1


def test_target_and_future_do_not_change_features():
    series = synthetic_series(hours=960)
    before = make_samples(series)
    changed = series.copy(); changed.iloc[500:] += 10_000
    after = make_samples(changed)
    keep = before.timestamps <= series.index[500]
    np.testing.assert_array_equal(before.tabular[keep], after.tabular[keep])
    np.testing.assert_array_equal(before.history[keep], after.history[keep])


def test_missing_target_and_history_never_imputed():
    series = synthetic_series(hours=960); series.iloc[400] = np.nan
    samples = make_samples(series)
    assert not ((samples.timestamps >= series.index[400]) & (samples.timestamps <= series.index[568])).any()


def test_scaling_ignores_validation_and_test_values():
    series = synthetic_series(hours=960)
    before = make_samples(series)
    masks = split_masks(before.timestamps, series.index[550], series.index[750])
    changed = series.copy(); changed.iloc[550:] += 1_000_000
    after = make_samples(changed)
    for left, right in zip(fit_scalers(before, masks["train"]), fit_scalers(after, masks["train"])):
        np.testing.assert_array_equal(left.mean_, right.mean_)
        np.testing.assert_array_equal(left.scale_, right.scale_)


def test_partition_boundaries_and_historical_context():
    series = synthetic_series(hours=960)
    samples = make_samples(series)
    masks = split_masks(samples.timestamps, series.index[550], series.index[750])
    assert sum(int(m.sum()) for m in masks.values()) == len(samples.target)
    assert samples.timestamps[masks["validation"]][0] == series.index[550]
    np.testing.assert_array_equal(samples.history[masks["test"]][0], series.iloc[582:750])


def test_invalid_series_and_split():
    series = synthetic_series(hours=960)
    with pytest.raises(ValueError, match="complete hourly"):
        validate_series(series.drop(series.index[400]))
    with pytest.raises(ValueError, match="precede"):
        split_masks(series.index, "2023-01-01", "2022-01-01")


def test_all_models_train_and_score_on_identical_targets():
    series = synthetic_series(hours=960)
    metrics, predictions, histories, metadata = benchmark(series, series.index[550], series.index[750], epochs=1, trees=5)
    assert set(metrics.model) == {"persistence", "seasonal24", "seasonal168", "random_forest", "lightgbm", "ensemble", "lstm24", "lstm168", "transformer24"}
    assert len(metrics) == 18
    assert metrics.groupby("partition").n.nunique().eq(1).all()
    assert np.isfinite(metrics[["mae_mw", "rmse_mw", "r2"]].to_numpy()).all()
    for frame in predictions.values():
        np.testing.assert_allclose(frame.ensemble, (frame.random_forest + frame.lightgbm) / 2)
    assert all(len(rows) == 1 for rows in histories.values())
    assert metadata["partitions"]["test"]["n"] == 210


def test_cli_does_not_overwrite(tmp_path):
    with pytest.raises(SystemExit):
        main(["--demo", "--output-dir", str(tmp_path)])
