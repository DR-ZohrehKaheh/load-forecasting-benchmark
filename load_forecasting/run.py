"""Command-line benchmark; write generated outputs only to a new local directory."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_eia, synthetic_series
from .features import make_samples, split_masks, fit_scalers
from .models import fit_network


def metric_row(actual, predicted):
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Prediction length or finiteness check failed.")
    return {"n": len(actual), "mae_mw": float(mean_absolute_error(actual, predicted)),
            "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def benchmark(series, validation_start, test_start, epochs=20, trees=100, seed=42):
    from lightgbm import LGBMRegressor
    if epochs < 1 or trees < 1:
        raise ValueError("Epoch and tree counts must be positive.")
    samples = make_samples(series)
    masks = split_masks(samples.timestamps, validation_start, test_start)
    x_scaler, y_scaler = fit_scalers(samples, masks["train"])
    # Use named columns for both fitting and prediction to preserve the feature contract.
    x = pd.DataFrame(x_scaler.transform(samples.tabular), columns=samples.feature_names)
    outputs = {part: pd.DataFrame({"timestamp": samples.timestamps[mask],
                                  "actual_mw": samples.target[mask],
                                  "persistence": samples.history[mask, -1],
                                  "seasonal24": samples.history[mask, -24],
                                  "seasonal168": samples.history[mask, -168]})
               for part, mask in masks.items() if part != "train"}
    model_specs = {
        "random_forest": RandomForestRegressor(n_estimators=trees, max_depth=16,
                    min_samples_leaf=3, random_state=seed, n_jobs=2),
        "lightgbm": LGBMRegressor(n_estimators=trees, learning_rate=0.05, num_leaves=31,
                    random_state=seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True),
    }
    for name, model in model_specs.items():
        print(f"Fitting {name}", flush=True)
        model.fit(x.loc[masks["train"]], samples.target[masks["train"]])
        for partition in outputs:
            outputs[partition][name] = model.predict(x.loc[masks[partition]])
    for frame in outputs.values():
        frame["ensemble"] = (frame["random_forest"] + frame["lightgbm"]) / 2
    histories = {}
    for name in ["lstm24", "lstm168", "transformer24"]:
        print(f"Fitting {name} (up to {epochs} epochs)", flush=True)
        predictions, histories[name] = fit_network(name, samples.history, samples.target, masks,
                                                   y_scaler, epochs=epochs, seed=seed)
        for partition in outputs:
            outputs[partition][name] = predictions[partition]
    metrics = []
    for partition, frame in outputs.items():
        for name in frame.columns[2:]:
            metrics.append({"partition": partition, "model": name,
                            **metric_row(frame.actual_mw.to_numpy(), frame[name].to_numpy())})
    partitions = {name: {"n": int(mask.sum()), "first_target": str(samples.timestamps[mask][0]),
                         "last_target": str(samples.timestamps[mask][-1])} for name, mask in masks.items()}
    metadata = {"partitions": partitions, "feature_names": samples.feature_names,
                "feature_scaler_mean": x_scaler.mean_.tolist(), "feature_scaler_scale": x_scaler.scale_.tolist(),
                "target_scaler_mean": y_scaler.mean_.tolist(), "target_scaler_scale": y_scaler.scale_.tolist(),
                "valid_target_rows": len(samples.target), "dropped_target_candidates": len(series) - 168 - len(samples.target),
                "epochs_limit": epochs, "tree_count": trees, "seed": seed,
                "forecast_protocol": "rolling one-step retrospective evaluation; observed past demand becomes available at each origin",
                "model_config": {name: model.get_params() for name, model in model_specs.items()}}
    return pd.DataFrame(metrics), outputs, histories, metadata


def plot_demo(frame, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = frame.iloc[:168]
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in ["actual_mw", "persistence", "random_forest", "lstm24"]:
        ax.plot(frame.timestamp, frame[name], label=name.replace("_", " "), linewidth=1)
    ax.set(xlabel="Target interval start", ylabel="Demand (MW)", title="Synthetic demonstration — first test week")
    ax.legend(ncol=2, frameon=False)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Supervised load forecasting portfolio benchmark")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use clearly labelled artificial data")
    source.add_argument("--data-dir", type=Path, help="Local directory of legacy EIA*.csv inputs")
    parser.add_argument("--authority", help="Required with --data-dir; no implicit aggregation")
    parser.add_argument("--max-demand-mw", type=float, default=1_000_000)
    parser.add_argument("--validation-start")
    parser.add_argument("--test-start")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.trees < 1:
        parser.error("--epochs and --trees must be positive")
    if args.demo and args.authority:
        parser.error("--authority only applies to --data-dir")
    if args.data_dir and not args.authority:
        parser.error("--authority is required for EIA input")
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir to avoid overwriting an earlier run")
    if args.demo:
        series = synthetic_series(seed=args.seed)
        audit = {"source": "synthetic demonstration; not research data", "hours": len(series)}
        val = args.validation_start or str(series.index[int(len(series) * .6)])
        test = args.test_start or str(series.index[int(len(series) * .8)])
    else:
        series, audit = load_eia(args.data_dir, args.authority, args.max_demand_mw)
        val, test = args.validation_start or "2023-01-01", args.test_start or "2023-07-01"
    metrics, outputs, histories, metadata = benchmark(series, val, test, args.epochs, args.trees, args.seed)
    metadata.update({"data_audit": audit, "python_version": platform.python_version(),
                     "versions": {name: importlib.metadata.version(name) for name in
                         ["numpy", "pandas", "scikit-learn", "lightgbm", "torch", "matplotlib"]}})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    for partition, frame in outputs.items():
        frame.to_csv(args.output_dir / f"{partition}_predictions.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "training_history.json").write_text(json.dumps(histories, indent=2, allow_nan=False) + "\n")
    if args.demo:
        plot_demo(outputs["test"], args.output_dir / "synthetic_forecast.png")
    print(metrics.to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
