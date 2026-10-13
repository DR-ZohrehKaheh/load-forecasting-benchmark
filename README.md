# Short-Term Load Forecasting Benchmark

[![Tests](https://github.com/DR-ZohrehKaheh/load-forecasting-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/DR-ZohrehKaheh/load-forecasting-benchmark/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23335492.svg)](https://doi.org/10.5281/zenodo.23335492)

A Python research portfolio for evaluating electricity-demand forecasts with explicit data checks, chronological validation and past-only features.

**Project supervisor: Dr Zohreh Kaheh. Original undergraduate researcher: Lei Xing.** The project grew from supervised work in which Dr Kaheh taught the concepts and methodology, guided the initial implementation and helped develop and validate the code. See [AUTHORS.md](AUTHORS.md) for contributions.

This portfolio edition is a revised implementation with a narrower, documented scope. It is not an exact reproduction of the experiments or numerical results in the related publication. It contains source code, tests and original software documentation. Research manuscripts, the student report, raw datasets and trained models are not distributed.

## What this repository demonstrates

- Validating data from multiple balancing authorities and selecting one explicitly.
- Building demand lags and rolling features using observations before each target.
- Fitting preprocessing statistics on training data only.
- Comparing statistical baselines, tree ensembles and neural sequence models on the same target timestamps.
- Recording split boundaries, file hashes, configuration and package versions for each local run.

## Models and evaluation

| Model | Input representation |
| --- | --- |
| Persistence | Last observed demand |
| Daily and weekly seasonal baselines | Demand 24 and 168 hours earlier |
| Random Forest and LightGBM | Lagged demand, past-only rolling summaries and calendar features |
| Equal-weight ensemble | Mean of Random Forest and LightGBM predictions |
| LSTM with 24-hour and 168-hour histories | Ordered demand sequences; identical network architecture, different lookback |
| Transformer with 24-hour history | Ordered demand sequence with positional encoding |

All models are evaluated on a common set of targets with a complete preceding week. Default real-data partitions are training before 1 January 2023, validation from 1 January to 30 June 2023, and testing from 1 July 2023 onward. **Use only the supplied 2021–2023 files for that specific experiment.** Other files or split dates define a different experiment and are recorded in the run metadata.

MAE and RMSE are reported in the input demand unit, MW, alongside R². Neural early stopping uses validation loss; test results do not select epochs or model settings. Tree hyperparameters and ensemble weights are fixed. The models use different representations, so this is a comparison of complete forecasting pipelines, not a claim about intrinsic architectural superiority.

## Quick start without research data

Use Python 3.12 in a fresh virtual environment. Run commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m load_forecasting.run --demo --epochs 2 --trees 20 --output-dir outputs/demo
```

The demo generates artificial demand locally, trains every included model, and writes a labelled synthetic plot. Two epochs demonstrate execution only; they do not establish model quality.

## Run with locally held EIA files

Place the six half-year electricity CSV files in `data/raw/`, following [the data guide](data/README.md). Files remain local and are ignored by Git. Select one balancing authority explicitly:

```bash
python -m load_forecasting.run --data-dir data/raw --authority PJM --validation-start 2023-01-01 --test-start 2023-07-01 --epochs 20 --trees 100 --output-dir outputs/pjm
```

PJM is the authority used for local execution verification; it is not asserted to be the authority or aggregation used in the published study. The upper demand bound defaults to 1,000,000 MW and is configurable with `--max-demand-mw`. Review its suitability before using another series. No percentile threshold is inferred from the test data.

Each run requires a new output directory and produces:

- `metrics.csv`: validation and test scores for all nine predictors.
- `validation_predictions.csv` and `test_predictions.csv`: aligned observations and predictions.
- `run_metadata.json`: data audit, splits, transformations, configuration, versions and input hashes.
- `training_history.json`: training and validation loss by neural epoch.
- `synthetic_forecast.png`: generated only for the artificial demo.

These outputs stay local. The repository does not include numerical results from the supplied research files.

## Data handling and limitations

- Legacy EIA exports contain several authorities. They are never concatenated into a single load trajectory or implicitly summed.
- Only dates containing exactly hours 1–24 for the selected authority are accepted. A 23-hour or 25-hour day is excluded in full. The complete hourly grid is retained with missing values, and any affected forecasting window is excluded. This conservative policy avoids guessing how legacy daylight-saving rows map to UTC; it does not provide a full DST reconstruction.
- Timestamps label the start of local hourly intervals and are timezone-naive. The input `Hour` is treated as a one-based hour-ending label on accepted 24-hour dates. Do not combine authorities or weather feeds using these labels without verified timezone metadata.
- Missing, non-finite, negative or above-bound demand is marked missing, never silently clipped or imputed. Targets are unsmoothed observed demand.
- Weather and IPMAN files are not used in this edition. Their geographic/timezone alignment and historical publication availability must be established before inclusion.
- Evaluation is rolling and retrospective: observations preceding each target are assumed available. It is not a fixed-origin day-ahead forecast or a simulation of EIA reporting latency.
- The neural models are compact portfolio implementations. Transformer and LSTM results depend on training budget and representation. Temporal Fusion Transformer and the legacy R pipeline are outside the verified scope of this edition.
- Local execution and automated tests do not establish production readiness or validate the published research results.

See [METHODS.md](docs/METHODS.md) and [VALIDATION.md](docs/VALIDATION.md) for the precise implementation and verification scope.

## Attribution and citation

Please cite both authors in this order: **Lei Xing and Zohreh Kaheh**. [AUTHORS.md](AUTHORS.md) describes the original undergraduate contribution and Dr Kaheh's supervision, teaching of the concepts and methodology, guidance of the initial implementation, and contributions to code improvement and validation.

Software version **0.1.0**, released on **13 October 2026**, is preserved in [Zenodo](https://zenodo.org/records/23335492) and corresponds to the [GitHub release](https://github.com/DR-ZohrehKaheh/load-forecasting-benchmark/releases/tag/v0.1.0).

- **Version DOI:** [10.5281/zenodo.23335492](https://doi.org/10.5281/zenodo.23335492) — identifies this specific archived software release.
- **Concept DOI:** [10.5281/zenodo.23335491](https://doi.org/10.5281/zenodo.23335491) — identifies the software across archived versions and resolves to the latest version.

> Xing, L., & Kaheh, Z. (2026). *Short-Term Load Forecasting Benchmark: Supervised Research Portfolio* (Version 0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23335492

Use [CITATION.cff](CITATION.cff), [CITATION.bib](CITATION.bib), or GitHub's **Cite this repository** menu. Cite the software version actually consulted; include a commit identifier if discussing later changes on `main`.

The archive contains exactly the 18 tracked files at release commit `416c34e43046a4905cdb435c8138d6d9f5b70414`. DOI citation metadata were added to `main` after registration and are not part of that archived snapshot. The release tag has not been moved. See [archive provenance](docs/ARCHIVING.md).

Related publication: Xing, L., and Kaheh, Z. (2026). *Fair Benchmarking in Short-Term Load Forecasting*. Artificial Intelligence for Engineering. [doi:10.1049/aie2.70011](https://doi.org/10.1049/aie2.70011).

**The article DOI and software DOI identify different works.** This software is a substantially revised portfolio implementation, not an exact reproduction of the article. Citation does not grant reuse permission; see [RIGHTS.md](RIGHTS.md).

## Rights

No open-source software license has been assigned. See [RIGHTS.md](RIGHTS.md). Third-party data and dependencies have their own terms. Public visibility and a citation request do not grant additional reuse permissions.
