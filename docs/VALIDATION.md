# Verification record

Verification date: 13 October 2026. This record describes local execution of the portfolio edition; it is not a certification of the article's results.

## Automated tests

The regression suite passes 12 tests, including in a fresh virtual environment installed from `requirements.txt`. It covers authority separation, exclusion of both 23-hour and 25-hour dates, duplicate-key rejection, implausible-demand handling, exclusion of missing targets and histories, past-only features, train-only scaling, target-date split boundaries, all-model training/prediction, and protection against output overwrites.

The all-model test uses artificial data and checks that every prediction is finite, every method scores the same targets, and the equal-weight ensemble matches its definition. A separate command-line synthetic demonstration successfully produces metrics, predictions, metadata, training histories and a labelled plot.

## Execution on supplied historical data

The loader scanned all six supplied 2021–2023 EIA CSV files, containing 1,648,318 rows across authorities. The verification experiment explicitly selected PJM, with 26,280 input rows. This is an independently specified portfolio experiment; the source materials do not establish that it matches the published aggregation.

Six non-24-hour dates were excluded in full. The resulting complete grid retained 147 missing entries, including excluded intervals and three missing/invalid values on accepted dates. Every affected 168-hour history was excluded from all models equally.

| Partition | Valid target rows | Target dates |
| --- | ---: | --- |
| Training | 16,413 | 8 January 2021 through 31 December 2022 |
| Validation | 4,152 | 1 January through 30 June 2023 |
| Test | 4,224 | 1 July through 31 December 2023 |

The run completed for all nine predictors: persistence, two seasonal baselines, Random Forest, LightGBM, their equal-weight ensemble, two LSTMs and the Transformer. It used 100 trees, seed 42, and a 20-epoch neural limit. Early stopping completed LSTM24 at 18 epochs, LSTM168 at 16 epochs and Transformer24 at 17 epochs. Predictions and all reported metrics were finite.

Research observations, predictions, model scores and input hashes remain in the private verification workspace and are not distributed with the repository. The software regenerates its own local run records when executed with corresponding inputs.

## Environment and remaining scope

Verification used Python 3.12.14 on Linux x86-64 CPU with the package versions pinned in `requirements.txt`. Windows and macOS execution have not been verified; the requirements select the PyTorch CPU wheel distribution.

The CI-equivalent one-epoch synthetic command also completed in the fresh environment. `CITATION.cff` passed the Citation File Format 1.2.0 schema validator. Relative documentation links and the 18-file distribution were checked; no raw CSVs, PDFs, generated models, personal machine paths or common credential patterns were found.

Remote test results are recorded in [GitHub Actions](https://github.com/DR-ZohrehKaheh/load-forecasting-benchmark/actions/workflows/tests.yml); the README badge shows the workflow status. The original R/TensorFlow scripts, weather/IPMAN integration, TFT experiments, multi-step forecasting and exact paper reproduction are not included in the verified scope.

The publication DOI in the citation metadata is a related-article reference, not a software DOI. No software license or DOI deposit has been created by this preparation step.
