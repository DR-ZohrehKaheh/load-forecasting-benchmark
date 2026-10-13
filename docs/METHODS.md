# Implementation and evaluation protocol

## Task

For a selected balancing authority, predict demand for interval t from observations before t. A timestamp labels the interval's local start, not a live prediction issue time. The evaluation assumes the preceding interval's demand is already known. This assumption must be revisited for operational use because data publication may lag the operating interval.

## Data validation

The loader checks required columns, numeric hours, one authority, duplicate keys and demand validity. It retains a regular hourly grid. Only dates whose hour-label set is exactly 1 through 24 are mapped; all other dates are absent from the usable observations. Above-bound, negative, missing and non-finite demand values become missing observations and are counted. The configurable bound is applied consistently in all partitions and is not estimated from held-out outcomes.

Every target requires 168 finite preceding observations plus a finite target. This common eligibility rule ensures all nine predictors have identical test targets and prevents sequence windows from jumping over excluded dates. Evaluation excludes these gaps, so it does not measure performance during the excluded intervals.

## Features and splits

Tree predictors use demand at lags 1, 2, 24 and 168, the mean and population standard deviation of the preceding 24 intervals, and sine/cosine representations of target hour, weekday and month. No demand feature includes the target itself.

LSTM and Transformer inputs contain only ordered past demand. Validation/test windows may contain observed history from preceding partitions, which is valid for this rolling one-step protocol. Target timestamps, not input-window start dates, determine the partitions. There is no random train/test split.

Feature standardisation and target scaling use training samples only. Neural histories use the training target mean and scale. All predictions are converted back to MW before evaluation; the target is never smoothed or replaced.

## Models

Random Forest uses 100 trees, maximum depth 16 and minimum leaf size 3. LightGBM uses 100 estimators, learning rate 0.05 and 31 leaves. Both tree counts can be changed from the CLI. The ensemble is the arithmetic mean of their predictions.

Both LSTMs use two recurrent layers with 32 hidden units and inter-layer dropout 0.1, followed by a 16-unit ReLU head and scalar output. Their lookbacks are 24 and 168. The Transformer uses a 24-step sequence, a 16-dimensional projection, fixed sinusoidal positions, two attention heads, one encoder layer with feed-forward width 32 and dropout 0.1, and a scalar output from the final token. All attention tokens occur before the target, so attention does not expose a target or future observation.

Neural training uses Adam at 0.001, MSE, batch size 128, no sample shuffling, gradient clipping at norm 1 and a default limit of 20 epochs. Validation MSE selects the checkpoint; four consecutive epochs without improvement stop training. Test data never select the checkpoint. Seeds are fixed to 42 by default and recorded. Reproducibility is scoped to the pinned CPU environment; bitwise equality across all hardware and package versions is not claimed.

## Interpretation

This edition compares complete model/input pipelines. Trees receive engineered weekly lags and calendar information while the neural models receive demand sequences of specified lengths. Results must not be presented as an architecture-only fair comparison, nor as exact reproduction of the article. A single authority, a single split and a fixed training budget do not demonstrate superiority across networks, years or operational settings.

The legacy R/TensorFlow scripts, TFT experiments, weather and economic inputs, multi-step forecasts, article figures and published result tables are outside this verified edition.

## Implementation references

- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [LightGBM regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
