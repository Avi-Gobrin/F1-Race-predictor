# F1 Race Winner Predictor

A supervised machine-learning pipeline that predicts Formula 1 race winners from
driver and constructor rolling form, qualifying results, track characteristics,
and weather. Per-race win prediction is framed as a classification problem over
historical data, and the model is evaluated on held-out seasons.

The project is deliberately built around the statistical tools from an
undergraduate regression and multivariate-analysis course: penalized regression
(LASSO), ordinary and logistic regression, factor analysis / principal
components, and leave-one-out style cross-validation.

## Walk-forward design

The headline experiment predicts each season using only the seasons before it,
so the model never sees the future:

| Train on            | Predict |
| ------------------- | ------- |
| 2023                | 2024    |
| 2023, 2024          | 2025    |
| 2023, 2024, 2025    | 2026    |

This expanding-window scheme is the honest way to estimate out-of-sample skill
for time-ordered sport results. A plain shuffled split would leak information
from later races into the training set.

## Pipeline

```
data ingestion  ->  feature engineering  ->  modelling  ->  walk-forward eval
  (FastF1)          (rolling form, quali,     (LASSO logit,    (per-season
                     track history, pit         linear reg,      winner / podium
                     reliability, weather)      factor analysis) accuracy, log-loss)
```

1. **Data** (`data.py`) pulls race results, qualifying, and weather from the
   official F1 timing API via [FastF1](https://docs.fastf1.dev), cached on disk.
2. **Features** (`features.py`) engineer rolling driver/constructor form,
   qualifying and grid position, driver-track history, pit-stop reliability,
   and per-race weather summaries. All rolling features are computed causally
   (using only races that already happened).
3. **Models** (`models.py`):
   - LASSO-penalized **logistic regression** for the win/podium classifier,
     with the penalty chosen by cross-validation.
   - **Linear regression** on finishing position as an interpretable baseline.
   - **Factor analysis / PCA** to summarize correlated form features and to
     inspect what latent structure drives results.
4. **Evaluation** (`evaluate.py`) runs the walk-forward splits and reports
   race-winner accuracy, podium accuracy, log-loss, and Brier score per season.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Build the feature table (downloads + caches data on first run)
python -m f1predictor.cli build-dataset --seasons 2023 2024 2025 2026

# Run the walk-forward evaluation
python -m f1predictor.cli evaluate

# Predict a single upcoming race
python -m f1predictor.cli predict --season 2026 --round 9
```

## Metrics

Race-winner prediction is hard: there are roughly twenty drivers per race, so a
naive baseline that always picks the pole-sitter is a strong reference point.
The classifier is reported against that baseline on each held-out season.

## Repository layout

```
src/f1predictor/
  config.py      paths, seasons, walk-forward split definitions
  data.py        FastF1 ingestion and caching
  features.py    causal feature engineering
  models.py      LASSO logistic, linear regression, factor analysis
  evaluate.py    walk-forward training and scoring
  cli.py         command-line entry points
tests/           unit tests for the causal feature logic and splits
reports/         generated metrics and figures
```

## License

See [LICENSE](LICENSE).
