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

# Write all metrics CSVs and figures to reports/
python -m f1predictor.cli report

# Predict a single upcoming race
python -m f1predictor.cli predict --season 2026 --round 9
```

## Results

Real timing data for 2023, 2024, 2025, and the partial 2026 season (five rounds
run so far). Run `f1predict report` to regenerate the tables and figures in
`reports/`.

| Train on            | Predict | Winner acc | Podium prec@3 | Pole baseline | Log-loss |
| ------------------- | ------- | ---------- | ------------- | ------------- | -------- |
| 2023                | 2024    | 0.458      | 0.625         | 0.500         | 0.264    |
| 2023, 2024          | 2025    | 0.333      | 0.736         | 0.667         | 0.882    |
| 2023, 2024, 2025    | 2026*   | 0.400      | 0.533         | 0.800         | 0.258    |

*2026 is a partial season (five rounds).

The model lands around 40 to 46% race-winner accuracy and 53 to 74% podium
precision, in line with the difficulty of the task. The honest headline is that
**the pole-sitter baseline is very strong**: pole position converts to a win in
half to four-fifths of races over this period, and the model is competitive with
it rather than dominant. That is the expected result for modern F1, where grid
position is overwhelmingly predictive.

### What the model learns

LASSO drives most coefficients to zero and keeps a sparse, sensible set
(standardized, most recent split):

| Feature              | Coefficient |
| -------------------- | ----------- |
| driver_form_points   | +1.01       |
| grid_position        | -0.84       |
| quali_position       | -0.84       |
| driver_track_finish  | -0.06       |
| (all others)         | 0           |

Better recent form raises win probability; a better (lower) grid/qualifying
position raises it too. Everything else is shrunk out.

### Latent structure (factor analysis / PCA)

Factor analysis of the features recovers four interpretable factors: a
competitiveness/form factor (grid, qualifying, recent and team form), a
temperature factor (air and track temperature against humidity and rain), a wind
factor, and a reliability factor. The first two principal components explain
about 57% of the variance and the first four about 74%.

## Limitations

Honest constraints worth stating, several forced by data availability:

- **Grid uses qualifying as a proxy.** With Ergast/Jolpica unreachable, grid
  position is taken from the qualifying classification, so grid penalties are
  not reflected. Grid and qualifying are therefore identical columns here, which
  is why they receive identical coefficients and factor loadings.
- **Points exclude sprint and fastest-lap bonuses** (computed from the standard
  finishing-position table), so season point totals run slightly below official.
- **Reliability is a DNF-rate proxy**, not literal pit-stop counts (the timing
  feed does not expose clean pit data without extra parsing).
- **The pole baseline is hard to beat**, as the results above show.

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
