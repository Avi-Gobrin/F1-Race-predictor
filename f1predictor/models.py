"""Models and cross-validation.

The statistical toolbox, kept deliberately close to an undergraduate regression
and multivariate-analysis course:

* **LASSO logistic regression** for the win/podium classifier. The L1 penalty
  performs automatic feature selection; its strength is chosen by cross-
  validation (leave-one-race-out by default, classic leave-one-out optional).
* **LASSO / ordinary linear regression** on finishing position, an interpretable
  regression baseline that also exposes which predictors survive shrinkage.
* **Factor analysis and PCA** to study the latent structure of the correlated
  form features (recent points, recent finish, track history all move together).

All estimators sit behind a shared preprocessing pipeline (median imputation
then standardization). Imputation and scaling are fit *inside* each CV fold, so
no information leaks from validation rows into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_MODEL_CONFIG, ModelConfig
from .features import FEATURE_COLUMNS


def _preprocessing_steps() -> list[tuple[str, object]]:
    """Median imputation followed by standardization (shared by all models).

    ``keep_empty_features`` matters for the first walk-forward split: when
    training on 2023 alone, ``driver_track_finish`` is entirely missing (every
    circuit has been visited at most once, so there is no prior history yet).
    Keeping the column (filled with 0, then standardized to 0) preserves a
    stable 13-feature design matrix across all splits.
    """
    return [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]


def _make_cv(cfg: ModelConfig):
    """Return a cross-validation splitter for the requested scheme."""
    if cfg.cv_scheme == "logo":
        return LeaveOneGroupOut()
    if cfg.cv_scheme == "loo":
        return LeaveOneOut()
    raise ValueError(f"unknown cv_scheme {cfg.cv_scheme!r}")


# ---------------------------------------------------------------------------
# Win / podium classifier: LASSO logistic regression
# ---------------------------------------------------------------------------
@dataclass
class FittedClassifier:
    """A fitted win/podium classifier plus the penalty chosen by CV."""

    search: GridSearchCV
    best_C: float
    cv_log_loss: float

    @property
    def estimator(self) -> Pipeline:
        return self.search.best_estimator_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class (win or podium)."""
        return self.estimator.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def coefficients(self) -> pd.Series:
        """LASSO coefficients on the standardized features (sparse by design)."""
        clf = self.estimator.named_steps["clf"]
        return pd.Series(clf.coef_.ravel(), index=FEATURE_COLUMNS).sort_values(
            key=np.abs, ascending=False
        )


def fit_win_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> FittedClassifier:
    """Fit a LASSO logistic classifier, selecting C by cross-validation.

    Scored by negative log-loss because we care about calibrated win
    probabilities, not just the hard label of a heavily imbalanced target
    (only one winner per ~20 drivers).
    """
    pipe = Pipeline(
        _preprocessing_steps()
        + [
            (
                "clf",
                # L1 (LASSO) logistic regression. In scikit-learn 1.8 a pure L1
                # penalty is expressed as l1_ratio=1.0 and requires the saga
                # solver; class_weight balances the rare winner label.
                LogisticRegression(
                    solver="saga",
                    l1_ratio=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                ),
            )
        ]
    )
    grid = {"clf__C": list(cfg.lasso_C_grid)}
    cv = _make_cv(cfg)
    search = GridSearchCV(
        pipe,
        grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    fit_kwargs = {}
    if isinstance(cv, LeaveOneGroupOut):
        fit_kwargs["groups"] = groups
    search.fit(X[FEATURE_COLUMNS], y, **fit_kwargs)
    return FittedClassifier(
        search=search,
        best_C=float(search.best_params_["clf__C"]),
        cv_log_loss=float(-search.best_score_),
    )


# ---------------------------------------------------------------------------
# Finishing-position regression (interpretable baseline)
# ---------------------------------------------------------------------------
@dataclass
class FittedRegressor:
    pipeline: Pipeline
    kind: str

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[FEATURE_COLUMNS])

    def coefficients(self) -> pd.Series:
        reg = self.pipeline.named_steps["reg"]
        return pd.Series(np.ravel(reg.coef_), index=FEATURE_COLUMNS).sort_values(
            key=np.abs, ascending=False
        )


def fit_position_regressor(
    X: pd.DataFrame,
    y_position: pd.Series,
    cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
    *,
    lasso: bool = True,
    alpha: float = 0.1,
) -> FittedRegressor:
    """Regress finishing position on the engineered features.

    With ``lasso=True`` this is an L1-penalized linear regression that drives
    weak predictors to exactly zero; with ``lasso=False`` it is ordinary least
    squares. Useful as an interpretable sanity check on the classifier.
    """
    reg = Lasso(alpha=alpha, max_iter=5000) if lasso else LinearRegression()
    pipe = Pipeline(_preprocessing_steps() + [("reg", reg)])
    pipe.fit(X[FEATURE_COLUMNS], y_position.astype(float))
    return FittedRegressor(pipeline=pipe, kind="lasso" if lasso else "ols")


# ---------------------------------------------------------------------------
# Latent structure: factor analysis and PCA
# ---------------------------------------------------------------------------
def _prepared_matrix(X: pd.DataFrame) -> np.ndarray:
    """Impute and standardize a feature matrix for unsupervised analysis."""
    pipe = Pipeline(_preprocessing_steps())
    return pipe.fit_transform(X[FEATURE_COLUMNS])


def factor_analysis(
    X: pd.DataFrame, n_factors: int = DEFAULT_MODEL_CONFIG.n_factors
) -> pd.DataFrame:
    """Factor-analysis loadings of the engineered features.

    Returns a (feature x factor) loadings table. Uses ``factor_analyzer`` when
    available (varimax rotation for interpretability) and falls back to a PCA
    approximation otherwise so the pipeline never hard-fails on a missing
    optional dependency.
    """
    Z = _prepared_matrix(X)
    cols = [f"Factor{i + 1}" for i in range(n_factors)]
    try:
        from factor_analyzer import FactorAnalyzer

        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
        fa.fit(Z)
        loadings = fa.loadings_
    except Exception:  # pragma: no cover - optional dependency / degenerate data
        pca = PCA(n_components=n_factors)
        pca.fit(Z)
        loadings = pca.components_.T
    return pd.DataFrame(loadings, index=FEATURE_COLUMNS, columns=cols)


def pca_explained_variance(
    X: pd.DataFrame, n_components: int = DEFAULT_MODEL_CONFIG.n_factors
) -> pd.Series:
    """Explained-variance ratio of the leading principal components."""
    Z = _prepared_matrix(X)
    n_components = min(n_components, Z.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(Z)
    return pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i + 1}" for i in range(n_components)],
        name="explained_variance_ratio",
    )
