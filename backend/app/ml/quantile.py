"""Direct multi-horizon quantile forecasting with LightGBM.

Design choices, and why
-----------------------
**Direct, not recursive.** One model per (horizon, quantile) predicting
*h* days ahead from today's features, rather than one one-step model iterated.
Recursive forecasting compounds its own errors and, worse, gives no honest way
to widen intervals with horizon. Direct costs more models and is the right
trade.

**Quantiles, not points.** ``objective="quantile"`` with ``alpha`` set per
level. The downstream consumer is an optimiser taking a CVaR over a tail; a
point forecast cannot feed it.

**Predict the log level, return the level.** Freight is lognormal-ish and
strictly positive. Fitting in log space keeps the intervals positive without
clipping and makes the error distribution closer to symmetric, which is what
the pinball objective assumes.

**Small-data discipline.** With a short history the model must be small or it
will memorise. The defaults below are deliberately restrictive — shallow
trees, high leaf minimums, strong bagging — and ``MIN_TRAIN_ROWS`` refuses to
fit at all below a floor rather than returning a confident-looking curve
fitted on nothing.

The SARIMAX second opinion lives in ``sarimax_forecast``: interpretable,
cheap, and a useful disagreement signal when the boosted model wanders.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from .conformal import ConformalAdjustment, apply as apply_conformal, calibrate

QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# Horizons in days. The strategy document specifies 1/2/4/8/13/26 weeks; the
# grid is config-driven so a run uses only what the data actually supports.
HORIZONS_DAYS: tuple[int, ...] = (7, 14, 28, 56, 91, 182)

# Below this many usable training rows, refuse to fit. A model fitted on 40
# points will produce a beautiful, meaningless fan chart.
MIN_TRAIN_ROWS = 250

# Share of the training window held out, unseen, for conformal calibration.
CALIBRATION_FRACTION = 0.25
MIN_CALIBRATION_ROWS = 40

LGB_PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_estimators": 250,
}


@dataclass
class QuantileForecast:
    """The predictive distribution of one target at one horizon."""

    target_key: str
    origin_date: date
    horizon_days: int
    target_date: date
    quantiles: dict[float, float]
    model_name: str = "lgbm_quantile"
    model_version: str = "1"
    n_train: int = 0
    conformal: dict | None = None

    @property
    def median(self) -> float:
        return self.quantiles.get(0.5, float("nan"))

    def as_dict(self) -> dict:
        return {
            "target_key": self.target_key,
            "origin_date": self.origin_date.isoformat(),
            "horizon_days": self.horizon_days,
            "target_date": self.target_date.isoformat(),
            "median": round(self.median, 4),
            "quantiles": {str(k): round(v, 4) for k, v in sorted(self.quantiles.items())},
            "n_train": self.n_train,
        }


@dataclass
class FittedHorizon:
    """Fitted models plus their calibration, for one horizon."""

    horizon_days: int
    models: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)
    adjustment: ConformalAdjustment = field(default_factory=ConformalAdjustment)
    feature_names: list[str] = field(default_factory=list)
    n_train: int = 0


def _training_frame(
    design: pd.DataFrame, target: pd.Series, horizon_days: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features to the h-ahead label and drop rows without a label.

    The final ``horizon_days`` rows have no label by construction, which is
    what stops the model training on a future it could not have seen.
    """
    label = target.shift(-horizon_days)
    frame = design.copy()
    frame["__label__"] = label
    frame = frame[frame["__label__"].notna()]
    labels = frame.pop("__label__")
    return frame, labels


def fit_horizon(
    design: pd.DataFrame,
    target: pd.Series,
    horizon_days: int,
    *,
    quantiles: tuple[float, ...] = QUANTILES,
    params: dict | None = None,
) -> FittedHorizon | None:
    """Fit every quantile at one horizon, then calibrate on a held-out tail.

    Returns ``None`` when there is not enough data to fit honestly.
    """
    features, labels = _training_frame(design, target, horizon_days)
    if len(features) < MIN_TRAIN_ROWS:
        return None

    # The calibration fold is the *most recent* block, not a random sample.
    # A random split would leak: neighbouring days in a near-unit-root series
    # are almost the same observation.
    n_calibration = max(
        MIN_CALIBRATION_ROWS, int(len(features) * CALIBRATION_FRACTION)
    )
    if len(features) - n_calibration < MIN_TRAIN_ROWS // 2:
        n_calibration = max(MIN_CALIBRATION_ROWS, len(features) // 5)

    train_features = features.iloc[:-n_calibration]
    train_labels = labels.iloc[:-n_calibration]
    calibration_features = features.iloc[-n_calibration:]
    calibration_labels = labels.iloc[-n_calibration:]

    if len(train_features) < MIN_CALIBRATION_ROWS:
        return None

    # Fit in log space: freight is strictly positive and lognormal-ish.
    log_train_labels = np.log(train_labels.where(train_labels > 0))
    usable = log_train_labels.notna()
    train_features = train_features[usable]
    log_train_labels = log_train_labels[usable]
    if len(train_features) < MIN_CALIBRATION_ROWS:
        return None

    fitted = FittedHorizon(
        horizon_days=horizon_days,
        feature_names=[str(column) for column in features.columns],
        n_train=int(len(train_features)),
    )

    settings = {**LGB_PARAMS, **(params or {})}
    for level in quantiles:
        model = lgb.LGBMRegressor(**{**settings, "alpha": level})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_features, log_train_labels)
        fitted.models[level] = model

    # Calibrate on the untouched tail, in the target's own units.
    predictions = {
        level: np.exp(model.predict(calibration_features))
        for level, model in fitted.models.items()
    }
    fitted.adjustment = calibrate(
        calibration_labels.to_numpy(dtype=float), predictions
    )
    return fitted


def predict_horizon(
    fitted: FittedHorizon,
    latest_features: pd.DataFrame,
    *,
    target_key: str,
    origin_date: date,
) -> QuantileForecast:
    """Predict the calibrated quantiles for one horizon from the newest row."""
    row = latest_features.iloc[[-1]][fitted.feature_names]
    raw = {
        level: float(np.exp(model.predict(row)[0]))
        for level, model in fitted.models.items()
    }
    calibrated = apply_conformal(raw, fitted.adjustment, floor=0.0)

    return QuantileForecast(
        target_key=target_key,
        origin_date=origin_date,
        horizon_days=fitted.horizon_days,
        target_date=origin_date + timedelta(days=fitted.horizon_days),
        quantiles=calibrated,
        n_train=fitted.n_train,
        conformal=fitted.adjustment.as_dict(),
    )


def fit_and_predict(
    design: pd.DataFrame,
    target: pd.Series,
    *,
    target_key: str,
    origin_date: date,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    quantiles: tuple[float, ...] = QUANTILES,
) -> tuple[list[QuantileForecast], list[int]]:
    """Fit and predict across the horizon grid.

    Returns the forecasts produced and the horizons that were skipped for want
    of data, so the caller can report the gap rather than hide it.
    """
    forecasts: list[QuantileForecast] = []
    skipped: list[int] = []

    for horizon in horizons:
        fitted = fit_horizon(design, target, horizon, quantiles=quantiles)
        if fitted is None:
            skipped.append(horizon)
            continue
        forecasts.append(
            predict_horizon(
                fitted, design, target_key=target_key, origin_date=origin_date
            )
        )
    return forecasts, skipped


def sarimax_forecast(
    target: pd.Series, horizon_days: int, *, order=(1, 1, 1)
) -> float | None:
    """Interpretable second opinion, fitted on the log level.

    Deliberately plain. Its job is to disagree usefully with the boosted
    model, not to win; a large divergence is itself a signal that the feature
    set is driving the forecast somewhere the univariate dynamics do not
    support.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    clean = target.dropna()
    clean = clean[clean > 0]
    if len(clean) < 120:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                np.log(clean.to_numpy()),
                order=order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            return float(np.exp(result.forecast(steps=horizon_days)[-1]))
    except Exception:
        return None
