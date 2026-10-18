"""Regime identification: Markov switching and change-point detection.

Why the platform needs this
---------------------------
Every statistical guarantee the forecaster offers assumes the future looks
like the calibration period. Conformal intervals are valid under
exchangeability; a boosted model extrapolates the relationships it was
trained on. A structural break — Red Sea rerouting, a Panama drought, an
Indonesian export ban, a new Indian import duty — violates that assumption
silently, and the model keeps emitting confident numbers.

So the platform runs a regime model alongside, and its job is not to improve
the forecast. Its job is to say *when to distrust it*. That is what the
``REGIME_SHIFT`` alert reports, and it is why the timing engine downgrades
``signal_strength`` when a shift is fresh.

Two estimators, because they fail differently:

* **Markov switching** (``statsmodels.MarkovRegression``) on log returns with
  switching variance. Gives a smoothed probability of being in the
  high-volatility state, which is a graded, well-behaved signal — but it is
  fitted on the whole sample and is therefore slow to acknowledge something
  genuinely new.
* **CUSUM** on standardised returns. Crude, online, and fast. It detects a
  mean shift within days, with no distributional assumption beyond a scale
  estimate. It is noisier, and that is the trade.

Agreement between them is a strong signal. Disagreement is worth surfacing
rather than resolving.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

MIN_OBSERVATIONS = 250

# CUSUM decision interval, in standard deviations of the standardised series.
# 5.0 is the conventional choice: roughly a 1-in-500 false-alarm rate per
# observation under a stable mean, which on daily data is about two spurious
# signals a year. Lower cries wolf; higher misses the break that matters.
CUSUM_THRESHOLD = 5.0
CUSUM_SLACK = 0.5


@dataclass
class RegimeState:
    """Where the market is, and how confident we are that it just moved."""

    regime: str  # calm | normal | turbulent
    probability_turbulent: float
    changed_recently: bool
    days_since_change: int | None
    last_change_date: date | None
    detector: str
    n_observations: int

    def as_dict(self) -> dict:
        return {
            "regime": self.regime,
            "probability_turbulent": round(self.probability_turbulent, 4),
            "changed_recently": self.changed_recently,
            "days_since_change": self.days_since_change,
            "last_change_date": (
                self.last_change_date.isoformat() if self.last_change_date else None
            ),
            "detector": self.detector,
            "n_observations": self.n_observations,
        }


def cusum_changepoints(
    values: pd.Series,
    *,
    threshold: float = CUSUM_THRESHOLD,
    slack: float = CUSUM_SLACK,
    window: int = 60,
) -> list[date]:
    """Two-sided CUSUM on standardised values; returns the dates it fired.

    Standardisation uses a trailing window so the detector adapts to the
    series' own scale rather than assuming one. The accumulators reset on
    each detection, which is what makes it an online detector rather than a
    retrospective one.
    """
    clean = values.dropna()
    if len(clean) < window + 10:
        return []

    rolling_mean = clean.rolling(window, min_periods=window // 2).mean()
    rolling_std = clean.rolling(window, min_periods=window // 2).std()
    standardised = ((clean - rolling_mean) / rolling_std.where(rolling_std > 0)).dropna()

    positive = negative = 0.0
    detections: list[date] = []
    for stamp, value in standardised.items():
        positive = max(0.0, positive + value - slack)
        negative = max(0.0, negative - value - slack)
        if positive > threshold or negative > threshold:
            detected = stamp.date() if hasattr(stamp, "date") else stamp
            detections.append(detected)
            positive = negative = 0.0

    return detections


def markov_regime(returns: pd.Series) -> tuple[float, str] | None:
    """Smoothed probability of the high-variance state, and a label.

    Two states with switching variance: dry bulk regimes differ far more in
    dispersion than in mean, so switching the variance is the specification
    that matches the phenomenon.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    clean = returns.dropna()
    clean = clean[np.isfinite(clean)]
    if len(clean) < MIN_OBSERVATIONS:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(
                clean.to_numpy() * 100.0, k_regimes=2, switching_variance=True
            )
            result = model.fit(disp=False)
    except Exception:
        return None

    try:
        variances = [
            float(result.params[f"sigma2[{state}]"]) for state in range(2)
        ]
    except (KeyError, IndexError):
        return None

    turbulent_state = int(np.argmax(variances))
    probability = float(result.smoothed_marginal_probabilities[turbulent_state][-1])

    if probability >= 0.70:
        label = "turbulent"
    elif probability <= 0.30:
        label = "calm"
    else:
        label = "normal"
    return probability, label


def regime_state(
    returns: pd.Series, *, recent_days: int = 21
) -> RegimeState | None:
    """Combine the two detectors into one reportable state.

    The Markov model supplies the level; CUSUM supplies the timing. Where the
    Markov fit fails - which it does on short or degenerate samples - CUSUM
    alone still gives a usable answer, and the ``detector`` field says which
    produced the result so nothing is over-claimed.
    """
    clean = returns.dropna()
    if clean.empty:
        return None

    changes = cusum_changepoints(clean)
    last_change = changes[-1] if changes else None

    latest_index = clean.index[-1]
    latest = latest_index.date() if hasattr(latest_index, "date") else latest_index
    days_since = (latest - last_change).days if last_change else None
    changed_recently = days_since is not None and days_since <= recent_days

    markov = markov_regime(clean)
    if markov is not None:
        probability, label = markov
        detector = "markov+cusum"
    else:
        # Fall back to a dispersion percentile: crude, but honest and labelled.
        realised = clean.rolling(20, min_periods=10).std().dropna()
        if realised.empty:
            return None
        probability = float((realised <= realised.iloc[-1]).mean())
        label = (
            "turbulent" if probability >= 0.80
            else "calm" if probability <= 0.20
            else "normal"
        )
        detector = "cusum+dispersion"

    return RegimeState(
        regime=label,
        probability_turbulent=probability,
        changed_recently=changed_recently,
        days_since_change=days_since,
        last_change_date=last_change,
        detector=detector,
        n_observations=int(len(clean)),
    )
