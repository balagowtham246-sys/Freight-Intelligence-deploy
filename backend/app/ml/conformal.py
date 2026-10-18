"""Conformalised quantile regression: making the intervals mean what they say.

Why this is not optional
------------------------
The timing engine does not consume a point forecast. It consumes a
*distribution*, samples scenarios from it, and takes a CVaR over the tail. If
the 90th percentile is really the 75th, every downstream number — the value of
waiting, the regret figure, the contract-mix frontier — is wrong in the
direction that makes the platform look more confident than it is.

Gradient-boosted quantile regression is not calibrated out of the box. It is
fitted by minimising pinball loss on the training set and its nominal
quantiles drift, especially in the tails and especially on short histories.

The method
----------
Split-conformal, in the CQR form of Romano, Patterson and Candès (2019):

1. Fit quantile models on the training fold.
2. On a held-out calibration fold the model never saw, compute the conformity
   score for each interval::

       E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

   Positive when the truth fell outside the interval, negative when inside,
   and its magnitude is how far out.
3. Take the ``ceil((n+1)(1-alpha))/n`` empirical quantile of those scores and
   widen the interval by it.

The guarantee is *distribution-free* and finite-sample: coverage is at least
``1 - alpha`` under exchangeability alone, with no assumption about the error
distribution. That is exactly the right property for a series whose error
distribution is fat-tailed and regime-dependent.

The honest caveat, recorded because it matters here: exchangeability is
violated by a structural break. Conformal intervals stay valid on average over
the calibration period, not conditionally on today's regime. That is why the
regime model runs alongside and why ``MARKET_VOL`` and ``REGIME_SHIFT`` alerts
exist — they tell the user when the assumption is under strain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalAdjustment:
    """Per-interval widening, learned on a calibration fold."""

    # alpha -> additive widening in the target's own units
    widening: dict[float, float] = field(default_factory=dict)
    n_calibration: int = 0

    def as_dict(self) -> dict:
        return {
            "n_calibration": self.n_calibration,
            "widening": {str(k): round(v, 6) for k, v in sorted(self.widening.items())},
        }


def conformity_scores(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """CQR score: how far outside the interval the truth fell.

    Negative means the truth was comfortably inside.
    """
    return np.maximum(lower - y_true, y_true - upper)


def conformal_widening(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-valid widening for a ``1 - alpha`` interval.

    Uses the ``ceil((n+1)(1-alpha))``-th order statistic, which is the
    correction that makes the guarantee exact rather than asymptotic. When the
    calibration fold is too small for that order statistic to exist, the
    interval cannot be validly calibrated and we return the largest score
    seen — deliberately conservative, because under-covering is the failure
    that propagates into bad decisions.
    """
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size
    if n == 0:
        return 0.0

    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float(np.max(clean))
    return float(np.sort(clean)[rank - 1])


def calibrate(
    y_true: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
) -> ConformalAdjustment:
    """Learn the widening for every symmetric interval in the quantile grid.

    ``quantile_predictions`` maps quantile level to predictions on the
    calibration fold. Quantiles are paired symmetrically (0.05 with 0.95, 0.10
    with 0.90, ...); the median is never widened.
    """
    levels = sorted(quantile_predictions)
    adjustment = ConformalAdjustment(n_calibration=int(np.size(y_true)))

    for low in levels:
        high = round(1.0 - low, 6)
        if low >= 0.5 or high not in quantile_predictions:
            continue
        alpha = round(2.0 * low, 6)
        scores = conformity_scores(
            np.asarray(y_true, dtype=float),
            np.asarray(quantile_predictions[low], dtype=float),
            np.asarray(quantile_predictions[high], dtype=float),
        )
        adjustment.widening[alpha] = conformal_widening(scores, alpha)

    return adjustment


def apply(
    quantile_predictions: dict[float, float],
    adjustment: ConformalAdjustment,
    *,
    floor: float | None = 0.0,
) -> dict[float, float]:
    """Widen a set of quantile predictions by the calibrated amounts.

    Also enforces monotonicity. Independently-fitted quantile models can cross
    — the fitted 75th coming out below the fitted 50th — which is nonsense as
    a distribution and would silently corrupt scenario sampling. Sorting the
    widened values is the standard, and cheapest, repair.
    """
    widened: dict[float, float] = {}
    for level, value in quantile_predictions.items():
        if level == 0.5:
            widened[level] = value
            continue
        alpha = round(2.0 * min(level, 1.0 - level), 6)
        delta = adjustment.widening.get(alpha, 0.0)
        widened[level] = value - delta if level < 0.5 else value + delta

    levels = sorted(widened)
    values = np.sort(np.array([widened[level] for level in levels], dtype=float))
    if floor is not None:
        values = np.maximum(values, floor)
    return {level: float(value) for level, value in zip(levels, values)}


def empirical_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Share of truths that landed inside the interval. The calibration check."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower, dtype=float)) & (
        y_true <= np.asarray(upper, dtype=float)
    )
    valid = np.isfinite(y_true)
    return float(inside[valid].mean()) if valid.any() else float("nan")
