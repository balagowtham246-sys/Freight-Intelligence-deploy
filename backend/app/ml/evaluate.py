"""Walk-forward evaluation: the part that decides whether to trust anything.

This module is the platform's acceptance gate. `MarketCockpit.tsx` promised
that no forecast ships until it has been *"measured honestly against a
random-walk baseline"*, and this is where that measurement happens.

The protocol
------------
Strictly walk-forward, with an embargo:

    |-------------- train --------------|--embargo--|--test--|
                                          h days

The embargo is exactly the forecast horizon. Without it the last training
label overlaps the first test window and the two share a future — a subtle
leak that inflates skill at long horizons precisely where skill is weakest.

No random splits, ever. Neighbouring days in a near-unit-root series are
almost the same observation, so a shuffled split trains on the answer.

The metrics
-----------
* **Pinball loss** — the proper scoring rule for a quantile. Averaged over the
  grid it is the discrete CRPS, so one computation gives both.
* **CRPS** — the whole distribution scored against the realised point.
* **MASE** — mean absolute scaled error, scaled by the in-sample naive
  change. Below 1.0 means the median beat a random walk.
* **Empirical coverage** — share of truths inside the 80% band, which should
  be 0.80. This is the check that the conformal calibration actually worked.
* **PIT histogram** — where in its own predicted distribution each realised
  value fell. Flat is calibrated; a U-shape means intervals too narrow; a
  hump means too wide.

Skill, and the "no edge" rule
-----------------------------
``skill = 1 - loss / baseline_loss``. Positive means the model beat the
baseline. Zero or negative means it did not, and the platform is required to
say so rather than emit a confident number — see ``signal_strength`` in the
timing engine, which reads these rows.

The benchmark restriction
-------------------------
``assert_benchmark_allowed`` refuses to score a forecast against a benchmark
derived from disclosed synthetic data. Beating a curve we generated ourselves
proves nothing, and SYNTHETIC_DATA.md commits the platform to reporting
"benchmark unavailable" instead. Baselines computed *from the target series
itself* (random walk, seasonal naive) remain valid on a synthetic target —
they measure whether the model extracts more than persistence from whatever
series it was given, which is a real question either way. What is forbidden is
comparing against a separately fabricated forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .baselines import BASELINES, mase_scale
from .quantile import QUANTILES, fit_horizon, predict_horizon

# Baselines derived from the target's own history. Always legitimate.
SELF_REFERENTIAL_BASELINES = frozenset({"random_walk", "drift", "seasonal_naive"})

# Minimum number of walk-forward folds before a skill number is reportable.
MIN_FOLDS = 5

PIT_BINS = 10


class BenchmarkUnavailable(RuntimeError):
    """Raised when a forbidden benchmark comparison is attempted."""


def assert_benchmark_allowed(baseline_name: str, target_series: str) -> None:
    """Refuse to benchmark against a fabricated forecast.

    See the module docstring and SYNTHETIC_DATA.md entry 2.
    """
    if baseline_name in SELF_REFERENTIAL_BASELINES:
        return
    if target_series.startswith("SYN_") or baseline_name.startswith("SYN_"):
        raise BenchmarkUnavailable(
            f"refusing to benchmark {target_series!r} against {baseline_name!r}: "
            "a benchmark derived from synthetic data proves nothing. "
            "Report 'benchmark unavailable' instead."
        )


def pinball_loss(y_true: float, predicted: float, level: float) -> float:
    """Quantile (pinball) loss for one prediction at one level."""
    error = y_true - predicted
    return float(max(level * error, (level - 1.0) * error))


def crps_from_quantiles(y_true: float, quantiles: dict[float, float]) -> float:
    """Discrete CRPS: mean pinball loss across the quantile grid.

    Twice the average pinball loss over a uniform quantile grid approximates
    the CRPS integral; we report the average itself, which is monotone in the
    same quantity and directly comparable between models.
    """
    if not quantiles or not np.isfinite(y_true):
        return float("nan")
    losses = [pinball_loss(y_true, value, level) for level, value in quantiles.items()]
    return float(np.mean(losses))


def pit_value(y_true: float, quantiles: dict[float, float]) -> float:
    """Where the realised value fell in its own predicted distribution.

    Interpolated between the bracketing quantiles, so the histogram is not
    artificially lumpy at the grid points.
    """
    levels = sorted(quantiles)
    values = [quantiles[level] for level in levels]
    if not np.isfinite(y_true) or not values:
        return float("nan")
    if y_true <= values[0]:
        return float(levels[0])
    if y_true >= values[-1]:
        return float(levels[-1])
    return float(np.interp(y_true, values, levels))


@dataclass
class FoldResult:
    """One walk-forward fold at one horizon."""

    origin_date: date
    target_date: date
    horizon_days: int
    actual: float
    quantiles: dict[float, float]
    baseline: dict[str, float] = field(default_factory=dict)


@dataclass
class HorizonMetrics:
    """Aggregated performance at one horizon."""

    target_key: str
    horizon_days: int
    n_folds: int
    crps: float
    mae: float
    mase: float
    coverage_80: float
    pit_histogram: list[float]
    baseline_crps: dict[str, float] = field(default_factory=dict)
    baseline_mae: dict[str, float] = field(default_factory=dict)
    computed_on_synthetic: bool = False

    def skill_vs(self, baseline_name: str) -> float | None:
        """Distributional skill: 1 - CRPS/baseline_CRPS.

        Read this carefully, because it flatters the model. A point baseline
        has to be scored as a degenerate distribution — the same value at every
        quantile — and a proper scoring rule punishes that hard. So a positive
        number here largely says "we supply an interval and the baseline does
        not", which is worth something but is *not* the claim "our central
        estimate is better". For that, see ``point_skill_vs``.
        """
        reference = self.baseline_crps.get(baseline_name)
        if reference is None or not np.isfinite(reference) or reference <= 0:
            return None
        return float(1.0 - self.crps / reference)

    def point_skill_vs(self, baseline_name: str) -> float | None:
        """Like-for-like skill: our median against the baseline's point forecast.

        This is the comparison a chartering desk actually cares about, and it
        is the one that decides ``has_edge``.
        """
        reference = self.baseline_mae.get(baseline_name)
        if reference is None or not np.isfinite(reference) or reference <= 0:
            return None
        return float(1.0 - self.mae / reference)

    @property
    def best_skill(self) -> float | None:
        """Distributional skill against the hardest baseline."""
        skills = [
            self.skill_vs(name)
            for name in self.baseline_crps
            if self.skill_vs(name) is not None
        ]
        return min(skills) if skills else None

    @property
    def best_point_skill(self) -> float | None:
        """Point skill against the hardest baseline. The honest headline."""
        skills = [
            self.point_skill_vs(name)
            for name in self.baseline_mae
            if self.point_skill_vs(name) is not None
        ]
        return min(skills) if skills else None

    @property
    def has_edge(self) -> bool:
        """Whether the platform is allowed to present this as a forecast.

        The bar is deliberately strict and is met only when *all* of:

        * enough folds for the number to mean anything;
        * the median beats every baseline's point forecast (``point_skill``);
        * the distribution beats every baseline under a proper score.

        Requiring the point skill is what stops the platform claiming an edge
        on the strength of merely having drawn an interval.
        """
        point = self.best_point_skill
        distributional = self.best_skill
        return (
            self.n_folds >= MIN_FOLDS
            and point is not None
            and point > 0.0
            and distributional is not None
            and distributional > 0.0
        )

    def as_dict(self) -> dict:
        return {
            "target_key": self.target_key,
            "horizon_days": self.horizon_days,
            "n_folds": self.n_folds,
            "crps": round(self.crps, 4),
            "mae": round(self.mae, 4),
            "mase": round(self.mase, 4) if np.isfinite(self.mase) else None,
            "coverage_80": round(self.coverage_80, 4),
            "pit_histogram": [round(value, 4) for value in self.pit_histogram],
            "baselines": {
                name: {
                    "crps": round(value, 4),
                    "mae": (
                        round(self.baseline_mae[name], 4)
                        if name in self.baseline_mae
                        else None
                    ),
                    "skill": (
                        round(self.skill_vs(name), 4)
                        if self.skill_vs(name) is not None
                        else None
                    ),
                    "point_skill": (
                        round(self.point_skill_vs(name), 4)
                        if self.point_skill_vs(name) is not None
                        else None
                    ),
                }
                for name, value in self.baseline_crps.items()
            },
            "best_skill": (
                round(self.best_skill, 4) if self.best_skill is not None else None
            ),
            "best_point_skill": (
                round(self.best_point_skill, 4)
                if self.best_point_skill is not None
                else None
            ),
            "has_edge": self.has_edge,
            "computed_on_synthetic": self.computed_on_synthetic,
        }


def walk_forward(
    design: pd.DataFrame,
    target: pd.Series,
    *,
    target_key: str,
    horizon_days: int,
    n_folds: int = 8,
    step_days: int = 21,
    min_train_days: int = 400,
    quantiles: tuple[float, ...] = QUANTILES,
    baselines: tuple[str, ...] = ("random_walk", "seasonal_naive"),
) -> HorizonMetrics | None:
    """Run a walk-forward backtest at one horizon.

    Each fold refits from scratch on data up to its own cut date, with an
    embargo of ``horizon_days`` before the evaluation point. Refitting per
    fold is slow and is the only way the number means anything.
    """
    for name in baselines:
        assert_benchmark_allowed(name, target_key)

    observed = target.dropna()
    if len(observed) < min_train_days + horizon_days + n_folds * step_days:
        return None

    last_index = len(observed) - 1
    folds: list[FoldResult] = []

    for fold in range(n_folds):
        # Walk backwards from the end so the most recent data is always tested.
        evaluate_at = last_index - fold * step_days
        origin_at = evaluate_at - horizon_days
        if origin_at <= min_train_days:
            break

        origin_stamp = observed.index[origin_at]
        target_stamp = observed.index[evaluate_at]
        actual = float(observed.iloc[evaluate_at])

        # Embargo: the model may see nothing at or after the origin date. The
        # label of the last training row therefore ends before the origin.
        train_design = design.loc[design.index <= origin_stamp]
        train_target = target.loc[target.index <= origin_stamp]

        fitted = fit_horizon(
            train_design, train_target, horizon_days, quantiles=quantiles
        )
        if fitted is None:
            continue

        prediction = predict_horizon(
            fitted,
            train_design,
            target_key=target_key,
            origin_date=origin_stamp.date(),
        )

        history = train_target.dropna()
        baseline_values = {
            name: float(BASELINES[name](history, horizon_days)) for name in baselines
        }

        folds.append(
            FoldResult(
                origin_date=origin_stamp.date(),
                target_date=target_stamp.date(),
                horizon_days=horizon_days,
                actual=actual,
                quantiles=prediction.quantiles,
                baseline=baseline_values,
            )
        )

    if not folds:
        return None

    return aggregate(folds, target_key=target_key, target=target)


def aggregate(
    folds: list[FoldResult], *, target_key: str, target: pd.Series
) -> HorizonMetrics:
    """Turn fold results into reportable metrics."""
    actuals = np.array([fold.actual for fold in folds], dtype=float)
    medians = np.array([fold.quantiles.get(0.5, np.nan) for fold in folds], dtype=float)

    crps = float(
        np.nanmean([crps_from_quantiles(f.actual, f.quantiles) for f in folds])
    )
    mae = float(np.nanmean(np.abs(actuals - medians)))

    # MASE must be scaled by the naive error at the *same* horizon. Scaling an
    # h-step forecast by the mean absolute one-day change would make MASE
    # enormous at every horizon and mean nothing.
    horizon = folds[0].horizon_days
    scale = mase_scale(target.dropna(), season_days=horizon)
    mase = mae / scale if np.isfinite(scale) and scale > 0 else float("nan")

    lower = np.array([fold.quantiles.get(0.10, np.nan) for fold in folds], dtype=float)
    upper = np.array([fold.quantiles.get(0.90, np.nan) for fold in folds], dtype=float)
    inside = (actuals >= lower) & (actuals <= upper)
    coverage = float(np.mean(inside)) if inside.size else float("nan")

    pits = np.array([pit_value(f.actual, f.quantiles) for f in folds], dtype=float)
    pits = pits[np.isfinite(pits)]
    histogram, _ = np.histogram(pits, bins=PIT_BINS, range=(0.0, 1.0))
    normalised = (
        (histogram / histogram.sum()).tolist()
        if histogram.sum()
        else [0.0] * PIT_BINS
    )

    baseline_names = sorted({name for fold in folds for name in fold.baseline})
    baseline_crps: dict[str, float] = {}
    baseline_mae: dict[str, float] = {}
    for name in baseline_names:
        predictions = np.array(
            [fold.baseline.get(name, np.nan) for fold in folds], dtype=float
        )
        # A point baseline is scored as a degenerate distribution: the same
        # value at every quantile. That is the fair comparison - it is exactly
        # what "the market gives you no interval" costs under a proper score.
        losses = [
            crps_from_quantiles(actual, {level: prediction for level in QUANTILES})
            for actual, prediction in zip(actuals, predictions)
        ]
        baseline_crps[name] = float(np.nanmean(losses))
        baseline_mae[name] = float(np.nanmean(np.abs(actuals - predictions)))

    return HorizonMetrics(
        target_key=target_key,
        horizon_days=folds[0].horizon_days,
        n_folds=len(folds),
        crps=crps,
        mae=mae,
        mase=mase,
        coverage_80=coverage,
        pit_histogram=normalised,
        baseline_crps=baseline_crps,
        baseline_mae=baseline_mae,
        computed_on_synthetic=target_key.startswith("SYN_"),
    )
