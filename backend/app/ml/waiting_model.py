"""Pre-berthing detention: a discrete-time hazard model.

Why a hazard model and not a regression
---------------------------------------
19% of the port calls in the archive are censored — still at anchorage when
the archive ends, or gone from the queue without ever being seen alongside.
A regression on ``waiting_days`` has three options for those rows and all
three are wrong: drop them (biases short, because long waits are exactly the
ones still unresolved), treat the bound as the answer (biases short again), or
impute (invents data). A hazard model uses them correctly — a censored call
contributes everything it genuinely tells us, namely that the vessel was still
waiting on each day up to the bound.

The formulation
---------------
Each port call is expanded into one row per day at anchorage::

    day 1: still waiting?  no  -> berthed=1, stop
    day 2: ...

and a logistic regression estimates the daily hazard, the probability of
berthing tomorrow given still waiting today. Censored calls contribute rows up
to their bound with ``berthed=0`` and simply stop. No special-casing.

The survival curve, and hence the full distribution of waiting time, follows
from the fitted hazards:

    S(t) = prod_{k<=t} (1 - h(k))

This is the standard person-period formulation. It handles censoring exactly,
needs no proportional-hazards assumption, and — the reason it is right *here*
— is well-posed on a few hundred observations, where fitting a parametric
survival distribution or a Cox model with time-varying covariates would not be.

Why this model matters commercially
------------------------------------
It is the most forecastable quantity in the platform. Queueing behaviour is
far more predictable than freight rates, and it feeds the demurrage term in
the timing engine, which is often the difference between two routings that
look identical on freight alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Days beyond which we stop modelling. Waits longer than this are rare enough
# that the hazard estimate is noise, and the survival curve is reported as
# censored past the horizon rather than extrapolated.
MAX_WAIT_DAYS = 45

MIN_CALLS = 40

# Berth dates recovered across a gap in the snapshot archive are upper bounds.
# Beyond this many days of uncertainty the observation is excluded from
# training rather than blessed with a precision the archive cannot support.
MAX_UNCERTAIN_DAYS = 3


@dataclass
class WaitingForecast:
    """The predicted distribution of pre-berthing detention for one call."""

    port_code: str
    queue_length: float
    expected_days: float
    median_days: float
    p80_days: float
    p95_days: float
    survival: list[float] = field(default_factory=list)
    model_name: str = "discrete_hazard"
    n_train_calls: int = 0
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "port_code": self.port_code,
            "queue_length": round(self.queue_length, 1),
            "expected_days": round(self.expected_days, 2),
            "median_days": round(self.median_days, 1),
            "p80_days": round(self.p80_days, 1),
            "p95_days": round(self.p95_days, 1),
            # The whole curve, not just its quantiles: the shape is what tells
            # a reader whether the risk is a long flat tail or a sharp cutoff,
            # and those imply very different laycan decisions.
            "survival": self.survival,
            "model": self.model_name,
            "n_train_calls": self.n_train_calls,
            "explanation": self.explanation,
        }


@dataclass
class HazardModel:
    """Fitted daily-hazard model plus the metadata needed to explain it."""

    model: LogisticRegression
    feature_names: list[str]
    n_calls: int
    n_events: int
    n_censored: int
    baseline_queue: float

    def daily_hazard(self, day: int, queue_length: float, tonnes: float) -> float:
        row = _design_row(day, queue_length, tonnes, self.baseline_queue)
        probability = self.model.predict_proba(
            np.asarray([row], dtype=float)
        )[0][1]
        return float(np.clip(probability, 1e-6, 1.0 - 1e-6))


def _design_row(
    day: int, queue_length: float, tonnes: float, baseline_queue: float
) -> list[float]:
    """Covariates for one vessel-day.

    ``log(day)`` rather than ``day`` because the hazard rises steeply over the
    first few days at anchorage then flattens: a vessel that has already
    waited three weeks is in a queue behind something structural, and one more
    day changes little.
    """
    queue = queue_length if np.isfinite(queue_length) else baseline_queue
    parcel = tonnes if np.isfinite(tonnes) and tonnes > 0 else 50_000.0
    return [
        np.log(max(day, 1)),
        queue / 10.0,
        np.log(max(parcel, 1_000.0)) - np.log(50_000.0),
    ]


FEATURE_NAMES = ["log_day", "queue_len_scaled", "log_parcel_rel"]


def build_person_periods(observations: pd.DataFrame) -> pd.DataFrame:
    """Expand each port call into one row per day waited.

    A call that berthed on day *k* yields *k* rows, the last with
    ``berthed=1``. A censored call yields rows to its bound, all with
    ``berthed=0`` — contributing exactly the information it carries and no
    more.
    """
    rows: list[dict] = []
    for record in observations.to_dict("records"):
        waited = int(min(record["waiting_days"], MAX_WAIT_DAYS))
        if waited < 1:
            waited = 1
        censored = bool(record["censored"])
        queue = record.get("queue_len_on_arrival")
        tonnes = record.get("cargo_tonnes")

        for day in range(1, waited + 1):
            berthed = int((not censored) and day == waited)
            rows.append(
                {
                    "day": day,
                    "queue_len": float(queue) if queue is not None else np.nan,
                    "tonnes": float(tonnes) if tonnes is not None else np.nan,
                    "berthed": berthed,
                }
            )
    return pd.DataFrame(rows)


def fit_hazard(observations: pd.DataFrame) -> HazardModel | None:
    """Fit the daily hazard. Returns None when the archive is too thin."""
    if observations.empty or len(observations) < MIN_CALLS:
        return None

    usable = observations[
        observations["berth_date_uncertain_days"].fillna(0) <= MAX_UNCERTAIN_DAYS
    ]
    if len(usable) < MIN_CALLS:
        usable = observations

    periods = build_person_periods(usable)
    if periods.empty or periods["berthed"].sum() < 10:
        return None

    baseline_queue = float(np.nanmedian(periods["queue_len"]))
    if not np.isfinite(baseline_queue):
        baseline_queue = 15.0

    design = np.asarray(
        [
            _design_row(row.day, row.queue_len, row.tonnes, baseline_queue)
            for row in periods.itertuples()
        ],
        dtype=float,
    )
    labels = periods["berthed"].to_numpy(dtype=int)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(design, labels)

    return HazardModel(
        model=model,
        feature_names=FEATURE_NAMES,
        n_calls=int(len(usable)),
        n_events=int((~usable["censored"].astype(bool)).sum()),
        n_censored=int(usable["censored"].astype(bool).sum()),
        baseline_queue=baseline_queue,
    )


def survival_curve(
    fitted: HazardModel, queue_length: float, tonnes: float
) -> np.ndarray:
    """S(t) for t = 1..MAX_WAIT_DAYS, from the fitted daily hazards."""
    survival = np.empty(MAX_WAIT_DAYS)
    running = 1.0
    for index, day in enumerate(range(1, MAX_WAIT_DAYS + 1)):
        running *= 1.0 - fitted.daily_hazard(day, queue_length, tonnes)
        survival[index] = running
    return survival


def _quantile_from_survival(survival: np.ndarray, level: float) -> float:
    """First day on which the survival function drops below ``1 - level``."""
    threshold = 1.0 - level
    below = np.nonzero(survival <= threshold)[0]
    if below.size == 0:
        return float(MAX_WAIT_DAYS)  # censored at the horizon, not extrapolated
    return float(below[0] + 1)


def predict_waiting(
    fitted: HazardModel,
    *,
    port_code: str,
    queue_length: float,
    tonnes: float = 50_000.0,
) -> WaitingForecast:
    """Predict the waiting-time distribution for a vessel joining the queue."""
    survival = survival_curve(fitted, queue_length, tonnes)

    # E[T] = sum of S(t) over t, the standard discrete expectation. Truncated
    # at the horizon, so it is a lower bound when the tail is still alive.
    expected = float(np.sum(survival))

    forecast = WaitingForecast(
        port_code=port_code,
        queue_length=queue_length,
        expected_days=expected,
        median_days=_quantile_from_survival(survival, 0.50),
        p80_days=_quantile_from_survival(survival, 0.80),
        p95_days=_quantile_from_survival(survival, 0.95),
        survival=[round(float(value), 4) for value in survival],
        n_train_calls=fitted.n_calls,
    )
    forecast.explanation = _explain(forecast, fitted)
    return forecast


def _explain(forecast: WaitingForecast, fitted: HazardModel) -> str:
    """Plain English, in the style of the loadable-tonnes engine."""
    return (
        f"With {forecast.queue_length:.0f} vessels waiting at {forecast.port_code}, "
        f"a vessel joining the queue now berths after {forecast.median_days:.0f} days "
        f"at the median and {forecast.p80_days:.0f} days at the 80th percentile. "
        f"Expected detention {forecast.expected_days:.1f} days. "
        f"Fitted on {fitted.n_calls} port calls "
        f"({fitted.n_events} berthed, {fitted.n_censored} still waiting or "
        f"unresolved when the archive ended)."
    )
