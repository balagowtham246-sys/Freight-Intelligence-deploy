"""Baselines the forecaster has to beat before it is allowed to say anything.

Dry bulk freight is close to a random walk in log space. That makes the naive
forecast — "tomorrow is today" — brutally hard to beat, and it is the reason
most freight forecasting projects quietly stop reporting skill numbers.

This module exists so that the platform cannot do that. Every forecast is
scored against these, the skill is stored in ``backtest_metrics``, and where
skill is not positive the UI shows a ``no edge`` badge instead of a number.

The FFA forward curve is the other benchmark the strategy document demands.
It is exchange-licensed and we do not hold it, and SYNTHETIC_DATA.md forbids
manufacturing one to beat. So the platform reports *benchmark unavailable*
for that comparison rather than a fabricated one. These baselines are computed
from the target series itself and are therefore honest whether that series is
real or disclosed-synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk(history: pd.Series, horizon_days: int) -> float:
    """Last observed value, carried forward. The hard one to beat."""
    clean = history.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[-1])


def drift(history: pd.Series, horizon_days: int, *, window: int = 60) -> float:
    """Random walk with the recent average drift extrapolated."""
    clean = history.dropna()
    if len(clean) < 2:
        return random_walk(history, horizon_days)
    recent = clean.iloc[-window:]
    if len(recent) < 2:
        return float(clean.iloc[-1])
    per_step = (recent.iloc[-1] - recent.iloc[0]) / (len(recent) - 1)
    return float(clean.iloc[-1] + per_step * horizon_days)


def seasonal_naive(
    history: pd.Series, horizon_days: int, *, season_days: int = 365
) -> float:
    """The value one season ago, at the same point in the annual cycle.

    Falls back to a random walk when the history is shorter than a season,
    which with a young archive is the common case — and saying so is better
    than silently returning something else.
    """
    clean = history.dropna()
    if clean.empty:
        return float("nan")
    target_index = len(clean) + horizon_days - season_days - 1
    if target_index < 0 or target_index >= len(clean):
        return random_walk(history, horizon_days)
    return float(clean.iloc[target_index])


BASELINES = {
    "random_walk": random_walk,
    "drift": drift,
    "seasonal_naive": seasonal_naive,
}


def baseline_forecast(name: str, history: pd.Series, horizon_days: int) -> float:
    try:
        return BASELINES[name](history, horizon_days)
    except KeyError as exc:
        raise ValueError(f"unknown baseline {name!r}") from exc


def mase_scale(history: pd.Series, *, season_days: int = 1) -> float:
    """Denominator for MASE: mean absolute change of the naive forecast.

    With ``season_days=1`` this is the in-sample mean absolute first
    difference, so MASE < 1 means the model beats a random walk on the
    training distribution. Returns NaN rather than zero for a flat series, so
    a degenerate scale surfaces as a missing metric instead of an infinity.
    """
    clean = history.dropna()
    if len(clean) <= season_days:
        return float("nan")
    changes = np.abs(clean.to_numpy()[season_days:] - clean.to_numpy()[:-season_days])
    scale = float(np.mean(changes))
    return scale if scale > 0 else float("nan")
