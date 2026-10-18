"""Orchestration: fit, persist and read back model output.

This is the only module in ``app.ml`` that touches the database. Everything
else is pure functions over frames, which is what keeps the test suite able to
exercise the modelling without a warehouse.

A note on model persistence
---------------------------
There is deliberately no pickled-model store. Fits here take seconds on a few
thousand rows, the daily job refits anyway, and a serialised LightGBM booster
is a version-skew hazard that silently returns wrong numbers when the library
moves under it. So the platform persists *outputs* — forecasts, metrics,
alerts — which are what the API serves and what the decision ledger needs, and
refits on demand behind a small in-process cache.

If fits ever become slow enough to matter, the place to add a store is here,
behind ``fitted_models``, without touching anything else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import (
    BacktestMetric,
    ForecastPoint,
    ForecastRun,
    RateObservation,
    WaitingTimeObservation,
)
from .evaluate import HorizonMetrics, walk_forward
from .features import build_features
from .panel import PanelSpec, build_panel
from .quantile import HORIZONS_DAYS, QUANTILES, fit_and_predict
from .regime import RegimeState, regime_state
from .volatility import VolatilityState, volatility_state
from .waiting_model import HazardModel, WaitingForecast, fit_hazard, predict_waiting

MODEL_NAME = "lgbm_quantile"
MODEL_VERSION = "1"

# Series the platform will forecast, in preference order. The real CDFI series
# take over automatically once they are deep enough to fit.
FORECASTABLE_KINDS = ("route", "freight_index")

_hazard_cache: dict[tuple[str, int], HazardModel | None] = {}


@dataclass
class TargetInfo:
    """What the platform knows about one forecastable series."""

    series_code: str
    series_kind: str
    unit: str
    n_observations: int
    first: date
    last: date
    is_synthetic: bool

    def as_dict(self) -> dict:
        return {
            "series_code": self.series_code,
            "series_kind": self.series_kind,
            "unit": self.unit,
            "observations": self.n_observations,
            "first": self.first.isoformat(),
            "last": self.last.isoformat(),
            "is_synthetic": self.is_synthetic,
        }


def list_targets(session: Session) -> list[TargetInfo]:
    """Every series that could be a forecast target, with its depth."""
    rows = session.execute(
        select(
            RateObservation.series_code,
            RateObservation.series_kind,
            RateObservation.unit,
            func.count().label("n"),
            func.min(RateObservation.obs_date),
            func.max(RateObservation.obs_date),
            func.max(RateObservation.is_synthetic),
        )
        .where(RateObservation.series_kind.in_(FORECASTABLE_KINDS))
        .group_by(RateObservation.series_code)
    ).all()

    return sorted(
        (
            TargetInfo(
                series_code=code,
                series_kind=kind,
                unit=unit,
                n_observations=count,
                first=first,
                last=last,
                is_synthetic=bool(synthetic),
            )
            for code, kind, unit, count, first, last, synthetic in rows
        ),
        key=lambda info: (-info.n_observations, info.series_code),
    )


def _design_and_target(
    session: Session, target_key: str, as_of: date
) -> tuple[pd.DataFrame, pd.Series]:
    panel = build_panel(session, PanelSpec(target_series=target_key, as_of=as_of))
    design = build_features(panel, target_key)
    return design, panel[target_key]


def run_forecast(
    session: Session,
    target_key: str,
    *,
    as_of: date | None = None,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
) -> ForecastRun:
    """Fit the horizon grid and persist the resulting fan chart."""
    latest = session.scalar(
        select(func.max(RateObservation.obs_date)).where(
            RateObservation.series_code == target_key
        )
    )
    if latest is None:
        raise ValueError(f"no observations for series {target_key!r}")
    as_of = as_of or latest

    design, target = _design_and_target(session, target_key, as_of)
    forecasts, skipped = fit_and_predict(
        design,
        target,
        target_key=target_key,
        origin_date=as_of,
        horizons=horizons,
    )

    input_series = sorted(
        {
            column.replace("SYN_", "SYN_")
            for column in (target_key,)
        }
        | {"COAL_AU_NEWC", "CRUDE_BRENT"}
    )

    run = ForecastRun(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        target_key=target_key,
        run_date=date.today(),
        trained_through=as_of,
        n_train=forecasts[0].n_train if forecasts else 0,
        input_series=",".join(input_series),
        trained_on_synthetic=target_key.startswith("SYN_"),
        params_json=json.dumps(
            {
                "horizons": list(horizons),
                "quantiles": list(QUANTILES),
                "skipped_horizons": skipped,
                "n_features": int(design.shape[1]),
            }
        ),
        status="ok" if forecasts else "skipped",
        message=(
            None
            if not skipped
            else f"insufficient history for horizons {skipped}"
        ),
    )
    session.add(run)
    session.flush()

    for forecast in forecasts:
        for level, value in forecast.quantiles.items():
            session.add(
                ForecastPoint(
                    run_id=run.id,
                    target_key=target_key,
                    origin_date=forecast.origin_date,
                    horizon_days=forecast.horizon_days,
                    target_date=forecast.target_date,
                    quantile=level,
                    value=value,
                )
            )
    session.flush()
    return run


def run_backtest(
    session: Session,
    target_key: str,
    *,
    as_of: date | None = None,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    n_folds: int = 8,
) -> list[HorizonMetrics]:
    """Walk-forward evaluation, persisted to ``backtest_metrics``."""
    latest = session.scalar(
        select(func.max(RateObservation.obs_date)).where(
            RateObservation.series_code == target_key
        )
    )
    if latest is None:
        raise ValueError(f"no observations for series {target_key!r}")
    as_of = as_of or latest

    design, target = _design_and_target(session, target_key, as_of)

    # Clear previous metrics for this target so the table always reflects the
    # current model rather than accumulating a confusing history of versions.
    for stale in session.scalars(
        select(BacktestMetric).where(
            BacktestMetric.target_key == target_key,
            BacktestMetric.model_name == MODEL_NAME,
        )
    ).all():
        session.delete(stale)
    session.flush()

    results: list[HorizonMetrics] = []
    for horizon in horizons:
        metrics = walk_forward(
            design, target, target_key=target_key, horizon_days=horizon, n_folds=n_folds
        )
        if metrics is None:
            continue
        results.append(metrics)

        for baseline_name, baseline_crps in metrics.baseline_crps.items():
            session.add(
                BacktestMetric(
                    target_key=target_key,
                    model_name=MODEL_NAME,
                    horizon_days=horizon,
                    metric="crps",
                    value=metrics.crps,
                    baseline_name=baseline_name,
                    baseline_value=baseline_crps,
                    skill=metrics.skill_vs(baseline_name),
                    n_folds=metrics.n_folds,
                    computed_on_synthetic=metrics.computed_on_synthetic,
                )
            )
            session.add(
                BacktestMetric(
                    target_key=target_key,
                    model_name=MODEL_NAME,
                    horizon_days=horizon,
                    metric="mae",
                    value=metrics.mae,
                    baseline_name=baseline_name,
                    baseline_value=metrics.baseline_mae.get(baseline_name),
                    skill=metrics.point_skill_vs(baseline_name),
                    n_folds=metrics.n_folds,
                    computed_on_synthetic=metrics.computed_on_synthetic,
                )
            )

        for name, value in (
            ("coverage_80", metrics.coverage_80),
            ("mase", metrics.mase),
        ):
            if value is None or value != value:  # NaN
                continue
            session.add(
                BacktestMetric(
                    target_key=target_key,
                    model_name=MODEL_NAME,
                    horizon_days=horizon,
                    metric=name,
                    value=float(value),
                    n_folds=metrics.n_folds,
                    computed_on_synthetic=metrics.computed_on_synthetic,
                )
            )

    session.flush()
    return results


def point_skill(session: Session, target_key: str, horizon_days: int) -> float | None:
    """Worst point skill against any baseline at a horizon.

    This is what ``signal_strength`` reads. Returns None when the target has
    never been backtested, which the caller must treat as "no edge" rather
    than as "fine" — an unmeasured model is not a good one.
    """
    rows = session.scalars(
        select(BacktestMetric).where(
            BacktestMetric.target_key == target_key,
            BacktestMetric.horizon_days == horizon_days,
            BacktestMetric.metric == "mae",
            BacktestMetric.skill.isnot(None),
        )
    ).all()
    if not rows:
        return None
    return min(float(row.skill) for row in rows)


def nearest_horizon(horizon_days: int, horizons=HORIZONS_DAYS) -> int:
    return min(horizons, key=lambda candidate: abs(candidate - horizon_days))


def latest_run(session: Session, target_key: str) -> ForecastRun | None:
    return session.scalars(
        select(ForecastRun)
        .where(ForecastRun.target_key == target_key, ForecastRun.status == "ok")
        .order_by(ForecastRun.id.desc())
        .limit(1)
    ).first()


def forecast_bands(session: Session, run: ForecastRun) -> list[dict]:
    """The fan chart for a run: one row per horizon, quantiles as columns."""
    points = session.scalars(
        select(ForecastPoint)
        .where(ForecastPoint.run_id == run.id)
        .order_by(ForecastPoint.horizon_days, ForecastPoint.quantile)
    ).all()

    by_horizon: dict[int, dict] = {}
    for point in points:
        entry = by_horizon.setdefault(
            point.horizon_days,
            {
                "horizon_days": point.horizon_days,
                "date": point.target_date.isoformat(),
                "quantiles": {},
            },
        )
        entry["quantiles"][str(point.quantile)] = round(point.value, 4)

    return [by_horizon[key] for key in sorted(by_horizon)]


def series_frame(session: Session, series_code: str) -> pd.Series:
    """One series as a datetime-indexed pandas Series."""
    rows = session.execute(
        select(RateObservation.obs_date, RateObservation.value)
        .where(RateObservation.series_code == series_code)
        .order_by(RateObservation.obs_date)
    ).all()
    if not rows:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(rows, columns=["obs_date", "value"])
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])
    return frame.set_index("obs_date")["value"]


def market_state(
    session: Session, series_code: str
) -> tuple[VolatilityState | None, RegimeState | None]:
    """Volatility and regime for one series. Both may be None on thin data."""
    import numpy as np

    values = series_frame(session, series_code)
    if values.empty or len(values) < 60:
        return None, None
    returns = np.log(values.where(values > 0)).diff()
    return volatility_state(returns), regime_state(returns)


def waiting_observations(session: Session, port_code: str) -> pd.DataFrame:
    rows = session.execute(
        select(
            WaitingTimeObservation.waiting_days,
            WaitingTimeObservation.censored,
            WaitingTimeObservation.queue_len_on_arrival,
            WaitingTimeObservation.cargo_tonnes,
            WaitingTimeObservation.berth_date_uncertain_days,
        ).where(WaitingTimeObservation.port_code == port_code)
    ).all()
    return pd.DataFrame(
        rows,
        columns=[
            "waiting_days",
            "censored",
            "queue_len_on_arrival",
            "cargo_tonnes",
            "berth_date_uncertain_days",
        ],
    )


def hazard_model(session: Session, port_code: str) -> HazardModel | None:
    """Fitted waiting-time model for a port, cached per archive depth.

    The cache key includes the row count, so the model is refitted whenever
    the archive grows rather than going stale for the life of the process.
    """
    count = (
        session.scalar(
            select(func.count()).select_from(WaitingTimeObservation).where(
                WaitingTimeObservation.port_code == port_code
            )
        )
        or 0
    )
    key = (port_code, int(count))
    if key not in _hazard_cache:
        observations = waiting_observations(session, port_code)
        _hazard_cache.clear()
        _hazard_cache[key] = fit_hazard(observations) if not observations.empty else None
    return _hazard_cache[key]


def current_queue_length(session: Session, port_code: str) -> float | None:
    from ..models import PortCallSnapshot

    latest = session.scalar(
        select(func.max(PortCallSnapshot.snapshot_date)).where(
            PortCallSnapshot.port_code == port_code
        )
    )
    if latest is None:
        return None
    return float(
        session.scalar(
            select(func.count())
            .select_from(PortCallSnapshot)
            .where(
                PortCallSnapshot.port_code == port_code,
                PortCallSnapshot.snapshot_date == latest,
                PortCallSnapshot.status == "waiting",
            )
        )
        or 0
    )


def waiting_forecast(
    session: Session,
    port_code: str,
    *,
    tonnes: float = 50_000.0,
    queue_length: float | None = None,
) -> WaitingForecast | None:
    """Predicted pre-berthing detention for a vessel joining the queue now."""
    fitted = hazard_model(session, port_code)
    if fitted is None:
        return None
    queue = (
        queue_length
        if queue_length is not None
        else current_queue_length(session, port_code)
    )
    if queue is None:
        return None
    return predict_waiting(
        fitted, port_code=port_code, queue_length=queue, tonnes=tonnes
    )
