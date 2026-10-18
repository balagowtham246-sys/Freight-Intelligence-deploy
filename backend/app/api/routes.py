"""API routes.

Every data-bearing response carries provenance, so the client can show where
each figure came from without a second call.
"""
from __future__ import annotations

import json
import statistics
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_session
from ..domain.feasibility import screen_all
from ..domain.loadable_tonnes import calculate_loadable_tonnes
from ..domain.reference import (
    berths_for_port,
    load_cargoes,
    load_ports_list,
    load_vessel_classes,
)
from ..ml import pipeline, service
from ..models import (
    Alert,
    BacktestMetric,
    CoalStockObservation,
    IngestRun,
    PortCallSnapshot,
    RateObservation,
)
from ..opt.timing import recommend_entry_window
from .schemas import (
    AlertOut,
    DataFreshnessOut,
    FeasibilityRequest,
    ForecastTargetOut,
    LoadableRequest,
    ProvenanceOut,
    QueueEntry,
    SeriesOut,
    TimingRequest,
)

router = APIRouter()


# --------------------------------------------------------------- reference


@router.get("/reference/ports", tags=["reference"])
def get_ports() -> list[dict]:
    """Ports and their berth constraints, with the source of each figure."""
    return [
        {
            "code": port["code"],
            "name": port["name"],
            "country": port["country"],
            "role": port["role"],
            "latitude": port["latitude"],
            "longitude": port["longitude"],
            "water_density": port.get("water_density"),
            "tidal_range_m": port.get("tidal_range_m"),
            "tide_constrained": port.get("tide_constrained", False),
            "source": port.get("source"),
            "notes": port.get("notes"),
            "berths": port.get("berths", []),
        }
        for port in load_ports_list()
    ]


@router.get("/reference/vessels", tags=["reference"])
def get_vessel_classes() -> list[dict]:
    return [
        {
            "name": vessel.name,
            "dwt": vessel.dwt,
            "loa_m": vessel.loa_m,
            "beam_m": vessel.beam_m,
            "summer_draft_m": vessel.summer_draft_m,
            "lightship_t": vessel.lightship_t,
            "tpc": vessel.tpc,
            "grain_cubic_m3": vessel.grain_cubic_m3,
            "geared": vessel.geared,
        }
        for vessel in load_vessel_classes().values()
    ]


@router.get("/reference/cargoes", tags=["reference"])
def get_cargoes() -> list[dict]:
    return [
        {
            "name": cargo.name,
            "category": cargo.berth_category,
            "stowage_factor_m3_per_t": cargo.stowage_factor_m3_per_t,
            "gcv_kcal_per_kg": cargo.gcv_kcal_per_kg,
        }
        for cargo in load_cargoes().values()
    ]


# ------------------------------------------------------------- feasibility


@router.post("/feasibility/screen", tags=["feasibility"])
def post_feasibility(request: FeasibilityRequest) -> dict:
    """Rank every way of moving a cargo, with reasons for each rejection."""
    ports = {port["code"] for port in load_ports_list()}
    if request.discharge_port not in ports:
        raise HTTPException(404, f"unknown discharge port {request.discharge_port}")
    if request.load_port and request.load_port not in ports:
        raise HTTPException(404, f"unknown load port {request.load_port}")
    if request.cargo not in load_cargoes():
        raise HTTPException(404, f"unknown cargo {request.cargo}")

    result = screen_all(
        request.discharge_port,
        request.required_tonnes,
        load_port=request.load_port,
        cargo_name=request.cargo,
        bunkers_rob_t=request.bunkers_rob_t,
        tide_height_m=request.tide_height_m,
    )
    result["provenance"] = ProvenanceOut(
        source_name="port_constraint_master",
        source_url="backend/data/reference/port_constraints.yaml",
        is_synthetic=False,
    ).model_dump()
    return result


@router.post("/feasibility/loadable", tags=["feasibility"])
def post_loadable(request: LoadableRequest) -> dict:
    """Worked loadable-tonnage for one vessel at one berth."""
    vessel = load_vessel_classes().get(request.vessel_class)
    if vessel is None:
        raise HTTPException(404, f"unknown vessel class {request.vessel_class}")

    berths = berths_for_port(request.port_code)
    if not berths:
        raise HTTPException(404, f"no berths known for port {request.port_code}")

    if request.berth_name:
        berths = [b for b in berths if b.berth_name == request.berth_name]
        if not berths:
            raise HTTPException(404, f"unknown berth {request.berth_name}")

    cargo = load_cargoes().get(request.cargo) if request.cargo else None

    results = []
    for berth in berths:
        outcome = calculate_loadable_tonnes(
            vessel,
            berth,
            cargo=cargo,
            bunkers_rob_t=request.bunkers_rob_t,
            tide_height_m=request.tide_height_m,
            ukc_policy_m=request.ukc_policy_m,
        )
        results.append({"berth": berth.berth_name, **outcome.as_dict()})

    return {
        "vessel_class": request.vessel_class,
        "port_code": request.port_code,
        "results": results,
    }


# -------------------------------------------------------------- port watch


@router.get("/portwatch/{port_code}", tags=["portwatch"])
def get_port_watch(
    port_code: str,
    session: Session = Depends(get_session),
) -> dict:
    """Latest queue snapshot for a port, from the scraped berthing programme."""
    latest = session.scalar(
        select(func.max(PortCallSnapshot.snapshot_date)).where(
            PortCallSnapshot.port_code == port_code
        )
    )
    if latest is None:
        raise HTTPException(404, f"no snapshots held for port {port_code}")

    rows = session.scalars(
        select(PortCallSnapshot).where(
            PortCallSnapshot.port_code == port_code,
            PortCallSnapshot.snapshot_date == latest,
        )
    ).all()

    counts = {"working": 0, "waiting": 0, "expected": 0}
    queue: list[QueueEntry] = []
    waits: list[float] = []

    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        waiting_days = None
        if row.status == "waiting" and row.arrival_date:
            waiting_days = (latest - row.arrival_date).days
            waits.append(waiting_days)

        queue.append(
            QueueEntry(
                vessel_name=row.vessel_name,
                status=row.status,
                cargo=row.cargo,
                cargo_tonnes=row.cargo_tonnes,
                berth=row.berth,
                arrival_date=row.arrival_date,
                waiting_days=waiting_days,
                loa_m=row.loa_m,
                draft_m=row.draft_m,
            )
        )

    sample = rows[0] if rows else None
    return {
        "port_code": port_code,
        "snapshot_date": latest,
        "working": counts.get("working", 0),
        "waiting": counts.get("waiting", 0),
        "expected": counts.get("expected", 0),
        "median_waiting_days": statistics.median(waits) if waits else None,
        "queue": [entry.model_dump() for entry in queue],
        "provenance": ProvenanceOut(
            source_name=sample.source_name if sample else "unknown",
            source_url=sample.source_url if sample else None,
            retrieved_at=sample.retrieved_at.isoformat() if sample else None,
            is_synthetic=sample.is_synthetic if sample else False,
        ).model_dump(),
    }


@router.get("/portwatch/{port_code}/history", tags=["portwatch"])
def get_queue_history(
    port_code: str,
    days: int = Query(120, ge=1, le=730),
    session: Session = Depends(get_session),
) -> dict:
    """Daily queue length, from the archived berthing programmes."""
    since = date.today() - timedelta(days=days)
    rows = session.execute(
        select(
            PortCallSnapshot.snapshot_date,
            PortCallSnapshot.status,
            func.count().label("n"),
        )
        .where(
            PortCallSnapshot.port_code == port_code,
            PortCallSnapshot.snapshot_date >= since,
        )
        .group_by(PortCallSnapshot.snapshot_date, PortCallSnapshot.status)
        .order_by(PortCallSnapshot.snapshot_date)
    ).all()

    by_date: dict[date, dict[str, int]] = {}
    for snapshot_date, status, count in rows:
        by_date.setdefault(snapshot_date, {})[status] = count

    return {
        "port_code": port_code,
        "points": [
            {
                "date": key.isoformat(),
                "working": value.get("working", 0),
                "waiting": value.get("waiting", 0),
                "expected": value.get("expected", 0),
            }
            for key, value in sorted(by_date.items())
        ],
    }


# ----------------------------------------------------------------- series


@router.get("/series/{series_code}", response_model=SeriesOut, tags=["market"])
def get_series(
    series_code: str,
    since: date | None = None,
    session: Session = Depends(get_session),
) -> SeriesOut:
    """One commodity or freight series."""
    query = select(RateObservation).where(RateObservation.series_code == series_code)
    if since:
        query = query.where(RateObservation.obs_date >= since)
    rows = session.scalars(query.order_by(RateObservation.obs_date)).all()

    if not rows:
        raise HTTPException(404, f"no observations for series {series_code}")

    return SeriesOut(
        series_code=series_code,
        unit=rows[0].unit,
        points=[{"date": row.obs_date, "value": row.value} for row in rows],
        provenance=ProvenanceOut(
            source_name=rows[-1].source_name,
            source_url=rows[-1].source_url,
            retrieved_at=rows[-1].retrieved_at.isoformat(),
            is_synthetic=rows[-1].is_synthetic,
        ),
    )


@router.get("/series", tags=["market"])
def list_series(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.execute(
        select(
            RateObservation.series_code,
            RateObservation.series_kind,
            RateObservation.unit,
            func.count().label("n"),
            func.min(RateObservation.obs_date),
            func.max(RateObservation.obs_date),
            func.max(RateObservation.is_synthetic),
        ).group_by(RateObservation.series_code)
    ).all()

    return [
        {
            "series_code": code,
            "series_kind": kind,
            "unit": unit,
            "observations": count,
            "first": first.isoformat(),
            "last": last.isoformat(),
            "is_synthetic": bool(synthetic),
        }
        for code, kind, unit, count, first, last, synthetic in rows
    ]


# ------------------------------------------------------------ coal demand


@router.get("/demand/coal-stock", tags=["demand"])
def get_coal_stock(
    import_only: bool = True,
    session: Session = Depends(get_session),
) -> dict:
    """Plant coal stock days-of-burn, the import urgency signal."""
    latest = session.scalar(select(func.max(CoalStockObservation.obs_date)))
    if latest is None:
        raise HTTPException(404, "no coal stock observations held")

    query = select(CoalStockObservation).where(CoalStockObservation.obs_date == latest)
    if import_only:
        query = query.where(CoalStockObservation.is_import_based.is_(True))

    rows = session.scalars(query.order_by(CoalStockObservation.days_of_burn)).all()
    burns = [row.days_of_burn for row in rows if row.days_of_burn is not None]

    return {
        "obs_date": latest.isoformat(),
        "import_only": import_only,
        "plant_count": len(rows),
        "median_days_of_burn": statistics.median(burns) if burns else None,
        "critical_count": sum(1 for value in burns if value < 5),
        "plants": [
            {
                "plant_name": row.plant_name,
                "state": row.state,
                "region": row.region,
                "capacity_mw": row.capacity_mw,
                "stock_kt": row.stock_kt,
                "days_of_burn": row.days_of_burn,
            }
            for row in rows
        ],
        "provenance": ProvenanceOut(
            source_name=rows[0].source_name if rows else "unknown",
            source_url=rows[0].source_url if rows else None,
            retrieved_at=rows[0].retrieved_at.isoformat() if rows else None,
            is_synthetic=rows[0].is_synthetic if rows else False,
        ).model_dump(),
    }


# ---------------------------------------------------------------- meta


@router.get("/meta/freshness", response_model=list[DataFreshnessOut], tags=["meta"])
def get_freshness(session: Session = Depends(get_session)) -> list[DataFreshnessOut]:
    """Most recent run of each adapter, so staleness is visible in the UI."""
    subquery = (
        select(IngestRun.adapter, func.max(IngestRun.id).label("latest_id"))
        .group_by(IngestRun.adapter)
        .subquery()
    )
    rows = session.scalars(
        select(IngestRun).join(subquery, IngestRun.id == subquery.c.latest_id)
    ).all()

    return [
        DataFreshnessOut(
            adapter=row.adapter,
            status=row.status,
            rows_written=row.rows_written,
            finished_at=row.finished_at.isoformat() if row.finished_at else None,
            used_cache=row.used_cache,
            message=row.message,
        )
        for row in rows
    ]


@router.get("/meta/provenance-summary", tags=["meta"])
def get_provenance_summary(session: Session = Depends(get_session)) -> dict:
    """How much of the platform data is real versus synthetic.

    Backs the claim made in SYNTHETIC_DATA.md with a live count rather than
    asking anyone to take it on trust.
    """
    summary: dict[str, dict] = {}
    total_real = 0
    total_synthetic = 0

    for model, label in (
        (RateObservation, "rate_observations"),
        (PortCallSnapshot, "port_call_snapshots"),
        (CoalStockObservation, "coal_stock_observations"),
    ):
        rows = session.execute(
            select(model.source_name, model.is_synthetic, func.count()).group_by(
                model.source_name, model.is_synthetic
            )
        ).all()
        entries = []
        for source_name, is_synthetic, count in rows:
            entries.append(
                {"source_name": source_name, "is_synthetic": bool(is_synthetic), "rows": count}
            )
            if is_synthetic:
                total_synthetic += count
            else:
                total_real += count
        summary[label] = entries

    total = total_real + total_synthetic
    return {
        "tables": summary,
        "total_rows": total,
        "real_rows": total_real,
        "synthetic_rows": total_synthetic,
        "pct_real": round(100.0 * total_real / total, 2) if total else None,
    }


# --------------------------------------------------------------- forecasting


def _skill_rows(session: Session, target_key: str) -> list[dict]:
    """Measured performance per horizon, assembled from backtest_metrics.

    Skill is taken as the *worst* against any baseline, so a model that beats
    a random walk but loses to seasonal naive does not get to claim an edge.
    """
    rows = session.scalars(
        select(BacktestMetric).where(BacktestMetric.target_key == target_key)
    ).all()

    by_horizon: dict[int, dict] = {}
    for row in rows:
        entry = by_horizon.setdefault(
            row.horizon_days,
            {
                "horizon_days": row.horizon_days,
                "n_folds": row.n_folds,
                "point_skill": None,
                "distributional_skill": None,
                "mase": None,
                "coverage_80": None,
                "computed_on_synthetic": row.computed_on_synthetic,
            },
        )
        if row.metric == "mae" and row.skill is not None:
            worst = entry["point_skill"]
            entry["point_skill"] = row.skill if worst is None else min(worst, row.skill)
        elif row.metric == "crps" and row.skill is not None:
            worst = entry["distributional_skill"]
            entry["distributional_skill"] = (
                row.skill if worst is None else min(worst, row.skill)
            )
        elif row.metric == "mase":
            entry["mase"] = row.value
        elif row.metric == "coverage_80":
            entry["coverage_80"] = row.value

    for entry in by_horizon.values():
        point = entry["point_skill"]
        distributional = entry["distributional_skill"]
        entry["has_edge"] = bool(
            entry["n_folds"] >= 5
            and point is not None
            and point > 0
            and distributional is not None
            and distributional > 0
        )

    return [by_horizon[key] for key in sorted(by_horizon)]


@router.get("/forecast/targets", tags=["forecast"])
def get_forecast_targets(session: Session = Depends(get_session)) -> list[dict]:
    """Every series that could be forecast, with its depth and whether it can be."""
    from ..ml.quantile import MIN_TRAIN_ROWS

    out = []
    for info in service.list_targets(session):
        forecastable = info.n_observations >= MIN_TRAIN_ROWS
        out.append(
            ForecastTargetOut(
                **info.as_dict(),
                forecastable=forecastable,
                reason=(
                    None
                    if forecastable
                    else (
                        f"{info.n_observations} observations held; {MIN_TRAIN_ROWS} "
                        "needed before a model is fitted. Depth accumulates daily."
                    )
                ),
            ).model_dump()
        )
    return out


@router.get("/forecast/{target_key}", tags=["forecast"])
def get_forecast(
    target_key: str,
    history_days: int = Query(365, ge=30, le=4000),
    session: Session = Depends(get_session),
) -> dict:
    """History plus the calibrated fan chart, with measured skill attached."""
    run = service.latest_run(session, target_key)
    if run is None:
        raise HTTPException(
            404,
            f"no forecast has been fitted for {target_key}. Run scripts/run_models.py.",
        )

    since = date.today() - timedelta(days=history_days)
    observations = session.scalars(
        select(RateObservation)
        .where(
            RateObservation.series_code == target_key,
            RateObservation.obs_date >= since,
        )
        .order_by(RateObservation.obs_date)
    ).all()

    skill = _skill_rows(session, target_key)
    best = max(
        (row["point_skill"] for row in skill if row["point_skill"] is not None),
        default=None,
    )
    any_edge = any(row["has_edge"] for row in skill)

    if not skill:
        signal = "no_edge"
        caveat = (
            "This forecast has not been backtested. An unmeasured model is not a "
            "trustworthy one, so the platform reports no edge."
        )
    elif not any_edge:
        signal = "no_edge"
        caveat = (
            "Measured against a random walk and a seasonal naive baseline over "
            "walk-forward folds, this model does not beat persistence at any "
            "horizon. The bands are shown because they are calibrated and useful "
            "for sizing risk, but the platform will not time the market on them."
        )
    else:
        signal = "strong" if (best or 0) >= 0.15 else "weak"
        caveat = None

    if run.trained_on_synthetic:
        note = (
            "Trained on a disclosed synthetic development series (see "
            "SYNTHETIC_DATA.md). The real CDFI archive takes over automatically "
            "once it is deep enough."
        )
        caveat = f"{caveat} {note}" if caveat else note

    return {
        "target_key": target_key,
        "unit": observations[0].unit if observations else "usd_per_t",
        "history": [
            {"date": row.obs_date.isoformat(), "value": row.value}
            for row in observations
        ],
        "bands": service.forecast_bands(session, run),
        "skill": skill,
        "signal_strength": signal,
        "caveat": caveat,
        "model": {
            "model_name": run.model_name,
            "model_version": run.model_version,
            "trained_through": run.trained_through.isoformat(),
            "n_train": run.n_train,
            "input_series": run.input_series.split(",") if run.input_series else [],
            "trained_on_synthetic": run.trained_on_synthetic,
        },
        "provenance": ProvenanceOut(
            source_name=observations[-1].source_name if observations else "unknown",
            source_url=observations[-1].source_url if observations else None,
            retrieved_at=(
                observations[-1].retrieved_at.isoformat() if observations else None
            ),
            is_synthetic=observations[-1].is_synthetic if observations else False,
        ).model_dump(),
    }


@router.get("/forecast/{target_key}/backtest", tags=["forecast"])
def get_backtest(target_key: str, session: Session = Depends(get_session)) -> dict:
    """The honesty screen: walk-forward performance against every baseline."""
    rows = session.scalars(
        select(BacktestMetric)
        .where(BacktestMetric.target_key == target_key)
        .order_by(BacktestMetric.horizon_days, BacktestMetric.metric)
    ).all()
    if not rows:
        raise HTTPException(404, f"no backtest held for {target_key}")

    return {
        "target_key": target_key,
        "computed_on_synthetic": any(row.computed_on_synthetic for row in rows),
        "ffa_benchmark": (
            "unavailable - exchange-licensed, and SYNTHETIC_DATA.md forbids "
            "benchmarking against a curve we generated ourselves"
        ),
        "horizons": _skill_rows(session, target_key),
        "detail": [
            {
                "horizon_days": row.horizon_days,
                "metric": row.metric,
                "value": round(row.value, 5),
                "baseline_name": row.baseline_name,
                "baseline_value": (
                    round(row.baseline_value, 5)
                    if row.baseline_value is not None
                    else None
                ),
                "skill": round(row.skill, 4) if row.skill is not None else None,
                "n_folds": row.n_folds,
            }
            for row in rows
        ],
    }


@router.get("/portwatch/{port_code}/waiting-forecast", tags=["portwatch"])
def get_waiting_forecast(
    port_code: str,
    tonnes: float = Query(50_000.0, gt=0),
    session: Session = Depends(get_session),
) -> dict:
    """Predicted pre-berthing detention for a vessel joining the queue now."""
    forecast = service.waiting_forecast(session, port_code, tonnes=tonnes)
    if forecast is None:
        raise HTTPException(
            404,
            f"no waiting-time model for {port_code}: the berthing-programme "
            "archive is not deep enough yet.",
        )
    return {
        **forecast.as_dict(),
        "provenance": ProvenanceOut(
            source_name="paradip_daily_traffic",
            source_url="https://www.paradipport.gov.in/",
            is_synthetic=False,
        ).model_dump(),
    }


# --------------------------------------------------------------------- timing


@router.post("/timing/window", tags=["timing"])
def post_timing_window(
    request: TimingRequest, session: Session = Depends(get_session)
) -> dict:
    """Optimal market entry timing for one cargo requirement."""
    ports = {port["code"] for port in load_ports_list()}
    if request.discharge_port not in ports:
        raise HTTPException(404, f"unknown discharge port {request.discharge_port}")
    if request.load_port and request.load_port not in ports:
        raise HTTPException(404, f"unknown load port {request.load_port}")
    if request.cargo not in load_cargoes():
        raise HTTPException(404, f"unknown cargo {request.cargo}")
    if request.not_later_than <= date.today():
        raise HTTPException(400, "not_later_than must be in the future")

    target_key = request.target_series or pipeline.choose_target(session)
    if target_key is None:
        raise HTTPException(404, "no freight series held to price against")

    run = service.latest_run(session, target_key)
    if run is None:
        raise HTTPException(
            404, f"no forecast fitted for {target_key}. Run scripts/run_models.py."
        )
    bands = service.forecast_bands(session, run)

    waiting = service.waiting_forecast(
        session, request.discharge_port, tonnes=request.required_tonnes
    )
    if waiting is None:
        # Fall back to the one port with an archive rather than silently
        # assuming zero waiting time, which would understate delivered cost.
        waiting = service.waiting_forecast(
            session, "INPRT", tonnes=request.required_tonnes
        )
    median_wait = waiting.median_days if waiting else 3.0
    p95_wait = waiting.p95_days if waiting else 10.0

    horizon = (request.not_later_than - date.today()).days
    skill = service.point_skill(session, target_key, service.nearest_horizon(horizon))
    _, regime = service.market_state(session, target_key)
    burn, _, _ = pipeline._coal_stock_summary(session)

    window = recommend_entry_window(
        discharge_port=request.discharge_port,
        required_tonnes=request.required_tonnes,
        cargo=request.cargo,
        load_port=request.load_port,
        not_later_than=request.not_later_than,
        forecast_bands=bands,
        waiting_median_days=median_wait,
        waiting_p95_days=p95_wait,
        point_skill=skill,
        risk_aversion=request.risk_aversion,
        days_of_burn=burn,
        demurrage_usd_per_day=request.demurrage_usd_per_day,
        bunker_usd_per_tonne=request.bunker_usd_per_tonne,
        n_scenarios=request.n_scenarios,
        regime_changed_recently=bool(regime and regime.changed_recently),
    )

    return {
        **window.as_dict(),
        "target_series": target_key,
        "waiting_forecast": waiting.as_dict() if waiting else None,
        "model": {
            "model_name": run.model_name,
            "model_version": run.model_version,
            "trained_through": run.trained_through.isoformat(),
            "n_train": run.n_train,
            "input_series": run.input_series.split(",") if run.input_series else [],
            "trained_on_synthetic": run.trained_on_synthetic,
        },
    }


# ----------------------------------------------------------------------- risk


def _alert_out(row: Alert) -> dict:
    try:
        evidence = json.loads(row.evidence_json) if row.evidence_json else {}
    except ValueError:
        evidence = {}
    return AlertOut(
        id=row.id,
        code=row.code,
        severity=row.severity,
        scope=row.scope,
        port_code=row.port_code,
        series_code=row.series_code,
        raised_date=row.raised_date,
        cleared_date=row.cleared_date,
        last_seen_date=row.last_seen_date,
        title=row.title,
        body=row.body,
        action=row.action,
        evidence=evidence,
    ).model_dump()


@router.get("/risk/alerts", tags=["risk"])
def get_alerts(
    cleared_within_days: int = Query(14, ge=0, le=120),
    session: Session = Depends(get_session),
) -> dict:
    """Active warnings, recently cleared ones, and every monitor's reading.

    The monitors matter as much as the alerts: an empty alert list should read
    as "nothing is wrong", not as "nothing is being watched".
    """
    severity_rank = {"critical": 0, "warning": 1, "watch": 2, "info": 3}

    active = sorted(
        session.scalars(select(Alert).where(Alert.cleared_date.is_(None))).all(),
        key=lambda row: (severity_rank.get(row.severity, 9), row.raised_date),
    )
    since = date.today() - timedelta(days=cleared_within_days)
    cleared = session.scalars(
        select(Alert)
        .where(Alert.cleared_date.isnot(None), Alert.cleared_date >= since)
        .order_by(Alert.cleared_date.desc())
    ).all()

    return {
        "as_of": date.today().isoformat(),
        "active": [_alert_out(row) for row in active],
        "recently_cleared": [_alert_out(row) for row in cleared],
        "monitors": pipeline.monitor_status(session),
    }
