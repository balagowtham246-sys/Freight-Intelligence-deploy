"""The daily model pipeline: derive, fit, forecast, evaluate, alert.

Invoked by ``scripts/run_models.py`` after ingestion. Kept in the package
rather than in the script so the API and the tests can call the same code
paths, and so a failure in one stage is contained rather than taking the run
down — a stale forecast is bad, a crashed pipeline that stops the alerts is
worse.
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..domain import alerts as alert_rules
from ..domain.reference import load_ports_list
from ..models import (
    CoalStockObservation,
    CycloneTrackPoint,
    IngestRun,
    PortCallSnapshot,
    RateObservation,
)
from . import service
from .waiting import rebuild_waiting_times

CONGESTION_PORTS = ("INPRT",)

# Queue history depth used for the congestion z-score.
QUEUE_HISTORY_DAYS = 120


def choose_target(session: Session) -> str | None:
    """The deepest forecastable series, preferring real data over synthetic.

    Once the real CDFI archive is deep enough to fit, it wins automatically and
    the synthetic development series stops being used - which is the whole
    point of keeping them in separate namespaces.
    """
    targets = service.list_targets(session)
    if not targets:
        return None

    real = [
        info
        for info in targets
        if not info.is_synthetic and info.n_observations >= 600
    ]
    if real:
        return real[0].series_code
    return targets[0].series_code


def run_models(
    session: Session,
    *,
    target_key: str | None = None,
    backtest: bool = True,
    n_folds: int = 8,
) -> dict:
    """Fit the forecaster and persist forecasts and metrics."""
    target_key = target_key or choose_target(session)
    if target_key is None:
        return {"status": "skipped", "reason": "no forecastable series held"}

    run = service.run_forecast(session, target_key)
    result = {
        "target": target_key,
        "forecast_run_id": run.id,
        "status": run.status,
        "trained_through": run.trained_through.isoformat(),
        "trained_on_synthetic": run.trained_on_synthetic,
    }

    if backtest:
        metrics = service.run_backtest(session, target_key, n_folds=n_folds)
        result["backtest"] = [
            {
                "horizon_days": item.horizon_days,
                "n_folds": item.n_folds,
                "point_skill": (
                    round(item.best_point_skill, 4)
                    if item.best_point_skill is not None
                    else None
                ),
                "coverage_80": round(item.coverage_80, 3),
                "has_edge": item.has_edge,
            }
            for item in metrics
        ]
    return result


def _queue_history(session: Session, port_code: str) -> list[tuple[date, int]]:
    since = date.today() - timedelta(days=QUEUE_HISTORY_DAYS)
    rows = session.execute(
        select(PortCallSnapshot.snapshot_date, func.count())
        .where(
            PortCallSnapshot.port_code == port_code,
            PortCallSnapshot.status == "waiting",
            PortCallSnapshot.snapshot_date >= since,
        )
        .group_by(PortCallSnapshot.snapshot_date)
        .order_by(PortCallSnapshot.snapshot_date)
    ).all()
    return [(row[0], int(row[1])) for row in rows]


def _freshness(session: Session) -> list[dict]:
    subquery = (
        select(IngestRun.adapter, func.max(IngestRun.id).label("latest_id"))
        .group_by(IngestRun.adapter)
        .subquery()
    )
    rows = session.scalars(
        select(IngestRun).join(subquery, IngestRun.id == subquery.c.latest_id)
    ).all()
    return [
        {
            "adapter": row.adapter,
            "status": row.status,
            "used_cache": row.used_cache,
            "message": row.message,
            "finished_at": row.finished_at.date() if row.finished_at else None,
        }
        for row in rows
    ]


def _recent_cyclone_points(session: Session, as_of: date) -> list[dict]:
    cutoff = as_of - timedelta(days=alert_rules.CYCLONE_LOOKBACK_DAYS + 1)
    rows = session.scalars(
        select(CycloneTrackPoint).where(CycloneTrackPoint.timestamp >= cutoff)
    ).all()
    return [
        {
            "date": row.timestamp.date(),
            "latitude": row.latitude,
            "longitude": row.longitude,
            "max_wind_kt": row.max_wind_kt,
            "storm_id": row.storm_id,
            "storm_name": row.storm_name,
        }
        for row in rows
    ]


def _coal_stock_summary(session: Session) -> tuple[float | None, int, date | None]:
    latest = session.scalar(select(func.max(CoalStockObservation.obs_date)))
    if latest is None:
        return None, 0, None

    rows = session.scalars(
        select(CoalStockObservation).where(
            CoalStockObservation.obs_date == latest,
            CoalStockObservation.is_import_based.is_(True),
            CoalStockObservation.days_of_burn.isnot(None),
        )
    ).all()
    if not rows:
        return None, 0, latest

    import statistics

    burns = [row.days_of_burn for row in rows]
    return (
        float(statistics.median(burns)),
        sum(1 for value in burns if value < 5),
        latest,
    )


def _spread_history(session: Session) -> tuple[float | None, float | None, list[float]]:
    rows = session.execute(
        select(
            RateObservation.obs_date,
            RateObservation.series_code,
            RateObservation.value,
        ).where(RateObservation.series_code.in_(("COAL_AU_NEWC", "COAL_ZA_RB")))
    ).all()

    by_date: dict[date, dict[str, float]] = {}
    for obs_date, code, value in rows:
        by_date.setdefault(obs_date, {})[code] = value

    spreads: list[float] = []
    newc = richards = None
    for obs_date in sorted(by_date):
        entry = by_date[obs_date]
        if "COAL_AU_NEWC" in entry and "COAL_ZA_RB" in entry:
            spreads.append(entry["COAL_AU_NEWC"] - entry["COAL_ZA_RB"])
            newc, richards = entry["COAL_AU_NEWC"], entry["COAL_ZA_RB"]
    return newc, richards, spreads[-120:]


def run_alerts(session: Session, *, as_of: date | None = None) -> dict:
    """Evaluate every rule and reconcile against what is already open."""
    as_of = as_of or date.today()
    drafts: list[alert_rules.AlertDraft] = []

    target = choose_target(session)
    if target:
        volatility, regime = service.market_state(session, target)
        drafts += alert_rules.market_volatility_alerts(target, volatility, regime)
        drafts += alert_rules.regime_shift_alerts(target, regime)

    for port_code in CONGESTION_PORTS:
        history = _queue_history(session, port_code)
        forecast = service.waiting_forecast(session, port_code)
        drafts += alert_rules.congestion_alerts(port_code, history, forecast)

    median_burn, critical_count, obs_date = _coal_stock_summary(session)
    drafts += alert_rules.demand_urgency_alerts(
        median_burn, critical_count, obs_date=obs_date
    )

    ports = [
        {
            "code": port["code"],
            "name": port["name"],
            "latitude": port.get("latitude"),
            "longitude": port.get("longitude"),
        }
        for port in load_ports_list()
        if port.get("role") in {"discharge", "transhipment"}
    ]
    drafts += alert_rules.cyclone_alerts(
        _recent_cyclone_points(session, as_of), ports, as_of=as_of
    )

    newc, richards, spreads = _spread_history(session)
    drafts += alert_rules.price_spread_alerts(newc, richards, spreads)

    drafts += alert_rules.data_health_alerts(_freshness(session), as_of=as_of)

    summary = alert_rules.sync_alerts(session, drafts, as_of=as_of)
    summary["evaluated"] = len(drafts)
    return summary


def monitor_status(session: Session, *, as_of: date | None = None) -> list[dict]:
    """Current reading of every monitor, whether or not it fired.

    An empty alert list is the *good* state, but it makes a useless screen: the
    reader cannot tell the difference between "nothing is wrong" and "nothing
    is being watched". This returns the live reading of each monitor next to
    the threshold that would trip it, so a quiet day still shows the watch
    working.
    """
    as_of = as_of or date.today()
    monitors: list[dict] = []

    target = choose_target(session)
    if target:
        volatility, regime = service.market_state(session, target)
        if volatility:
            monitors.append(
                {
                    "code": "MARKET_VOL",
                    "label": f"Freight volatility ({target})",
                    "value": round(volatility.percentile_of_history, 3),
                    "display": f"{volatility.current_annual_vol:.0%} annualised",
                    "threshold": alert_rules.VOL_PERCENTILE_WARNING,
                    "threshold_label": "90th percentile of own history",
                    "triggered": volatility.percentile_of_history
                    >= alert_rules.VOL_PERCENTILE_WARNING,
                }
            )
        if regime:
            monitors.append(
                {
                    "code": "REGIME_SHIFT",
                    "label": f"Market regime ({target})",
                    "value": regime.days_since_change,
                    "display": (
                        f"{regime.regime}, last break "
                        f"{regime.days_since_change} days ago"
                        if regime.days_since_change is not None
                        else regime.regime
                    ),
                    "threshold": 21,
                    "threshold_label": "break within 21 days",
                    "triggered": regime.changed_recently,
                }
            )

    for port_code in CONGESTION_PORTS:
        history = _queue_history(session, port_code)
        if len(history) >= 20:
            counts = [count for _, count in history]
            baseline = counts[:-1]
            mean = sum(baseline) / len(baseline)
            spread = (
                sum((value - mean) ** 2 for value in baseline) / len(baseline)
            ) ** 0.5
            z = (counts[-1] - mean) / spread if spread > 0 else 0.0
            monitors.append(
                {
                    "code": "PORT_CONGESTION",
                    "label": f"Queue length at {port_code}",
                    "value": round(z, 2),
                    "display": f"{counts[-1]} waiting vs {mean:.1f} average",
                    "threshold": alert_rules.CONGESTION_Z_WATCH,
                    "threshold_label": "1.0 sd above own average",
                    "triggered": z >= alert_rules.CONGESTION_Z_WATCH,
                }
            )

        forecast = service.waiting_forecast(session, port_code)
        if forecast:
            monitors.append(
                {
                    "code": "PORT_CONGESTION",
                    "label": f"Predicted detention at {port_code}",
                    "value": round(forecast.expected_days, 1),
                    "display": (
                        f"{forecast.expected_days:.1f} days expected, "
                        f"{forecast.p80_days:.0f} at p80"
                    ),
                    "threshold": alert_rules.WAITING_DAYS_WARNING,
                    "threshold_label": f"{alert_rules.WAITING_DAYS_WARNING:.0f} days expected",
                    "triggered": forecast.expected_days
                    >= alert_rules.WAITING_DAYS_WARNING,
                }
            )

    median_burn, critical_count, obs_date = _coal_stock_summary(session)
    if median_burn is not None:
        monitors.append(
            {
                "code": "DEMAND_URGENCY",
                "label": "Import-plant coal stocks",
                "value": round(median_burn, 1),
                "display": (
                    f"{median_burn:.1f} days of burn, "
                    f"{critical_count} plants below five"
                ),
                "threshold": alert_rules.DAYS_OF_BURN_WATCH,
                "threshold_label": "below 7 days of burn",
                "triggered": median_burn <= alert_rules.DAYS_OF_BURN_WATCH,
            }
        )

    storms = _recent_cyclone_points(session, as_of)
    monitors.append(
        {
            "code": "CYCLONE",
            "label": "Bay of Bengal cyclone activity",
            "value": len(storms),
            "display": (
                f"{len({point['storm_id'] for point in storms})} active systems"
                if storms
                else "no active systems"
            ),
            "threshold": 1,
            "threshold_label": f"track within {alert_rules.CYCLONE_RADIUS_KM:.0f} km of a port",
            "triggered": bool(storms),
        }
    )

    freshness = _freshness(session)
    unhealthy = [
        row
        for row in freshness
        if row["status"] in {"failed", "stale"} or row["used_cache"]
    ]
    monitors.append(
        {
            "code": "DATA_HEALTH",
            "label": "Feed health",
            "value": len(unhealthy),
            "display": (
                f"{len(freshness) - len(unhealthy)} of {len(freshness)} feeds current"
            ),
            "threshold": 1,
            "threshold_label": "any feed stale, failed or cached",
            "triggered": bool(unhealthy),
        }
    )

    return monitors


def run_all(
    session: Session, *, backtest: bool = True, n_folds: int = 8
) -> dict:
    """The whole daily pipeline, with each stage isolated from the others."""
    results: dict = {}

    try:
        results["waiting"] = rebuild_waiting_times(session)
        session.commit()
    except Exception as exc:
        results["waiting"] = {"error": f"{type(exc).__name__}: {exc}"}
        session.rollback()

    try:
        results["models"] = run_models(session, backtest=backtest, n_folds=n_folds)
        session.commit()
    except Exception as exc:
        results["models"] = {"error": f"{type(exc).__name__}: {exc}"}
        session.rollback()

    try:
        results["alerts"] = run_alerts(session)
        session.commit()
    except Exception as exc:
        results["alerts"] = {"error": f"{type(exc).__name__}: {exc}"}
        session.rollback()

    return results
