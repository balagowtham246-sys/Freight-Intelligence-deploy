"""Risk Mitigation: early warning of things that change a chartering decision.

This answers deliverable (d) of the problem statement: *provide early warnings
for potential market volatility, port congestion, or other disruptions that
could impact chartering decisions*.

Design
------
Each rule is a **pure function over data already held**, returning zero or more
alerts. No rule fetches anything, so the whole engine is testable against
constructed inputs exactly as ``feasibility.py`` is.

Every alert carries four things, and the fourth is the one that matters:

* ``severity``  - how loud
* ``body``      - what was observed, with the numbers
* ``evidence``  - the figures and their source, so it can be checked
* ``action``    - **what it implies for a chartering decision**

An alert without an action is noise. "Paradip queue is up" is a fact; "expected
demurrage is up $0.40/t, shift the laycan or accept it in the freight" is a
decision. Every rule below is written to produce the second kind.

Persistence
-----------
Alerts are stored, not computed on request, so an alert that fired and later
cleared is a testable prediction. After a season the precision of each rule is
measurable from the ``alerts`` table, which is the only honest way to decide
whether a rule earns its place. ``sync_alerts`` handles the raise/refresh/clear
lifecycle idempotently.

Thresholds
----------
Stated as module constants with the reasoning attached. They are z-scores and
percentiles against each series' own history rather than absolute levels,
because an absolute threshold fires permanently on a volatile lane and never
on a quiet one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

# --- thresholds, with reasons -------------------------------------------------

# Volatility is "elevated" in the top decile of its own history. A decile is
# frequent enough to be actionable and rare enough not to be wallpaper.
VOL_PERCENTILE_WARNING = 0.90
VOL_PERCENTILE_CRITICAL = 0.97

# Queue length z-score against its own trailing mean.
CONGESTION_Z_WATCH = 1.0
CONGESTION_Z_WARNING = 1.5
CONGESTION_Z_CRITICAL = 2.5

# Predicted pre-berthing detention above which demurrage stops being noise.
WAITING_DAYS_WARNING = 7.0
WAITING_DAYS_CRITICAL = 12.0

# CEA days-of-burn. Indian norms treat 7 days as comfortable at coastal
# import plants and under 4 as critical.
DAYS_OF_BURN_WATCH = 7.0
DAYS_OF_BURN_WARNING = 5.0
DAYS_OF_BURN_CRITICAL = 3.0

# A cyclone track point within this distance of a port, in the last few days,
# is a live berth-closure risk.
CYCLONE_RADIUS_KM = 500.0
CYCLONE_LOOKBACK_DAYS = 5

# A feed silent for longer than this is stale enough to mislead.
FEED_STALE_DAYS = 3

SEVERITY_ORDER = {"info": 0, "watch": 1, "warning": 2, "critical": 3}


@dataclass
class AlertDraft:
    """One alert, before it is reconciled against what is already stored."""

    code: str
    severity: str
    scope: str
    title: str
    body: str
    action: str
    port_code: str | None = None
    series_code: str | None = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "scope": self.scope,
            "title": self.title,
            "body": self.body,
            "action": self.action,
            "port_code": self.port_code,
            "series_code": self.series_code,
            "evidence": self.evidence,
        }


def _severity_for(value: float, watch: float, warning: float, critical: float) -> str | None:
    """Ascending thresholds."""
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    if value >= watch:
        return "watch"
    return None


def _severity_for_descending(
    value: float, watch: float, warning: float, critical: float
) -> str | None:
    """Descending thresholds, for signals where low is bad."""
    if value <= critical:
        return "critical"
    if value <= warning:
        return "warning"
    if value <= watch:
        return "watch"
    return None


# --- rules --------------------------------------------------------------------


def market_volatility_alerts(
    series_code: str, volatility, regime=None
) -> list[AlertDraft]:
    """MARKET_VOL: conditional volatility high against its own history."""
    if volatility is None:
        return []

    percentile = volatility.percentile_of_history
    if percentile < VOL_PERCENTILE_WARNING:
        return []

    severity = "critical" if percentile >= VOL_PERCENTILE_CRITICAL else "warning"
    persistence = volatility.persistence
    sticky = np.isfinite(persistence) and persistence > 0.95

    return [
        AlertDraft(
            code="MARKET_VOL",
            severity=severity,
            scope=series_code,
            series_code=series_code,
            title=f"Freight volatility elevated on {series_code}",
            body=(
                f"Conditional volatility is {volatility.current_annual_vol:.0%} "
                f"annualised, in the {percentile:.0%} percentile of this series' own "
                f"history, against a long-run {volatility.long_run_annual_vol:.0%}. "
                f"The {volatility.model_name} model forecasts "
                f"{volatility.forecast_annual_vol:.0%} over the next month"
                + (
                    ", and with persistence above 0.95 the turbulence is likely to "
                    "still be here next month."
                    if sticky
                    else "."
                )
            ),
            action=(
                "Widening rate distributions cut both ways: the option value of "
                "waiting is up, but so is the cost of being wrong. Narrow the laycan, "
                "size any single fixture smaller, and look harder at the term cover "
                "point on the contract-mix frontier - this is the market state term "
                "contracts exist for."
            ),
            evidence={
                "current_annual_vol": round(volatility.current_annual_vol, 4),
                "forecast_annual_vol": round(volatility.forecast_annual_vol, 4),
                "percentile_of_history": round(percentile, 4),
                "persistence": (
                    round(float(persistence), 4) if np.isfinite(persistence) else None
                ),
                "model": volatility.model_name,
                "n_observations": volatility.n_observations,
            },
        )
    ]


def regime_shift_alerts(series_code: str, regime) -> list[AlertDraft]:
    """REGIME_SHIFT: the market just moved to a different footing."""
    if regime is None or not regime.changed_recently:
        return []

    return [
        AlertDraft(
            code="REGIME_SHIFT",
            severity="warning",
            scope=series_code,
            series_code=series_code,
            title=f"Regime change detected on {series_code}",
            body=(
                f"The {regime.detector} detector flagged a break "
                f"{regime.days_since_change} days ago "
                f"({regime.last_change_date.isoformat() if regime.last_change_date else 'date unknown'}). "
                f"The market now reads as {regime.regime} with "
                f"{regime.probability_turbulent:.0%} probability of the "
                f"high-dispersion state."
            ),
            action=(
                "The forecaster was fitted on the previous regime and its calibrated "
                "intervals assume the future resembles the calibration period. Treat "
                "the fan chart as provisional until the model has refitted through "
                "the break; the platform downgrades signal strength to 'weak' "
                "automatically while this is live."
            ),
            evidence=regime.as_dict(),
        )
    ]


def congestion_alerts(
    port_code: str,
    queue_history: list[tuple[date, int]],
    waiting_forecast=None,
    *,
    demurrage_usd_per_day: float = 22_000.0,
    parcel_tonnes: float = 50_000.0,
) -> list[AlertDraft]:
    """PORT_CONGESTION: the queue is unusual, or the predicted wait is costly.

    Two distinct triggers, deliberately. The z-score catches a queue that is
    building relative to normal; the predicted-detention threshold catches a
    queue that is *expensive* even if it is normal for this port.
    """
    drafts: list[AlertDraft] = []
    if len(queue_history) >= 20:
        counts = np.array([count for _, count in queue_history], dtype=float)
        current = counts[-1]
        baseline = counts[:-1]
        mean, spread = float(baseline.mean()), float(baseline.std())
        # Floor the spread at one vessel. A port whose queue is very stable has
        # a near-zero standard deviation, and dividing by it would either
        # explode or - worse, since we guarded against that - silently return
        # z = 0 and never alert however large the jump. Queue counts are whole
        # vessels, so dispersion below one is not meaningfully resolvable.
        z = (current - mean) / max(spread, 1.0)

        severity = _severity_for(
            z, CONGESTION_Z_WATCH, CONGESTION_Z_WARNING, CONGESTION_Z_CRITICAL
        )
        if severity:
            recent = counts[-7:] if counts.size >= 7 else counts
            slope = float(recent[-1] - recent[0])
            drafts.append(
                AlertDraft(
                    code="PORT_CONGESTION",
                    severity=severity,
                    scope=port_code,
                    port_code=port_code,
                    title=f"Queue at {port_code} above normal",
                    body=(
                        f"{current:.0f} vessels waiting against a {mean:.1f} average "
                        f"over the {len(baseline)} days archived "
                        f"(z = {z:+.1f}), and {slope:+.0f} over the last week."
                    ),
                    action=(
                        "Congestion absorbs tonnage and lengthens pre-berthing "
                        "detention. Expect demurrage exposure to rise and check "
                        "whether an alternative discharge port or a later laycan "
                        "clears the peak before committing."
                    ),
                    evidence={
                        "waiting_now": int(current),
                        "baseline_mean": round(mean, 2),
                        "z_score": round(z, 2),
                        "weekly_change": slope,
                        "days_archived": len(queue_history),
                    },
                )
            )

    if waiting_forecast is not None:
        expected = waiting_forecast.expected_days
        severity = _severity_for(
            expected,
            WAITING_DAYS_WARNING,
            WAITING_DAYS_WARNING,
            WAITING_DAYS_CRITICAL,
        )
        if severity:
            laytime = max(parcel_tonnes / 25_000.0, 1.0)
            demurrage_days = max(waiting_forecast.p80_days - laytime, 0.0)
            cost_per_t = demurrage_days * demurrage_usd_per_day / parcel_tonnes
            drafts.append(
                AlertDraft(
                    code="PORT_CONGESTION",
                    severity=severity,
                    scope=f"{port_code}:detention",
                    port_code=port_code,
                    title=f"Predicted detention at {port_code} is costly",
                    body=(
                        f"The waiting-time model puts expected pre-berthing detention "
                        f"at {expected:.1f} days, {waiting_forecast.p80_days:.0f} days "
                        f"at the 80th percentile, for a vessel joining the queue now."
                    ),
                    action=(
                        f"At the 80th percentile that is roughly "
                        f"${cost_per_t:.2f}/t of demurrage on a "
                        f"{parcel_tonnes:,.0f} t parcel. Price it into the freight "
                        f"comparison rather than treating it as an overrun - it is "
                        f"often larger than the difference between two routings."
                    ),
                    evidence=waiting_forecast.as_dict(),
                )
            )

    return drafts


def demand_urgency_alerts(
    median_days_of_burn: float | None,
    critical_plant_count: int,
    *,
    obs_date: date | None = None,
) -> list[AlertDraft]:
    """DEMAND_URGENCY: our own coal stocks are running down."""
    if median_days_of_burn is None:
        return []

    severity = _severity_for_descending(
        median_days_of_burn,
        DAYS_OF_BURN_WATCH,
        DAYS_OF_BURN_WARNING,
        DAYS_OF_BURN_CRITICAL,
    )
    if severity is None:
        return []

    return [
        AlertDraft(
            code="DEMAND_URGENCY",
            severity=severity,
            scope="cea_import_plants",
            title="Import-coal plant stocks running down",
            body=(
                f"Median days-of-burn across the import-designed plants is "
                f"{median_days_of_burn:.1f} days"
                + (
                    f", with {critical_plant_count} plants below five days"
                    if critical_plant_count
                    else ""
                )
                + f" as of {obs_date.isoformat() if obs_date else 'the latest report'}."
            ),
            action=(
                "This is our own urgency, not the market's, and it cuts the value of "
                "waiting: a buyer who must take cargo has no option to exercise. "
                "Expect the timing engine to favour fixing sooner, and treat a firm "
                "market as something to hedge rather than to wait out."
            ),
            evidence={
                "median_days_of_burn": round(median_days_of_burn, 2),
                "critical_plant_count": critical_plant_count,
                "obs_date": obs_date.isoformat() if obs_date else None,
            },
        )
    ]


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def cyclone_alerts(
    track_points: list[dict],
    ports: list[dict],
    *,
    as_of: date,
) -> list[AlertDraft]:
    """CYCLONE: an active storm within reach of a discharge port.

    ``track_points`` are recent best-track fixes; ``ports`` carry code, name,
    latitude and longitude. Only points inside the lookback window count, so a
    historical storm at the same position never fires.
    """
    drafts: list[AlertDraft] = []
    cutoff = as_of - timedelta(days=CYCLONE_LOOKBACK_DAYS)

    recent = [
        point
        for point in track_points
        if point.get("date") and cutoff <= point["date"] <= as_of
    ]
    if not recent:
        return drafts

    for port in ports:
        latitude, longitude = port.get("latitude"), port.get("longitude")
        if latitude is None or longitude is None:
            continue

        nearest = None
        for point in recent:
            distance = _haversine_km(
                latitude, longitude, point["latitude"], point["longitude"]
            )
            if distance <= CYCLONE_RADIUS_KM and (
                nearest is None or distance < nearest[0]
            ):
                nearest = (distance, point)

        if nearest is None:
            continue

        distance, point = nearest
        wind = point.get("max_wind_kt")
        severity = "critical" if wind and wind >= 64 else "warning"
        drafts.append(
            AlertDraft(
                code="CYCLONE",
                severity=severity,
                scope=f"{port['code']}:cyclone",
                port_code=port["code"],
                title=f"Cyclone within {distance:.0f} km of {port.get('name', port['code'])}",
                body=(
                    f"{point.get('storm_name') or 'An unnamed system'} was "
                    f"{distance:.0f} km from {port.get('name', port['code'])} on "
                    f"{point['date'].isoformat()}"
                    + (f", maximum sustained wind {wind:.0f} kt." if wind else ".")
                ),
                action=(
                    "East coast berths close ahead of a system and the queue rebuilds "
                    "behind it. Expect berth downtime inside the laycan, a spike in "
                    "pre-berthing detention afterwards, and price the demurrage "
                    "exposure accordingly."
                ),
                evidence={
                    "distance_km": round(distance, 1),
                    "storm_id": point.get("storm_id"),
                    "storm_name": point.get("storm_name"),
                    "max_wind_kt": wind,
                    "observed": point["date"].isoformat(),
                },
            )
        )

    return drafts


def price_spread_alerts(
    newc: float | None,
    richards_bay: float | None,
    history_spread: list[float] | None = None,
) -> list[AlertDraft]:
    """PRICE_SPREAD: the Pacific/Atlantic coal arbitrage has moved."""
    if newc is None or richards_bay is None or not history_spread:
        return []

    spread = newc - richards_bay
    values = np.array([v for v in history_spread if np.isfinite(v)], dtype=float)
    if values.size < 24:
        return []

    mean, deviation = float(values.mean()), float(values.std())
    if deviation <= 0:
        return []
    z = (spread - mean) / deviation
    if abs(z) < 2.0:
        return []

    direction = "Australian coal expensive against South African" if z > 0 else "Australian coal cheap against South African"
    return [
        AlertDraft(
            code="PRICE_SPREAD",
            severity="watch",
            scope="COAL_NEWC_RB",
            series_code="COAL_AU_NEWC",
            title="Coal arbitrage spread outside its normal band",
            body=(
                f"The Newcastle minus Richards Bay spread is ${spread:.2f}/t "
                f"({z:+.1f} standard deviations from its {len(values)}-month mean of "
                f"${mean:.2f}/t). {direction}."
            ),
            action=(
                "Arbitrage at this level redirects cargo between basins, and tonnage "
                "follows the cargo. Expect the origin mix to shift and the relative "
                "value of Pacific versus Atlantic tonnage to move with it."
            ),
            evidence={
                "spread": round(spread, 2),
                "z_score": round(z, 2),
                "mean": round(mean, 2),
                "months": int(values.size),
            },
        )
    ]


def data_health_alerts(
    freshness: list[dict], *, as_of: date
) -> list[AlertDraft]:
    """DATA_HEALTH: a feed is stale, failed, or silently serving cache.

    Not padding. ``SourceAdapter.fetch`` is designed to fall back to the last
    known-good payload rather than take the platform down, which is right - but
    it means every number on every screen can quietly go stale with no visible
    change. This rule is what makes that visible.
    """
    drafts: list[AlertDraft] = []
    for entry in freshness:
        adapter = entry.get("adapter", "unknown")
        status = entry.get("status")
        finished = entry.get("finished_at")

        age_days = None
        if isinstance(finished, date):
            age_days = (as_of - finished).days

        stale = status in {"failed", "stale"} or entry.get("used_cache")
        overdue = age_days is not None and age_days > FEED_STALE_DAYS

        if not (stale or overdue):
            continue

        severity = "warning" if status == "failed" else "watch"
        reasons = []
        if status == "failed":
            reasons.append("the last run failed")
        elif status == "stale" or entry.get("used_cache"):
            reasons.append("it is serving a cached payload")
        if overdue:
            reasons.append(f"the last successful run was {age_days} days ago")

        drafts.append(
            AlertDraft(
                code="DATA_HEALTH",
                severity=severity,
                scope=f"feed:{adapter}",
                title=f"Feed {adapter} is not current",
                body=(
                    f"The {adapter} adapter is not current: {' and '.join(reasons)}."
                    + (f" Last message: {entry['message']}" if entry.get("message") else "")
                ),
                action=(
                    "Every figure downstream of this feed is running on its last "
                    "known-good value. Treat forecasts and alerts derived from it as "
                    "indicative until the feed recovers, and do not read a flat line "
                    "on a chart as a quiet market."
                ),
                evidence={
                    "adapter": adapter,
                    "status": status,
                    "used_cache": entry.get("used_cache"),
                    "age_days": age_days,
                },
            )
        )
    return drafts


# --- persistence --------------------------------------------------------------


def sync_alerts(session, drafts: list[AlertDraft], *, as_of: date) -> dict:
    """Reconcile drafts against stored alerts: raise, refresh, clear.

    An alert is identified by ``(code, scope)``. Re-raising an alert that is
    already open just moves its ``last_seen_date`` forward, so the ``alerts``
    table records episodes rather than one row per day. An open alert whose
    condition no longer holds is cleared with today's date, which is what makes
    rule precision measurable later.
    """
    from ..models import Alert
    from sqlalchemy import select

    open_alerts = {
        (row.code, row.scope): row
        for row in session.scalars(
            select(Alert).where(Alert.cleared_date.is_(None))
        ).all()
    }
    drafted = {(draft.code, draft.scope): draft for draft in drafts}

    raised = refreshed = cleared = 0

    for key, draft in drafted.items():
        existing = open_alerts.get(key)
        if existing is None:
            session.add(
                Alert(
                    code=draft.code,
                    severity=draft.severity,
                    scope=draft.scope,
                    port_code=draft.port_code,
                    series_code=draft.series_code,
                    raised_date=as_of,
                    last_seen_date=as_of,
                    title=draft.title,
                    body=draft.body,
                    action=draft.action,
                    evidence_json=json.dumps(draft.evidence, default=str),
                )
            )
            raised += 1
        else:
            existing.last_seen_date = as_of
            existing.severity = draft.severity
            existing.title = draft.title
            existing.body = draft.body
            existing.action = draft.action
            existing.evidence_json = json.dumps(draft.evidence, default=str)
            refreshed += 1

    for key, row in open_alerts.items():
        if key not in drafted:
            row.cleared_date = as_of
            cleared += 1

    session.flush()
    return {"raised": raised, "refreshed": refreshed, "cleared": cleared}
