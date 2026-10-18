"""Reconstruct realised pre-berthing detention from archived berthing programmes.

This is a derivation, not an ingestion: it reads ``port_call_snapshots``, which
are real scraped port-authority PDFs, and writes ``waiting_time_observations``.
No new information is invented, so the derived rows inherit the provenance of
the snapshots they came from.

The state machine in the source data
------------------------------------
A Paradip daily traffic report puts each vessel in one of three sections, and a
call moves through them::

    expected  ->  waiting  ->  working
    (ETA set)     (at anchorage,   (alongside, arrival_date
                   arrival_date     is no longer carried)
                   set)

A vessel makes several calls in a season, so the unit of observation is the
*port call*, keyed on ``(port_code, vessel_name, arrival_date)`` — not the
vessel.

Censoring
---------
Three outcomes, and only the first is an observed event:

1. Seen ``working`` after its waiting window closed. ``berthed_date`` is the
   first such sighting; ``censored=False``.
2. Still ``waiting`` in the final snapshot we hold. ``censored=True``, and
   ``waiting_days`` is a lower bound.
3. Disappeared from the queue without ever being seen ``working``.
   ``censored=True``, bounded at the last waiting sighting. It may have
   berthed during a gap in the archive, or it may have left; we do not know,
   and recording an event we did not observe would be a fabrication.

With a 73-day archive, cases 2 and 3 are a large share of the sample. Dropping
them, or treating them as berthed at the last sighting, biases every
waiting-time estimate downward — which is exactly the direction that would
make the platform under-price demurrage risk.

Snapshot gaps
-------------
The archive has missing days. When a gap straddles the moment a vessel berthed,
the first ``working`` sighting is an upper bound on the true berthing date.
``berth_date_uncertain_days`` records the size of that gap: zero means the date
is exact, and the hazard model down-weights or excludes the rest rather than
pretending to a precision the archive does not have.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PortCallSnapshot, WaitingTimeObservation

# A call is considered a different call from a previous one by the same vessel
# when its arrival date differs. Working sightings are attributed to the most
# recent waiting window that closed before them, provided the gap is not
# absurd - beyond this many days we assume it is an unrelated later call.
MAX_ATTRIBUTION_GAP_DAYS = 21


@dataclass
class PortCall:
    """One vessel's stay in the queue, assembled from consecutive snapshots."""

    port_code: str
    vessel_name: str
    arrival_date: date
    first_waiting_seen: date
    last_waiting_seen: date
    cargo: str | None = None
    cargo_tonnes: float | None = None
    source_name: str = ""
    source_url: str | None = None
    retrieved_at = None
    is_synthetic: bool = False


def _snapshot_dates(rows: list[PortCallSnapshot]) -> list[date]:
    return sorted({row.snapshot_date for row in rows})


def build_calls(rows: list[PortCallSnapshot]) -> list[PortCall]:
    """Collapse ``waiting`` snapshot rows into one record per port call."""
    calls: dict[tuple[str, str, date], PortCall] = {}

    for row in rows:
        if row.status != "waiting" or row.arrival_date is None:
            continue
        key = (row.port_code, row.vessel_name, row.arrival_date)
        call = calls.get(key)
        if call is None:
            calls[key] = PortCall(
                port_code=row.port_code,
                vessel_name=row.vessel_name,
                arrival_date=row.arrival_date,
                first_waiting_seen=row.snapshot_date,
                last_waiting_seen=row.snapshot_date,
                cargo=row.cargo,
                cargo_tonnes=row.cargo_tonnes,
                source_name=row.source_name,
                source_url=row.source_url,
                is_synthetic=row.is_synthetic,
            )
            calls[key].retrieved_at = row.retrieved_at
        else:
            call.first_waiting_seen = min(call.first_waiting_seen, row.snapshot_date)
            call.last_waiting_seen = max(call.last_waiting_seen, row.snapshot_date)
            if call.cargo is None:
                call.cargo = row.cargo
            if call.cargo_tonnes is None:
                call.cargo_tonnes = row.cargo_tonnes

    return sorted(calls.values(), key=lambda c: (c.vessel_name, c.arrival_date))


def _working_dates(rows: list[PortCallSnapshot]) -> dict[tuple[str, str], list[date]]:
    """Dates on which each (port, vessel) was seen alongside."""
    seen: dict[tuple[str, str], set[date]] = defaultdict(set)
    for row in rows:
        if row.status == "working":
            seen[(row.port_code, row.vessel_name)].add(row.snapshot_date)
    return {key: sorted(value) for key, value in seen.items()}


def _queue_lengths(rows: list[PortCallSnapshot]) -> dict[tuple[str, date], int]:
    counts: dict[tuple[str, date], int] = defaultdict(int)
    for row in rows:
        if row.status == "waiting":
            counts[(row.port_code, row.snapshot_date)] += 1
    return dict(counts)


def _queue_on(
    queues: dict[tuple[str, date], int],
    port_code: str,
    when: date,
    *,
    tolerance_days: int = 3,
) -> int | None:
    """Queue length on a date, falling back to the nearest snapshot nearby.

    The archive has missing days, so an exact lookup drops observations that
    are otherwise perfectly good. It does *not* fall back for arrivals before
    the archive begins: there is no snapshot to stand in for those, and
    guessing would put a fabricated feature into the training set.
    """
    exact = queues.get((port_code, when))
    if exact is not None:
        return exact
    for offset in range(1, tolerance_days + 1):
        for candidate in (
            when.fromordinal(when.toordinal() + offset),
            when.fromordinal(when.toordinal() - offset),
        ):
            nearby = queues.get((port_code, candidate))
            if nearby is not None:
                return nearby
    return None


def derive_observations(rows: list[PortCallSnapshot]) -> list[dict]:
    """Turn raw snapshots into censored waiting-time observations.

    Pure function over the snapshot rows, so it is testable without a database.
    """
    if not rows:
        return []

    calls = build_calls(rows)
    working = _working_dates(rows)
    queues = _queue_lengths(rows)
    all_dates = _snapshot_dates(rows)
    final_snapshot = all_dates[-1]

    # Next arrival by the same vessel, so a working sighting is never
    # attributed across a later call.
    next_arrival: dict[int, date | None] = {}
    by_vessel: dict[tuple[str, str], list[PortCall]] = defaultdict(list)
    for call in calls:
        by_vessel[(call.port_code, call.vessel_name)].append(call)
    for vessel_calls in by_vessel.values():
        vessel_calls.sort(key=lambda c: c.arrival_date)
        for index, call in enumerate(vessel_calls):
            following = vessel_calls[index + 1] if index + 1 < len(vessel_calls) else None
            next_arrival[id(call)] = following.arrival_date if following else None

    observations: list[dict] = []
    for call in calls:
        cutoff = next_arrival[id(call)]
        candidates = [
            day
            for day in working.get((call.port_code, call.vessel_name), [])
            if day >= call.last_waiting_seen
            and (cutoff is None or day < cutoff)
            and (day - call.last_waiting_seen).days <= MAX_ATTRIBUTION_GAP_DAYS
        ]

        if candidates:
            berthed = candidates[0]
            # How much archive gap sits between the last waiting sighting and
            # the first working sighting.
            gap = sum(
                1
                for day in all_dates
                if call.last_waiting_seen < day < berthed
            )
            uncertain = max(0, (berthed - call.last_waiting_seen).days - 1 - gap)
            observations.append(
                {
                    "port_code": call.port_code,
                    "vessel_name": call.vessel_name,
                    "arrival_date": call.arrival_date,
                    "berthed_date": berthed,
                    "waiting_days": float((berthed - call.arrival_date).days),
                    "censored": False,
                    "berth_date_uncertain_days": uncertain,
                    "queue_len_on_arrival": _queue_on(
                        queues, call.port_code, call.arrival_date
                    ),
                    "cargo": call.cargo,
                    "cargo_tonnes": call.cargo_tonnes,
                    "source_name": call.source_name,
                    "source_url": call.source_url,
                    "retrieved_at": call.retrieved_at,
                    "is_synthetic": call.is_synthetic,
                }
            )
            continue

        # Never seen alongside. Censored either way; the bound differs.
        bound = (
            final_snapshot
            if call.last_waiting_seen == final_snapshot
            else call.last_waiting_seen
        )
        observations.append(
            {
                "port_code": call.port_code,
                "vessel_name": call.vessel_name,
                "arrival_date": call.arrival_date,
                "berthed_date": None,
                "waiting_days": float((bound - call.arrival_date).days),
                "censored": True,
                "berth_date_uncertain_days": 0,
                "queue_len_on_arrival": _queue_on(
                    queues, call.port_code, call.arrival_date
                ),
                "cargo": call.cargo,
                "cargo_tonnes": call.cargo_tonnes,
                "source_name": call.source_name,
                "source_url": call.source_url,
                "retrieved_at": call.retrieved_at,
                "is_synthetic": call.is_synthetic,
            }
        )

    return observations


def rebuild_waiting_times(session: Session, port_code: str | None = None) -> dict:
    """Rebuild the waiting-time table from the snapshot archive. Idempotent."""
    query = select(PortCallSnapshot)
    if port_code:
        query = query.where(PortCallSnapshot.port_code == port_code)
    rows = list(session.scalars(query).all())

    observations = derive_observations(rows)
    if not observations:
        return {"calls": 0, "events": 0, "censored": 0}

    existing = session.scalars(select(WaitingTimeObservation)).all()
    by_key = {
        (row.port_code, row.vessel_name, row.arrival_date): row for row in existing
    }

    for record in observations:
        key = (record["port_code"], record["vessel_name"], record["arrival_date"])
        current = by_key.get(key)
        if current is None:
            session.add(WaitingTimeObservation(**record))
        else:
            for field, value in record.items():
                setattr(current, field, value)
    session.flush()

    events = sum(1 for record in observations if not record["censored"])
    return {
        "calls": len(observations),
        "events": events,
        "censored": len(observations) - events,
    }
