"""Waiting-time derivation and the hazard model.

The censoring behaviour is the point of these tests. With a 73-day archive
nearly a fifth of port calls are unresolved, and every wrong way of handling
them biases the estimate downward -- which is the direction that would make
the platform under-price demurrage risk.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.ml import waiting, waiting_model


class FakeSnapshot:
    """Stand-in for PortCallSnapshot, so these tests need no database."""

    def __init__(
        self,
        snapshot_date: date,
        vessel_name: str,
        status: str,
        arrival_date: date | None = None,
        *,
        port_code: str = "INPRT",
        cargo: str = "TH.COAL",
        cargo_tonnes: float = 70_000.0,
    ):
        self.snapshot_date = snapshot_date
        self.vessel_name = vessel_name
        self.status = status
        self.arrival_date = arrival_date
        self.port_code = port_code
        self.cargo = cargo
        self.cargo_tonnes = cargo_tonnes
        self.source_name = "paradip_daily_traffic"
        self.source_url = "https://www.paradipport.gov.in/"
        self.retrieved_at = None
        self.is_synthetic = False


def _day(offset: int) -> date:
    return date(2026, 6, 1) + timedelta(days=offset)


def test_berthed_vessel_yields_an_observed_event():
    arrival = _day(0)
    rows = [FakeSnapshot(_day(i), "MV. ALPHA", "waiting", arrival) for i in range(0, 4)]
    rows += [FakeSnapshot(_day(i), "MV. ALPHA", "working") for i in (4, 5)]

    observations = waiting.derive_observations(rows)
    assert len(observations) == 1
    record = observations[0]
    assert record["censored"] is False
    assert record["berthed_date"] == _day(4)
    assert record["waiting_days"] == 4.0


def test_vessel_still_waiting_at_the_end_is_censored_not_berthed():
    """The bug this guards against would silently shorten every estimate."""
    arrival = _day(0)
    rows = [FakeSnapshot(_day(i), "MV. BRAVO", "waiting", arrival) for i in range(0, 6)]

    observations = waiting.derive_observations(rows)
    assert len(observations) == 1
    record = observations[0]
    assert record["censored"] is True
    assert record["berthed_date"] is None
    # A lower bound, never zero and never dropped.
    assert record["waiting_days"] == 5.0


def test_vessel_that_vanished_without_berthing_is_censored():
    """It may have berthed in an archive gap or left. We did not see it berth."""
    arrival = _day(0)
    rows = [FakeSnapshot(_day(i), "MV. CHARLIE", "waiting", arrival) for i in range(0, 3)]
    # Other vessels keep the archive running well past Charlie's last sighting.
    rows += [FakeSnapshot(_day(i), "MV. OTHER", "waiting", _day(i)) for i in range(3, 10)]

    observations = waiting.derive_observations(rows)
    charlie = [o for o in observations if o["vessel_name"] == "MV. CHARLIE"][0]
    assert charlie["censored"] is True
    assert charlie["berthed_date"] is None


def test_repeat_calls_by_the_same_vessel_are_separate_observations():
    """A vessel calls several times a season; the unit is the call, not the ship."""
    rows = [FakeSnapshot(_day(i), "MV. DELTA", "waiting", _day(0)) for i in (0, 1)]
    rows += [FakeSnapshot(_day(2), "MV. DELTA", "working")]
    rows += [FakeSnapshot(_day(i), "MV. DELTA", "waiting", _day(20)) for i in (20, 21)]
    rows += [FakeSnapshot(_day(22), "MV. DELTA", "working")]

    observations = waiting.derive_observations(rows)
    assert len(observations) == 2
    assert {o["arrival_date"] for o in observations} == {_day(0), _day(20)}
    assert all(o["censored"] is False for o in observations)


def test_working_sighting_is_not_attributed_across_a_later_call():
    """The second call's berthing must not close the first call's window."""
    rows = [FakeSnapshot(_day(0), "MV. ECHO", "waiting", _day(0))]
    rows += [FakeSnapshot(_day(i), "MV. ECHO", "waiting", _day(30)) for i in (30, 31)]
    rows += [FakeSnapshot(_day(32), "MV. ECHO", "working")]

    observations = waiting.derive_observations(rows)
    first = [o for o in observations if o["arrival_date"] == _day(0)][0]
    assert first["censored"] is True, "a later call's berthing closed an earlier one"


def test_queue_length_falls_back_to_a_nearby_snapshot_but_not_before_the_archive():
    arrival = _day(5)
    rows = [FakeSnapshot(_day(i), f"MV. Q{i}", "waiting", _day(i)) for i in (4, 6, 7)]
    rows += [FakeSnapshot(_day(i), "MV. FOXTROT", "waiting", arrival) for i in (6, 7)]

    observations = waiting.derive_observations(rows)
    foxtrot = [o for o in observations if o["vessel_name"] == "MV. FOXTROT"][0]
    # Day 5 has no snapshot; the nearest within tolerance is used.
    assert foxtrot["queue_len_on_arrival"] is not None


def test_person_periods_expand_correctly_and_censored_rows_carry_no_event():
    frame = pd.DataFrame(
        [
            {"waiting_days": 3, "censored": False, "queue_len_on_arrival": 10,
             "cargo_tonnes": 70000, "berth_date_uncertain_days": 0},
            {"waiting_days": 5, "censored": True, "queue_len_on_arrival": 20,
             "cargo_tonnes": 50000, "berth_date_uncertain_days": 0},
        ]
    )
    periods = waiting_model.build_person_periods(frame)
    assert len(periods) == 3 + 5
    # Exactly one event, on the last day of the uncensored call.
    assert periods["berthed"].sum() == 1


def test_hazard_model_predicts_longer_waits_for_longer_queues():
    """The whole commercial point: congestion should lengthen predicted detention."""
    rows = []
    for i in range(120):
        queue = 5 if i % 2 == 0 else 35
        waited = 3 if queue == 5 else 9
        rows.append(
            {
                "waiting_days": waited,
                "censored": False,
                "queue_len_on_arrival": queue,
                "cargo_tonnes": 60_000,
                "berth_date_uncertain_days": 0,
            }
        )
    fitted = waiting_model.fit_hazard(pd.DataFrame(rows))
    assert fitted is not None

    quiet = waiting_model.predict_waiting(
        fitted, port_code="INPRT", queue_length=5, tonnes=60_000
    )
    busy = waiting_model.predict_waiting(
        fitted, port_code="INPRT", queue_length=35, tonnes=60_000
    )
    assert busy.expected_days > quiet.expected_days
    assert busy.p95_days >= quiet.p95_days


def test_survival_curve_is_monotone_non_increasing():
    rows = [
        {"waiting_days": (i % 8) + 1, "censored": i % 5 == 0,
         "queue_len_on_arrival": 15, "cargo_tonnes": 60_000,
         "berth_date_uncertain_days": 0}
        for i in range(100)
    ]
    fitted = waiting_model.fit_hazard(pd.DataFrame(rows))
    curve = waiting_model.survival_curve(fitted, queue_length=15, tonnes=60_000)
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))
    assert curve[0] <= 1.0


def test_fit_returns_none_rather_than_a_model_on_too_little_data():
    frame = pd.DataFrame(
        [
            {"waiting_days": 2, "censored": False, "queue_len_on_arrival": 8,
             "cargo_tonnes": 50_000, "berth_date_uncertain_days": 0}
        ]
    )
    assert waiting_model.fit_hazard(frame) is None
