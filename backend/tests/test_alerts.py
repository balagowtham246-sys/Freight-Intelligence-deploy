"""The risk rules: each must fire when it should, stay silent when it should not.

Live data is usually quiet, so these are the tests that actually prove the risk
engine works. Every rule is a pure function over constructed inputs, in the
same style as the feasibility tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from app.domain import alerts


@dataclass
class FakeVolatility:
    current_annual_vol: float = 0.60
    forecast_annual_vol: float = 0.62
    long_run_annual_vol: float = 0.45
    percentile_of_history: float = 0.95
    persistence: float = 0.97
    model_name: str = "garch(1,1)-t"
    n_observations: int = 2000


@dataclass
class FakeRegime:
    regime: str = "turbulent"
    probability_turbulent: float = 0.85
    changed_recently: bool = True
    days_since_change: int = 4
    last_change_date: date = date(2026, 8, 30)
    detector: str = "markov+cusum"
    n_observations: int = 2000

    def as_dict(self):
        return {"regime": self.regime, "days_since_change": self.days_since_change}


@dataclass
class FakeWaiting:
    port_code: str = "INPRT"
    queue_length: float = 30.0
    expected_days: float = 11.0
    median_days: float = 10.0
    p80_days: float = 16.0
    p95_days: float = 22.0
    model_name: str = "discrete_hazard"
    n_train_calls: int = 400
    explanation: str = "test"

    def as_dict(self):
        return {"expected_days": self.expected_days, "p80_days": self.p80_days}


# --- MARKET_VOL ---------------------------------------------------------------


def test_market_vol_fires_when_volatility_is_in_the_top_decile():
    drafts = alerts.market_volatility_alerts("CDFI_X", FakeVolatility())
    assert len(drafts) == 1
    assert drafts[0].code == "MARKET_VOL"
    assert drafts[0].severity in {"warning", "critical"}
    assert drafts[0].action, "an alert without an action is noise"


def test_market_vol_is_silent_in_a_normal_market():
    calm = FakeVolatility(percentile_of_history=0.55)
    assert alerts.market_volatility_alerts("CDFI_X", calm) == []


def test_market_vol_escalates_to_critical_in_the_extreme():
    extreme = FakeVolatility(percentile_of_history=0.99)
    assert alerts.market_volatility_alerts("CDFI_X", extreme)[0].severity == "critical"


def test_market_vol_mentions_persistence_when_shocks_will_not_decay():
    sticky = FakeVolatility(percentile_of_history=0.95, persistence=0.98)
    body = alerts.market_volatility_alerts("CDFI_X", sticky)[0].body
    assert "still be here next month" in body


def test_market_vol_handles_a_missing_model():
    assert alerts.market_volatility_alerts("CDFI_X", None) == []


# --- REGIME_SHIFT -------------------------------------------------------------


def test_regime_shift_fires_on_a_recent_break():
    drafts = alerts.regime_shift_alerts("CDFI_X", FakeRegime())
    assert len(drafts) == 1
    assert "downgrades signal strength" in drafts[0].action


def test_regime_shift_is_silent_on_an_old_break():
    old = FakeRegime(changed_recently=False, days_since_change=200)
    assert alerts.regime_shift_alerts("CDFI_X", old) == []


# --- PORT_CONGESTION ----------------------------------------------------------


def _queue(counts: list[int]) -> list[tuple[date, int]]:
    start = date(2026, 6, 1)
    return [(start + timedelta(days=i), count) for i, count in enumerate(counts)]


def test_congestion_fires_when_the_queue_spikes_above_its_own_history():
    history = _queue([10] * 40 + [34])
    drafts = alerts.congestion_alerts("INPRT", history)
    assert any(d.code == "PORT_CONGESTION" for d in drafts)
    assert drafts[0].evidence["z_score"] > alerts.CONGESTION_Z_WARNING


def test_congestion_is_silent_on_a_normal_queue():
    history = _queue([10, 12, 9, 11, 10, 13, 8] * 6)
    assert alerts.congestion_alerts("INPRT", history) == []


def test_congestion_is_silent_on_too_little_history():
    """A z-score against five days of data is noise, not a signal."""
    assert alerts.congestion_alerts("INPRT", _queue([10, 40, 10, 40, 10])) == []


def test_costly_detention_fires_independently_of_the_queue_z_score():
    """A queue can be normal for a port and still be expensive."""
    history = _queue([20] * 40 + [20])  # perfectly normal
    drafts = alerts.congestion_alerts("INPRT", history, FakeWaiting())
    detention = [d for d in drafts if d.scope.endswith(":detention")]
    assert len(detention) == 1
    assert "/t of demurrage" in detention[0].action


def test_detention_alert_is_silent_when_the_wait_is_short():
    history = _queue([20] * 41)
    quiet = FakeWaiting(expected_days=3.0, p80_days=5.0)
    drafts = alerts.congestion_alerts("INPRT", history, quiet)
    assert not [d for d in drafts if d.scope.endswith(":detention")]


# --- DEMAND_URGENCY -----------------------------------------------------------


def test_demand_urgency_fires_when_stocks_run_down():
    drafts = alerts.demand_urgency_alerts(4.2, 12, obs_date=date(2026, 9, 1))
    assert len(drafts) == 1
    assert drafts[0].severity == "warning"
    assert "cuts the value of waiting" in drafts[0].action


def test_demand_urgency_escalates_below_three_days():
    assert alerts.demand_urgency_alerts(2.0, 30)[0].severity == "critical"


def test_demand_urgency_is_silent_at_comfortable_stock():
    assert alerts.demand_urgency_alerts(14.0, 0) == []


def test_demand_urgency_handles_a_missing_reading():
    assert alerts.demand_urgency_alerts(None, 0) == []


# --- CYCLONE ------------------------------------------------------------------


PARADIP = {"code": "INPRT", "name": "Paradip", "latitude": 20.26, "longitude": 86.67}


def test_cyclone_fires_for_a_storm_near_a_port():
    points = [
        {
            "date": date(2026, 11, 2),
            "latitude": 20.5,
            "longitude": 87.0,
            "max_wind_kt": 70.0,
            "storm_id": "X1",
            "storm_name": "TEST",
        }
    ]
    drafts = alerts.cyclone_alerts(points, [PARADIP], as_of=date(2026, 11, 3))
    assert len(drafts) == 1
    assert drafts[0].severity == "critical"  # 70 kt
    assert drafts[0].evidence["distance_km"] < alerts.CYCLONE_RADIUS_KM


def test_cyclone_is_silent_for_a_distant_storm():
    points = [
        {
            "date": date(2026, 11, 2),
            "latitude": 5.0,
            "longitude": 60.0,
            "max_wind_kt": 90.0,
            "storm_id": "X2",
            "storm_name": "FAR",
        }
    ]
    assert alerts.cyclone_alerts(points, [PARADIP], as_of=date(2026, 11, 3)) == []


def test_cyclone_ignores_a_historical_storm_at_the_same_position():
    """The archive is decades deep; only live systems may raise an alert."""
    points = [
        {
            "date": date(2019, 5, 3),
            "latitude": 20.3,
            "longitude": 86.7,
            "max_wind_kt": 120.0,
            "storm_id": "FANI",
            "storm_name": "FANI",
        }
    ]
    assert alerts.cyclone_alerts(points, [PARADIP], as_of=date(2026, 11, 3)) == []


# --- PRICE_SPREAD -------------------------------------------------------------


def test_price_spread_fires_outside_the_normal_band():
    history = [20.0] * 60
    drafts = alerts.price_spread_alerts(180.0, 100.0, history + [80.0])
    assert len(drafts) == 1
    assert abs(drafts[0].evidence["z_score"]) >= 2.0


def test_price_spread_is_silent_inside_the_band():
    import random

    random.seed(1)
    history = [20.0 + random.gauss(0, 3) for _ in range(60)]
    assert alerts.price_spread_alerts(121.0, 100.0, history) == []


# --- DATA_HEALTH --------------------------------------------------------------


def test_data_health_fires_on_a_failed_feed():
    freshness = [
        {
            "adapter": "sse_cdfi",
            "status": "failed",
            "used_cache": False,
            "message": "timeout",
            "finished_at": date(2026, 9, 3),
        }
    ]
    drafts = alerts.data_health_alerts(freshness, as_of=date(2026, 9, 3))
    assert len(drafts) == 1
    assert drafts[0].severity == "warning"


def test_data_health_fires_when_a_feed_is_silently_serving_cache():
    """The failure mode that is invisible on every other screen."""
    freshness = [
        {
            "adapter": "paradip_daily_traffic",
            "status": "stale",
            "used_cache": True,
            "message": None,
            "finished_at": date(2026, 9, 3),
        }
    ]
    drafts = alerts.data_health_alerts(freshness, as_of=date(2026, 9, 3))
    assert len(drafts) == 1
    assert "cached payload" in drafts[0].body
    assert "do not read a flat line" in drafts[0].action


def test_data_health_fires_on_an_overdue_feed():
    freshness = [
        {
            "adapter": "cea_daily_coal_stock",
            "status": "ok",
            "used_cache": False,
            "message": None,
            "finished_at": date(2026, 8, 20),
        }
    ]
    drafts = alerts.data_health_alerts(freshness, as_of=date(2026, 9, 3))
    assert len(drafts) == 1
    assert "14 days ago" in drafts[0].body


def test_data_health_is_silent_when_every_feed_is_current():
    freshness = [
        {
            "adapter": "worldbank_pinksheet",
            "status": "ok",
            "used_cache": False,
            "message": None,
            "finished_at": date(2026, 9, 3),
        }
    ]
    assert alerts.data_health_alerts(freshness, as_of=date(2026, 9, 3)) == []


# --- every rule produces an action --------------------------------------------


def test_every_draft_carries_an_action_and_evidence():
    """An alert without an action is a fact, not a decision."""
    drafts = (
        alerts.market_volatility_alerts("CDFI_X", FakeVolatility())
        + alerts.regime_shift_alerts("CDFI_X", FakeRegime())
        + alerts.congestion_alerts("INPRT", _queue([10] * 40 + [34]), FakeWaiting())
        + alerts.demand_urgency_alerts(3.0, 20)
        + alerts.data_health_alerts(
            [
                {
                    "adapter": "x",
                    "status": "failed",
                    "used_cache": False,
                    "message": None,
                    "finished_at": date(2026, 9, 3),
                }
            ],
            as_of=date(2026, 9, 3),
        )
    )
    assert len(drafts) >= 5
    for draft in drafts:
        assert draft.action.strip(), f"{draft.code} has no action"
        assert draft.evidence, f"{draft.code} has no evidence"
        assert draft.severity in alerts.SEVERITY_ORDER
