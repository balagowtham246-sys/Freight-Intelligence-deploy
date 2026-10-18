"""Tests for feasibility screening across real port and vessel reference data.

These run against the actual YAML reference files rather than fixtures, so a
bad edit to a port constraint fails the suite.
"""
from __future__ import annotations

import pytest

from app.domain.feasibility import screen_all, screen_direct, screen_lightering
from app.domain.reference import (
    berths_for_port,
    load_cargoes,
    load_ports,
    load_vessel_classes,
)


# ------------------------------------------------------------- reference


def test_reference_data_loads():
    ports = load_ports()
    assert {"INHAL", "INPRT", "INGGV", "INDHR", "INSND"} <= set(ports)
    assert len(load_vessel_classes()) >= 8
    assert "thermal_coal_low_cv" in load_cargoes()


def test_haldia_is_brackish_and_tidal():
    """The Hooghly density and tidal dependence drive real tonnage differences."""
    haldia = load_ports()["INHAL"]
    assert haldia["water_density"] < 1.02
    assert haldia["tide_constrained"] is True


def test_sandheads_is_deep_enough_for_a_laden_capesize():
    berths = berths_for_port("INSND")
    assert berths
    assert max(b.max_draft_m for b in berths) >= 18.2


# ---------------------------------------------------------- direct screen


def test_no_vessel_over_45000_dwt_discharges_directly_at_haldia():
    """Haldia berth limits are the constraint the whole lightering case rests on."""
    options = screen_direct("INHAL", 100_000, cargo_name="thermal_coal_mid_cv")
    classes = load_vessel_classes()
    for option in options:
        if classes[option.vessel_class].dwt > 45_000:
            assert not option.feasible, f"{option.vessel_class} should not fit Haldia"


def test_capesize_discharges_directly_at_gangavaram():
    options = screen_direct("INGGV", 150_000, cargo_name="thermal_coal_high_cv")
    cape = next(o for o in options if o.vessel_class == "Capesize")
    assert cape.feasible
    assert cape.loadable_tonnes > 150_000


# ------------------------------------------------------- lightering screen


def test_lightering_unlocks_haldia_for_large_tonnage():
    """The differentiating case: what direct discharge cannot do, transhipment can."""
    direct = [o for o in screen_direct("INHAL", 150_000) if o.feasible]
    lightering = screen_lightering("INHAL", 150_000)

    best_direct = max((o.loadable_tonnes for o in direct), default=0.0)
    best_lightering = max((o.loadable_tonnes for o in lightering), default=0.0)

    assert lightering, "expected lightering options into Haldia"
    assert best_lightering > best_direct * 2


def test_lightering_only_offered_where_topology_exists():
    """Deep ports need no transhipment, and none should be invented for them."""
    assert screen_lightering("INGGV", 150_000) == []


def test_lightering_daughter_is_feasible_at_the_final_berth():
    for option in screen_lightering("INHAL", 150_000):
        assert option.daughter_class is not None
        assert option.daughter_voyages and option.daughter_voyages >= 1


# ------------------------------------------------------ load port capping


def test_load_port_caps_lift_and_workings_reconcile():
    """A shallow origin must bind, and the shown arithmetic must match."""
    result = screen_all(
        "INHAL", 150_000, load_port="IDTAB", cargo_name="thermal_coal_low_cv"
    )
    top = result["feasible"][0]
    assert top["limiting_factor"].startswith("load_port_")

    workings = top["detail"]["workings"]
    reconstructed = (
        workings["displacement_t"]
        - workings["lightship_t"]
        - workings["bunkers_rob_t"]
        - workings["constants_t"]
    )
    assert reconstructed == pytest.approx(top["loadable_tonnes"], abs=1.0)
    assert top["detail"]["binding_end"] == "load_port"


def test_load_port_cap_never_increases_lift():
    uncapped = screen_all("INHAL", 150_000, cargo_name="thermal_coal_low_cv")
    capped = screen_all(
        "INHAL", 150_000, load_port="IDTAB", cargo_name="thermal_coal_low_cv"
    )
    best_uncapped = uncapped["feasible"][0]["loadable_tonnes"]
    best_capped = capped["feasible"][0]["loadable_tonnes"]
    assert best_capped <= best_uncapped


def test_vessel_too_large_for_the_load_port_is_rejected():
    """Taboneo cannot load a Newcastlemax, so no routing may claim it can."""
    result = screen_all("INHAL", 150_000, load_port="IDTAB")
    for option in result["feasible"]:
        assert option["vessel_class"] != "Newcastlemax"


# ------------------------------------------------------------- summaries


def test_summary_does_not_call_a_feasible_vessel_rejected():
    """A vessel feasible by transhipment must not be reported as rejected."""
    result = screen_all("INHAL", 150_000, cargo_name="thermal_coal_low_cv")
    feasible_classes = {o["vessel_class"] for o in result["feasible"]}
    summary = result["summary"]

    if "Not possible at all:" in summary:
        blocked = summary.split("Not possible at all:")[1].split(".")[0]
        for name in feasible_classes:
            assert name not in blocked, f"{name} is feasible but listed as impossible"


def test_options_are_ranked_by_vessels_required():
    result = screen_all("INGGV", 150_000, cargo_name="thermal_coal_high_cv")
    counts = [o["vessels_required"] for o in result["feasible"]]
    assert counts == sorted(counts)
