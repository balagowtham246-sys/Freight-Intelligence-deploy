"""Tests for the loadable-tonnes engine.

The expected values are worked by hand from the documented relations rather
than captured from a previous run, so a regression in the physics fails here
rather than being silently blessed.
"""
from __future__ import annotations

import math

import pytest

from app.domain.loadable_tonnes import (
    SALT_DENSITY,
    BerthLimits,
    Cargo,
    LimitingFactor,
    RejectionReason,
    Vessel,
    calculate_loadable_tonnes,
    displacement_at_draft,
    dock_water_allowance_m,
    fresh_water_allowance_mm,
)

KAMSARMAX = Vessel(
    name="Kamsarmax",
    dwt=82000,
    loa_m=229.0,
    beam_m=32.26,
    summer_draft_m=14.4,
    lightship_t=13100,
    tpc=65.9,
    grain_cubic_m3=97000,
    geared=False,
    constants_t=500,
)

SUPRAMAX = Vessel(
    name="Supramax",
    dwt=58000,
    loa_m=190.0,
    beam_m=32.26,
    summer_draft_m=12.8,
    lightship_t=10900,
    tpc=54.0,
    grain_cubic_m3=72000,
    geared=True,
    constants_t=450,
)

CAPESIZE = Vessel(
    name="Capesize",
    dwt=180000,
    loa_m=292.0,
    beam_m=45.0,
    summer_draft_m=18.2,
    lightship_t=24000,
    tpc=118.5,
    grain_cubic_m3=198000,
    geared=False,
    constants_t=800,
)

HALDIA = BerthLimits(
    port_code="INHAL",
    berth_name="HDC-2",
    max_draft_m=8.5,
    water_density=1.008,
    max_loa_m=230.0,
    max_beam_m=32.26,
    max_dwt=45000,
    requires_gear=True,
    tide_constrained=True,
)

GANGAVARAM = BerthLimits(
    port_code="INGGV",
    berth_name="MechanisedImport-1",
    max_draft_m=19.5,
    water_density=SALT_DENSITY,
    max_loa_m=300.0,
    max_beam_m=50.0,
    max_dwt=200000,
    requires_gear=False,
)

PARADIP_CB1 = BerthLimits(
    port_code="INPRT",
    berth_name="CB-1",
    max_draft_m=14.5,
    water_density=SALT_DENSITY,
    max_loa_m=250.0,
    max_beam_m=45.0,
    max_dwt=100000,
    requires_gear=False,
)

LOW_CV_COAL = Cargo("thermal_coal_low_cv", stowage_factor_m3_per_t=1.35, gcv_kcal_per_kg=4200)
HIGH_CV_COAL = Cargo("thermal_coal_high_cv", stowage_factor_m3_per_t=1.22, gcv_kcal_per_kg=6000)


# --------------------------------------------------------------- primitives


def test_displacement_at_summer_draft_equals_summer_displacement():
    """At summer draft in salt water, displacement is lightship + DWT."""
    got = displacement_at_draft(KAMSARMAX, KAMSARMAX.summer_draft_m, SALT_DENSITY)
    assert got == pytest.approx(13100 + 82000, rel=1e-9)


def test_displacement_falls_by_tpc_per_centimetre():
    """One metre less draft removes 100 * TPC tonnes of displacement."""
    full = displacement_at_draft(SUPRAMAX, 12.8, SALT_DENSITY)
    one_metre_less = displacement_at_draft(SUPRAMAX, 11.8, SALT_DENSITY)
    assert full - one_metre_less == pytest.approx(100 * 54.0, rel=1e-9)


def test_fresh_water_allowance_matches_standard_formula():
    """FWA (mm) = displacement / (4 * TPC)."""
    expected = (10900 + 58000) / (4 * 54.0)
    assert fresh_water_allowance_mm(SUPRAMAX) == pytest.approx(expected, rel=1e-9)


def test_dock_water_allowance_is_zero_in_full_salt_water():
    assert dock_water_allowance_m(SUPRAMAX, SALT_DENSITY) == pytest.approx(0.0, abs=1e-12)


def test_dock_water_allowance_positive_in_brackish_water():
    """Lighter water means the vessel floats deeper, so DWA must be positive."""
    assert dock_water_allowance_m(SUPRAMAX, 1.008) > 0


def test_brackish_water_reduces_displacement_proportionally():
    """Displacement scales linearly with density at fixed draft."""
    salt = displacement_at_draft(SUPRAMAX, 8.5, SALT_DENSITY)
    brackish = displacement_at_draft(SUPRAMAX, 8.5, 1.008)
    assert brackish / salt == pytest.approx(1.008 / SALT_DENSITY, rel=1e-9)


# ------------------------------------------------------- hard feasibility


def test_capesize_rejected_at_haldia():
    """The headline feasibility case: a Capesize cannot go to Haldia."""
    result = calculate_loadable_tonnes(CAPESIZE, HALDIA, cargo=HIGH_CV_COAL)
    assert result.feasible is False
    assert RejectionReason.LOA in result.rejections
    assert RejectionReason.BEAM in result.rejections
    assert RejectionReason.MAX_DWT in result.rejections
    assert RejectionReason.GEAR in result.rejections
    assert "rejected" in result.explanation.lower()


def test_capesize_accepted_at_gangavaram():
    """The same vessel is comfortably feasible at a deep berth."""
    result = calculate_loadable_tonnes(CAPESIZE, GANGAVARAM, cargo=HIGH_CV_COAL)
    assert result.feasible is True
    assert result.loadable_tonnes > 150_000


def test_kamsarmax_rejected_at_haldia_on_dwt_and_gear():
    result = calculate_loadable_tonnes(KAMSARMAX, HALDIA, cargo=HIGH_CV_COAL)
    assert result.feasible is False
    assert RejectionReason.MAX_DWT in result.rejections
    assert RejectionReason.GEAR in result.rejections


def test_supramax_feasible_at_paradip_coal_berth():
    result = calculate_loadable_tonnes(SUPRAMAX, PARADIP_CB1, cargo=HIGH_CV_COAL)
    assert result.feasible is True
    assert result.loadable_tonnes > 50_000


# ------------------------------------------------------- limiting factors


def test_deadweight_binds_when_berth_is_deep():
    """At a deep berth the vessel marks bind, not the berth."""
    result = calculate_loadable_tonnes(
        SUPRAMAX, GANGAVARAM, cargo=HIGH_CV_COAL, bunkers_rob_t=1000
    )
    assert result.limiting_factor is LimitingFactor.DEADWEIGHT
    # cargo = DWT - bunkers - constants
    assert result.loadable_tonnes == pytest.approx(58000 - 1000 - 450, rel=1e-9)


def test_draft_binds_at_a_shallow_berth():
    """A Supramax at a 10 m berth is draft limited, and the maths is checkable."""
    shallow = BerthLimits(
        port_code="TEST",
        berth_name="Shallow",
        max_draft_m=10.0,
        water_density=SALT_DENSITY,
        max_dwt=100000,
    )
    result = calculate_loadable_tonnes(
        SUPRAMAX, shallow, cargo=HIGH_CV_COAL, bunkers_rob_t=1000
    )
    assert result.limiting_factor is LimitingFactor.DRAFT

    # Hand-worked: 2.8 m less draft = 2.8 * 100 * 54.0 = 15,120 t of displacement
    expected_displacement = (10900 + 58000) - 2.8 * 100 * 54.0
    expected_cargo = expected_displacement - 10900 - 1000 - 450
    assert result.loadable_tonnes == pytest.approx(expected_cargo, rel=1e-9)


def test_volume_binds_for_low_calorific_value_coal():
    """Low CV Indonesian coal cubes out before the marks on a deep berth."""
    result = calculate_loadable_tonnes(
        SUPRAMAX, GANGAVARAM, cargo=LOW_CV_COAL, bunkers_rob_t=1000
    )
    assert result.limiting_factor is LimitingFactor.VOLUME
    assert result.loadable_tonnes == pytest.approx(72000 / 1.35, rel=1e-9)
    # And the denser grade must lift more on the identical vessel and berth.
    denser = calculate_loadable_tonnes(
        SUPRAMAX, GANGAVARAM, cargo=HIGH_CV_COAL, bunkers_rob_t=1000
    )
    assert denser.loadable_tonnes > result.loadable_tonnes


# ------------------------------------------------------------------- tide


def test_tide_credit_only_applies_at_tide_constrained_ports():
    """A tide credit at a non-tidal berth must not inflate the answer."""
    with_tide = calculate_loadable_tonnes(
        SUPRAMAX, PARADIP_CB1, cargo=HIGH_CV_COAL, tide_height_m=2.0
    )
    without = calculate_loadable_tonnes(SUPRAMAX, PARADIP_CB1, cargo=HIGH_CV_COAL)
    assert with_tide.loadable_tonnes == without.loadable_tonnes
    assert with_tide.tide_credit_m == 0.0


def test_tide_increases_loadable_tonnes_at_a_tidal_port():
    """At Haldia the tide is the difference between sailing and not sailing."""
    handy = Vessel(
        name="Handysize",
        dwt=32000,
        loa_m=180.0,
        beam_m=28.0,
        summer_draft_m=10.2,
        lightship_t=7800,
        tpc=38.4,
        grain_cubic_m3=42500,
        geared=True,
        constants_t=350,
    )
    neap = calculate_loadable_tonnes(handy, HALDIA, cargo=HIGH_CV_COAL, tide_height_m=0.0)
    # 1.0 m of tide takes the permissible draft from 8.5 m to 9.5 m, which is
    # still inside this vessel summer draft of 10.2 m, so the whole tidal gain
    # converts to cargo.
    flood = calculate_loadable_tonnes(handy, HALDIA, cargo=HIGH_CV_COAL, tide_height_m=1.0)

    assert flood.loadable_tonnes > neap.loadable_tonnes
    gain = 1.0 * 100 * 38.4 * (1.008 / SALT_DENSITY)
    assert flood.loadable_tonnes - neap.loadable_tonnes == pytest.approx(gain, rel=1e-6)


def test_tide_credit_is_capped_by_the_summer_marks():
    """A big tide cannot load a vessel past her own marks.

    Haldia permits 8.5 m and this Handysize is marked at 10.2 m, so a 2.5 m
    tide yields only 1.7 m of usable gain. Getting this wrong would overstate
    cargo on precisely the tidal lane where parcels are tightest.
    """
    handy = Vessel(
        name="Handysize",
        dwt=32000,
        loa_m=180.0,
        beam_m=28.0,
        summer_draft_m=10.2,
        lightship_t=7800,
        tpc=38.4,
        grain_cubic_m3=42500,
        geared=True,
        constants_t=350,
    )
    neap = calculate_loadable_tonnes(handy, HALDIA, cargo=HIGH_CV_COAL, tide_height_m=0.0)
    spring = calculate_loadable_tonnes(handy, HALDIA, cargo=HIGH_CV_COAL, tide_height_m=2.5)

    usable_gain_m = 10.2 - 8.5
    gain = usable_gain_m * 100 * 38.4 * (1.008 / SALT_DENSITY)
    assert spring.loadable_tonnes - neap.loadable_tonnes == pytest.approx(gain, rel=1e-6)
    assert spring.workings["governing_draft_m"] == pytest.approx(10.2, rel=1e-9)

    # Beyond the marks, more tide adds nothing at all.
    bigger = calculate_loadable_tonnes(handy, HALDIA, cargo=HIGH_CV_COAL, tide_height_m=4.0)
    assert bigger.loadable_tonnes == pytest.approx(spring.loadable_tonnes, rel=1e-12)


def test_ukc_holdback_reduces_cargo_where_draft_binds():
    """Extra under-keel clearance must cost cargo wherever draft is binding."""
    shallow = BerthLimits(
        port_code="TEST",
        berth_name="Shallow",
        max_draft_m=10.0,
        water_density=SALT_DENSITY,
        max_dwt=100000,
    )
    base = calculate_loadable_tonnes(SUPRAMAX, shallow, cargo=HIGH_CV_COAL)
    cautious = calculate_loadable_tonnes(
        SUPRAMAX, shallow, cargo=HIGH_CV_COAL, ukc_policy_m=0.5
    )
    assert cautious.loadable_tonnes < base.loadable_tonnes
    # 0.5 m of holdback is worth exactly 0.5 * 100 * TPC tonnes.
    assert base.loadable_tonnes - cautious.loadable_tonnes == pytest.approx(
        0.5 * 100 * 54.0, rel=1e-9
    )


def test_ukc_holdback_is_inert_when_vessel_is_on_her_marks():
    """At a deep berth the marks bind, so a holdback changes nothing."""
    base = calculate_loadable_tonnes(SUPRAMAX, PARADIP_CB1, cargo=HIGH_CV_COAL)
    cautious = calculate_loadable_tonnes(
        SUPRAMAX, PARADIP_CB1, cargo=HIGH_CV_COAL, ukc_policy_m=0.5
    )
    assert cautious.loadable_tonnes == pytest.approx(base.loadable_tonnes, rel=1e-12)
    assert base.limiting_factor is LimitingFactor.DEADWEIGHT


# ------------------------------------------------------------- degenerate


def test_no_cargo_possible_is_reported_not_negative():
    """A berth too shallow to float the ship returns zero, never a negative."""
    puddle = BerthLimits(
        port_code="TEST",
        berth_name="Puddle",
        max_draft_m=3.0,
        water_density=SALT_DENSITY,
        max_dwt=250000,
    )
    result = calculate_loadable_tonnes(CAPESIZE, puddle, cargo=HIGH_CV_COAL)
    assert result.feasible is False
    assert result.loadable_tonnes == 0.0
    assert RejectionReason.NO_CARGO_POSSIBLE in result.rejections


def test_bunkers_directly_reduce_cargo_one_for_one():
    light = calculate_loadable_tonnes(SUPRAMAX, GANGAVARAM, bunkers_rob_t=500)
    heavy = calculate_loadable_tonnes(SUPRAMAX, GANGAVARAM, bunkers_rob_t=1500)
    assert light.loadable_tonnes - heavy.loadable_tonnes == pytest.approx(1000.0, rel=1e-9)


def test_result_serialises_with_workings():
    result = calculate_loadable_tonnes(SUPRAMAX, PARADIP_CB1, cargo=HIGH_CV_COAL)
    payload = result.as_dict()
    assert payload["feasible"] is True
    assert payload["workings"]["water_density"] == SALT_DENSITY
    assert math.isfinite(payload["loadable_tonnes"])
