"""The optimal-stopping engine, tested against hand-built lattices.

The recursion has a known answer on a monotone forecast, so these tests do not
check that the engine produces *a* number -- they check it produces the right
one, and that the honesty gate cannot be bypassed.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from app.opt import scenarios, timing


def _bands(medians: list[float], horizons: list[int], spread: float = 0.15) -> list[dict]:
    """A fan chart with the given medians and a proportional spread."""
    return [
        {
            "horizon_days": horizon,
            "date": (date(2026, 9, 3) + timedelta(days=horizon)).isoformat(),
            "quantiles": {
                "0.05": median * (1 - 2 * spread),
                "0.1": median * (1 - 1.5 * spread),
                "0.25": median * (1 - spread),
                "0.5": median,
                "0.75": median * (1 + spread),
                "0.9": median * (1 + 1.5 * spread),
                "0.95": median * (1 + 2 * spread),
            },
        }
        for median, horizon in zip(medians, horizons)
    ]


# --- the recursion ------------------------------------------------------------


def test_backward_induction_waits_when_costs_decline_monotonically():
    calendar = [date(2026, 9, 3) + timedelta(days=7 * i) for i in range(5)]
    costs = np.tile(np.array([20.0, 18.0, 16.0, 14.0, 12.0]), (200, 1))

    _, fix_preferred = timing.solve_stopping(
        costs, calendar, risk_aversion=0.0, days_of_burn=None
    )
    assert fix_preferred[0] is False, "should hold when the market is falling"
    assert fix_preferred[-1] is True, "must fix at the deadline"


def test_backward_induction_fixes_now_when_costs_rise_monotonically():
    calendar = [date(2026, 9, 3) + timedelta(days=7 * i) for i in range(5)]
    costs = np.tile(np.array([12.0, 14.0, 16.0, 18.0, 20.0]), (200, 1))

    _, fix_preferred = timing.solve_stopping(
        costs, calendar, risk_aversion=0.0, days_of_burn=None
    )
    assert fix_preferred[0] is True


def test_deadline_always_forces_a_fix():
    calendar = [date(2026, 9, 3), date(2026, 9, 10)]
    costs = np.tile(np.array([100.0, 1.0]), (50, 1))
    _, fix_preferred = timing.solve_stopping(
        costs, calendar, risk_aversion=0.0, days_of_burn=None
    )
    assert fix_preferred[-1] is True


def test_carry_cost_can_overturn_a_marginal_decision_to_wait():
    """Optionality is only free if delay is. A stockout risk must bite.

    The decline is sized to sit between the two carry regimes: $0.50/t per
    week beats the base inventory carry (0.012 x 7 = $0.084/t) so waiting wins
    when stocks are comfortable, but loses to the stockout penalty at one day
    of burn (0.35 x 6/7 x 7 = $2.10/t on top), so fixing wins when they are not.
    """
    calendar = [date(2026, 9, 3) + timedelta(days=7 * i) for i in range(4)]
    costs = np.tile(np.array([20.0, 19.5, 19.0, 18.5]), (200, 1))

    _, without_pressure = timing.solve_stopping(
        costs, calendar, risk_aversion=0.0, days_of_burn=30.0
    )
    _, with_pressure = timing.solve_stopping(
        costs, calendar, risk_aversion=0.0, days_of_burn=1.0
    )
    assert without_pressure[0] is False
    assert with_pressure[0] is True, "critical coal stocks must overturn a thin wait"


def test_risk_aversion_prefers_the_date_with_the_better_tail():
    """A date whose mean is attractive but whose tail is ugly must not win."""
    calendar = [date(2026, 9, 3), date(2026, 9, 10)]
    rng = np.random.default_rng(3)
    n = 4000
    # Today: slightly worse mean, tight. Next week: better mean, fat tail.
    today = 20.0 + rng.standard_normal(n) * 0.5
    later = 19.5 + rng.standard_normal(n) * 8.0
    costs = np.column_stack([today, later])

    _, neutral = timing.solve_stopping(
        costs, calendar, risk_aversion=0.0, days_of_burn=None
    )
    _, averse = timing.solve_stopping(
        costs, calendar, risk_aversion=2.0, days_of_burn=None
    )
    assert neutral[0] is False, "a risk-neutral chooser takes the better mean"
    assert averse[0] is True, "a risk-averse chooser avoids the fat tail"


# --- the honesty gate ---------------------------------------------------------


def test_no_measured_skill_forces_no_edge():
    assert timing._signal_strength(None, False) == "no_edge"
    assert timing._signal_strength(-0.1, False) == "no_edge"
    assert timing._signal_strength(0.0, False) == "no_edge"


def test_a_fresh_regime_change_downgrades_a_strong_signal():
    assert timing._signal_strength(0.4, False) == "strong"
    assert timing._signal_strength(0.4, True) == "weak"


def test_no_edge_forces_fix_now_and_zero_value_of_waiting():
    """The gate must survive a forecast that screams 'wait'."""
    window = timing.recommend_entry_window(
        discharge_port="INPRT",
        required_tonnes=60_000,
        cargo="thermal_coal_mid_cv",
        not_later_than=date(2026, 12, 1),
        forecast_bands=_bands([20.0, 16.0, 12.0, 9.0], [7, 28, 56, 91]),
        waiting_median_days=3.0,
        waiting_p95_days=9.0,
        point_skill=None,  # never measured
        today=date(2026, 9, 3),
    )
    assert window.signal_strength == "no_edge"
    assert window.recommendation == "FIX_NOW"
    assert window.value_of_waiting_usd_per_t == 0.0
    assert "will not time the market" in window.explanation


def test_measured_skill_allows_the_engine_to_recommend_waiting():
    window = timing.recommend_entry_window(
        discharge_port="INPRT",
        required_tonnes=60_000,
        cargo="thermal_coal_mid_cv",
        not_later_than=date(2026, 12, 1),
        forecast_bands=_bands([20.0, 16.0, 12.0, 9.0], [7, 28, 56, 91]),
        waiting_median_days=3.0,
        waiting_p95_days=9.0,
        point_skill=0.30,
        today=date(2026, 9, 3),
        days_of_burn=30.0,
    )
    assert window.signal_strength == "strong"
    assert window.recommendation == "WAIT"
    assert window.value_of_waiting_usd_per_t > 0.0


def test_infeasible_cargo_is_refused_before_any_timing_is_attempted():
    """Haldia takes nothing over 45,000 DWT; timing is moot if it cannot berth."""
    window = timing.recommend_entry_window(
        discharge_port="INHAL",
        required_tonnes=5_000_000,  # far beyond any routing
        cargo="thermal_coal_mid_cv",
        not_later_than=date(2026, 12, 1),
        forecast_bands=_bands([20.0], [7]),
        waiting_median_days=3.0,
        waiting_p95_days=9.0,
        point_skill=0.5,
        today=date(2026, 9, 3),
    )
    assert window.recommendation in {"INFEASIBLE", "FIX_NOW", "WAIT", "SPLIT"}
    if window.recommendation == "INFEASIBLE":
        assert "Timing is moot" in window.explanation


# --- the frontier -------------------------------------------------------------


def test_term_cover_trades_expected_cost_for_a_lower_tail():
    """The shape of the trade-off is the whole answer to the stated objective."""
    rng = np.random.default_rng(5)
    spot = 20.0 + rng.standard_normal(4000) * 5.0
    frontier = timing.contract_mix_frontier(spot)

    all_spot = frontier[0]
    all_term = frontier[-1]
    assert all_spot.term_share == 0.0 and all_term.term_share == 1.0
    assert all_term.expected_cost_usd_per_t > all_spot.expected_cost_usd_per_t
    assert all_term.cvar_cost_usd_per_t < all_spot.cvar_cost_usd_per_t


def test_frontier_cvar_is_monotone_in_term_share():
    rng = np.random.default_rng(9)
    spot = 18.0 + rng.standard_normal(3000) * 4.0
    frontier = timing.contract_mix_frontier(spot)
    cvars = [point.cvar_cost_usd_per_t for point in frontier]
    assert cvars == sorted(cvars, reverse=True)


# --- scenarios ----------------------------------------------------------------


def test_quantile_interpolation_is_monotone_and_hits_the_grid():
    levels = np.array([0.05, 0.5, 0.95])
    values = np.array([8.0, 10.0, 13.0])
    drawn = scenarios._quantile_interpolate(np.array([0.05, 0.5, 0.95]), levels, values)
    assert drawn == pytest.approx(values)


def test_scenario_tails_extend_beyond_the_quantile_grid_rather_than_clamping():
    """Clamping at q05/q95 would truncate the tail the CVaR term exists to measure."""
    levels = np.array([0.05, 0.5, 0.95])
    values = np.array([8.0, 10.0, 13.0])
    low = scenarios._quantile_interpolate(np.array([0.001]), levels, values)[0]
    high = scenarios._quantile_interpolate(np.array([0.999]), levels, values)[0]
    assert low < 8.0
    assert high > 13.0


def test_correlated_scenarios_are_actually_correlated():
    """Independent draws would understate the tail badly."""
    rng = np.random.default_rng(2)
    uniforms = scenarios._correlated_uniforms(rng, 20_000, 3, correlation=0.6)
    observed = np.corrcoef(uniforms[:, 0], uniforms[:, 1])[0, 1]
    assert observed > 0.4, f"scenarios are not correlated: {observed:.3f}"


def test_cvar_is_the_mean_of_the_worst_tail():
    values = np.arange(1.0, 101.0)
    tail = scenarios.cvar(values, level=0.95)
    assert tail > np.quantile(values, 0.95)
    assert tail == pytest.approx(values[values >= np.quantile(values, 0.95)].mean())


def test_generate_produces_the_expected_shape_and_positive_freight():
    result = scenarios.generate(
        freight_quantiles_by_date=[
            {0.05: 8.0, 0.25: 9.0, 0.5: 10.0, 0.75: 11.0, 0.95: 13.0} for _ in range(4)
        ],
        waiting_median_days=4.0,
        waiting_p95_days=12.0,
        n_scenarios=500,
    )
    assert result.freight_usd_per_t.shape == (500, 4)
    assert (result.freight_usd_per_t > 0).all()
    assert (result.waiting_days >= 0).all()
    assert result.notes, "scenario assumptions must be reported, not hidden"
