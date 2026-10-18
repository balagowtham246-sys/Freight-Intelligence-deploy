"""Optimal Market Entry Timing.

This answers deliverable (a) of the problem statement: *identify ideal windows
to secure short-term or mid-term vessel charter contracts for specific cargo
requirements, minimising freight costs*.

Why this is not a forecast
--------------------------
"When should we fix?" is not answered by a fan chart. It is an **optimal
stopping problem**: on each decision date the charterer either fixes at
today's achievable rate, or waits and faces a distribution tomorrow, with the
option to wait again — until a deadline at which they must fix regardless.

The value of waiting is therefore *not* the expected rate improvement. It is
the value of an option, and it is worth more when volatility is high, worth
less when the deadline is near, and worth less again when our own coal stocks
are running down and we cannot afford to be wrong.

The recursion
-------------
Solved by backward induction over the decision calendar::

    V(T) = fix_cost(T)                                  # deadline: must fix
    V(d) = min( fix_cost(d),  E[V(d+1)] + carry(d) )    # earlier: fix or wait

evaluated per scenario, with the objective

    E[cost] + lambda * CVaR_95[cost]

``carry(d)`` is the cost of not having fixed: inventory carry plus a stockout
penalty that scales with how close the CEA days-of-burn signal is to
critical. Without a carry term the model would always wait, because optionality
is free if delay costs nothing — which is exactly the mistake a naive
"wait for a better rate" heuristic makes.

Feasibility gates everything
----------------------------
Before any of this runs, the existing constraint engine screens which vessels
can physically berth. Recommending a timing window for a vessel that cannot
enter the port would be worse than useless, and the constraint engine is the
part of this platform that is deterministic and provably right.

The "no edge" rule
------------------
``signal_strength`` is forced to ``no_edge`` whenever the backtest shows the
forecaster does not beat a random walk at the relevant horizon, or has never
been measured. In that state the engine still returns the full cost and
feasibility arithmetic — which is real and useful — but explicitly declines to
time the market, and the recommendation falls back to the deadline-driven
answer. This is the platform's standing commitment made operational.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from ..domain.feasibility import screen_all
from .scenarios import ScenarioSet, cvar, generate

# Decision calendar granularity. Weekly is the cadence a chartering desk
# actually works to, and daily would imply a precision the forecast lacks.
DECISION_STEP_DAYS = 7

# Cost assumptions. Every one is an explicit, disclosed default that the
# caller can override, in the same way the loadable-tonnes engine takes tide
# height rather than inventing one. See SYNTHETIC_DATA.md entries 4 and 8.
DEFAULT_DEMURRAGE_USD_PER_DAY = 22_000.0
DEFAULT_BUNKER_USD_PER_TONNE = 560.0
DEFAULT_INVENTORY_CARRY_USD_PER_T_PER_DAY = 0.012
DEFAULT_STOCKOUT_PENALTY_USD_PER_T_PER_DAY = 0.35

# Days-of-burn at or below which delay starts carrying a stockout penalty.
CRITICAL_DAYS_OF_BURN = 7.0

DEFAULT_RISK_AVERSION = 0.35
CVAR_LEVEL = 0.95

# Contract-mix grid: share of volume committed to term cover.
TERM_SPLITS = (0.0, 0.25, 0.50, 0.75, 1.0)

# A term contract trades a premium over expected spot for reduced dispersion.
# Stated as an assumption; a real desk would price this from broker quotes.
TERM_PREMIUM_FRACTION = 0.04
TERM_DISPERSION_FACTOR = 0.30


@dataclass
class DecisionPoint:
    """One date on the decision calendar, with its cost distribution."""

    decision_date: date
    horizon_days: int
    freight_median: float
    freight_p10: float
    freight_p90: float
    expected_cost_usd_per_t: float
    cvar_cost_usd_per_t: float
    fix_now_preferred: bool = False

    def as_dict(self) -> dict:
        return {
            "date": self.decision_date.isoformat(),
            "horizon_days": self.horizon_days,
            "freight_median": round(self.freight_median, 3),
            "freight_p10": round(self.freight_p10, 3),
            "freight_p90": round(self.freight_p90, 3),
            "expected_cost_usd_per_t": round(self.expected_cost_usd_per_t, 3),
            "cvar_cost_usd_per_t": round(self.cvar_cost_usd_per_t, 3),
            "fix_now_preferred": self.fix_now_preferred,
        }


@dataclass
class ContractMixPoint:
    """One point on the spot/term efficient frontier."""

    term_share: float
    expected_cost_usd_per_t: float
    cvar_cost_usd_per_t: float

    def as_dict(self) -> dict:
        return {
            "term_share": round(self.term_share, 3),
            "expected_cost_usd_per_t": round(self.expected_cost_usd_per_t, 3),
            "cvar_cost_usd_per_t": round(self.cvar_cost_usd_per_t, 3),
        }


@dataclass
class EntryWindow:
    """The recommendation. ``as_dict`` follows the LoadableResult idiom."""

    recommendation: str  # FIX_NOW | WAIT | SPLIT
    signal_strength: str  # strong | weak | no_edge
    window_start: date | None
    window_end: date | None
    value_of_waiting_usd_per_t: float
    value_of_waiting_usd_total: float
    p_better_later: float
    regret_if_wrong_usd: float
    fix_now_cost_usd_per_t: float
    best_expected_cost_usd_per_t: float
    decision_path: list[DecisionPoint] = field(default_factory=list)
    contract_mix: list[ContractMixPoint] = field(default_factory=list)
    recommended_term_share: float = 0.0
    routing: dict | None = None
    assumptions: dict = field(default_factory=dict)
    scenario_notes: list[str] = field(default_factory=list)
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "signal_strength": self.signal_strength,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "value_of_waiting_usd_per_t": round(self.value_of_waiting_usd_per_t, 3),
            "value_of_waiting_usd_total": round(self.value_of_waiting_usd_total, 0),
            "p_better_later": round(self.p_better_later, 4),
            "regret_if_wrong_usd": round(self.regret_if_wrong_usd, 0),
            "fix_now_cost_usd_per_t": round(self.fix_now_cost_usd_per_t, 3),
            "best_expected_cost_usd_per_t": round(self.best_expected_cost_usd_per_t, 3),
            "recommended_term_share": round(self.recommended_term_share, 3),
            "decision_path": [point.as_dict() for point in self.decision_path],
            "contract_mix": [point.as_dict() for point in self.contract_mix],
            "routing": self.routing,
            "assumptions": self.assumptions,
            "scenario_notes": self.scenario_notes,
            "explanation": self.explanation,
        }


def decision_calendar(today: date, not_later_than: date) -> list[date]:
    """Weekly decision dates from today to the deadline, inclusive."""
    if not_later_than <= today:
        return [today]
    dates = []
    cursor = today
    while cursor < not_later_than:
        dates.append(cursor)
        cursor += timedelta(days=DECISION_STEP_DAYS)
    dates.append(not_later_than)
    return dates


def interpolate_quantiles(
    forecast_bands: list[dict], horizon_days: int
) -> dict[float, float]:
    """Calibrated quantiles at an arbitrary horizon, from the fitted grid.

    Interpolates each quantile level separately across the horizon grid, which
    preserves monotonicity in the quantile dimension and widening in the
    horizon dimension.
    """
    if not forecast_bands:
        return {}

    horizons = np.array([band["horizon_days"] for band in forecast_bands], dtype=float)
    levels = sorted(float(level) for level in forecast_bands[0]["quantiles"])

    out: dict[float, float] = {}
    for level in levels:
        values = np.array(
            [float(band["quantiles"][str(level)]) for band in forecast_bands],
            dtype=float,
        )
        out[level] = float(np.interp(horizon_days, horizons, values))

    ordered = np.sort(np.array([out[level] for level in levels]))
    return {level: float(value) for level, value in zip(levels, ordered)}


def _carry_cost_per_t(days: float, days_of_burn: float | None) -> float:
    """Cost of not having fixed yet: inventory carry plus stockout pressure.

    The stockout term is what stops the engine from always preferring to wait.
    It scales with how close the coastal plants are to running dry, which is a
    real, free, daily signal (CEA) rather than a tuning knob.
    """
    carry = DEFAULT_INVENTORY_CARRY_USD_PER_T_PER_DAY * days
    if days_of_burn is None or days_of_burn >= CRITICAL_DAYS_OF_BURN:
        return carry
    urgency = (CRITICAL_DAYS_OF_BURN - days_of_burn) / CRITICAL_DAYS_OF_BURN
    return carry + DEFAULT_STOCKOUT_PENALTY_USD_PER_T_PER_DAY * urgency * days


def _delivered_cost(
    freight: np.ndarray,
    waiting: np.ndarray,
    bunker_multiplier: np.ndarray,
    *,
    tonnes: float,
    demurrage_usd_per_day: float,
    bunker_cost_per_t: float,
) -> np.ndarray:
    """Delivered $/t per scenario: freight + bunkers + expected demurrage.

    Demurrage is charged on waiting beyond the free laytime the parcel earns,
    which is why the waiting-time model feeds directly into the timing
    decision rather than sitting on a separate screen.
    """
    laytime_days = max(tonnes / 25_000.0, 1.0)  # indicative, at 25,000 t/day
    demurrage_days = np.maximum(waiting - laytime_days, 0.0)
    demurrage_per_t = demurrage_days * demurrage_usd_per_day / max(tonnes, 1.0)
    return freight + bunker_cost_per_t * bunker_multiplier + demurrage_per_t


def solve_stopping(
    costs: np.ndarray,
    calendar: list[date],
    *,
    risk_aversion: float,
    days_of_burn: float | None,
) -> tuple[np.ndarray, list[bool]]:
    """Backward induction over the decision calendar.

    ``costs`` is (n_scenarios, n_dates) delivered cost if fixed on that date.
    Returns the value function at each date and whether fixing is optimal
    there.

    The recursion is evaluated on the *risk-adjusted* objective, not on the
    expectation, so a date whose mean is attractive but whose tail is ugly does
    not win. That is the whole point of the CVaR term.
    """
    n_scenarios, n_dates = costs.shape
    value = np.empty_like(costs)
    fix_preferred = [False] * n_dates

    # At the deadline there is no option left.
    value[:, -1] = costs[:, -1]
    fix_preferred[-1] = True

    for index in range(n_dates - 2, -1, -1):
        elapsed = (calendar[index + 1] - calendar[index]).days
        carry = _carry_cost_per_t(elapsed, days_of_burn)
        wait_value = value[:, index + 1] + carry

        fix_objective = costs[:, index].mean() + risk_aversion * cvar(
            costs[:, index], CVAR_LEVEL
        )
        wait_objective = wait_value.mean() + risk_aversion * cvar(
            wait_value, CVAR_LEVEL
        )

        if fix_objective <= wait_objective:
            value[:, index] = costs[:, index]
            fix_preferred[index] = True
        else:
            value[:, index] = wait_value

    return value, fix_preferred


def contract_mix_frontier(
    spot_costs: np.ndarray, *, term_premium: float = TERM_PREMIUM_FRACTION
) -> list[ContractMixPoint]:
    """The spot/term frontier: the stated objective of the problem statement.

    A term contract is modelled as paying a premium over expected spot in
    exchange for a compressed distribution. Both parameters are disclosed
    assumptions — a real desk would price them from broker quotes — but the
    *shape* of the trade-off is the point, and it is what turns "should we
    move from spot to period cover?" into a number.
    """
    expected = float(spot_costs.mean())
    term_rate = expected * (1.0 + term_premium)
    term_costs = term_rate + (spot_costs - expected) * TERM_DISPERSION_FACTOR

    frontier: list[ContractMixPoint] = []
    for share in TERM_SPLITS:
        blended = share * term_costs + (1.0 - share) * spot_costs
        frontier.append(
            ContractMixPoint(
                term_share=share,
                expected_cost_usd_per_t=float(blended.mean()),
                cvar_cost_usd_per_t=cvar(blended, CVAR_LEVEL),
            )
        )
    return frontier


def _choose_term_share(
    frontier: list[ContractMixPoint], risk_aversion: float
) -> float:
    """Pick the frontier point that minimises the same risk-adjusted objective."""
    if not frontier:
        return 0.0
    scored = [
        (
            point.expected_cost_usd_per_t + risk_aversion * point.cvar_cost_usd_per_t,
            point.term_share,
        )
        for point in frontier
    ]
    return min(scored)[1]


def recommend_entry_window(
    *,
    discharge_port: str,
    required_tonnes: float,
    cargo: str,
    not_later_than: date,
    forecast_bands: list[dict],
    waiting_median_days: float,
    waiting_p95_days: float,
    point_skill: float | None,
    load_port: str | None = None,
    today: date | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    days_of_burn: float | None = None,
    demurrage_usd_per_day: float = DEFAULT_DEMURRAGE_USD_PER_DAY,
    bunker_usd_per_tonne: float = DEFAULT_BUNKER_USD_PER_TONNE,
    n_scenarios: int = 1000,
    regime_changed_recently: bool = False,
) -> EntryWindow:
    """Recommend when to enter the market for one cargo requirement."""
    today = today or date.today()

    # 1. Feasibility first. Never time a voyage that cannot physically happen.
    screen = screen_all(
        discharge_port,
        required_tonnes,
        load_port=load_port,
        cargo_name=cargo,
    )
    feasible = screen.get("feasible") or []
    if not feasible:
        return EntryWindow(
            recommendation="INFEASIBLE",
            signal_strength="no_edge",
            window_start=None,
            window_end=None,
            value_of_waiting_usd_per_t=0.0,
            value_of_waiting_usd_total=0.0,
            p_better_later=0.0,
            regret_if_wrong_usd=0.0,
            fix_now_cost_usd_per_t=0.0,
            best_expected_cost_usd_per_t=0.0,
            routing=None,
            explanation=(
                f"No feasible routing moves {required_tonnes:,.0f} t of {cargo} into "
                f"{discharge_port}. Timing is moot until the cargo can be discharged; "
                f"{screen.get('summary', '')}"
            ),
        )
    best_routing = feasible[0]

    # 2. Decision calendar and the forecast at each point on it.
    calendar = decision_calendar(today, not_later_than)
    quantiles_by_date = [
        interpolate_quantiles(forecast_bands, (day - today).days) for day in calendar
    ]
    if not any(quantiles_by_date):
        raise ValueError("no forecast bands supplied; cannot build a decision calendar")

    # 3. Joint scenarios preserving the correlation structure.
    scenarios: ScenarioSet = generate(
        freight_quantiles_by_date=quantiles_by_date,
        waiting_median_days=waiting_median_days,
        waiting_p95_days=waiting_p95_days,
        n_scenarios=n_scenarios,
    )

    # 4. Delivered cost per scenario per decision date.
    bunker_cost_per_t = bunker_usd_per_tonne * 0.004  # indicative tonnes burnt per t
    costs = np.empty_like(scenarios.freight_usd_per_t)
    for index in range(len(calendar)):
        costs[:, index] = _delivered_cost(
            scenarios.freight_usd_per_t[:, index],
            scenarios.waiting_days,
            scenarios.bunker_multiplier,
            tonnes=required_tonnes,
            demurrage_usd_per_day=demurrage_usd_per_day,
            bunker_cost_per_t=bunker_cost_per_t,
        )

    # 5. Backward induction.
    value, fix_preferred = solve_stopping(
        costs,
        calendar,
        risk_aversion=risk_aversion,
        days_of_burn=days_of_burn,
    )

    fix_now_cost = float(costs[:, 0].mean())
    best_expected = float(min(costs[:, index].mean() for index in range(len(calendar))))
    value_of_waiting = fix_now_cost - float(value[:, 0].mean())

    # Probability that any later date beats fixing today, scenario by scenario.
    if costs.shape[1] > 1:
        later_best = costs[:, 1:].min(axis=1)
        p_better_later = float((later_best < costs[:, 0]).mean())
        regret = (
            cvar(later_best - costs[:, 0], CVAR_LEVEL) * required_tonnes
        )
    else:
        p_better_later = 0.0
        regret = 0.0

    # 6. Signal strength. This is the honesty gate.
    signal_strength = _signal_strength(point_skill, regime_changed_recently)
    if signal_strength == "no_edge":
        # Without demonstrated skill the engine will not time the market. It
        # falls back to the deadline-driven answer and says so.
        fix_preferred = [True] + [False] * (len(calendar) - 1)
        value_of_waiting = 0.0

    window_start, window_end = _window_from(calendar, fix_preferred)

    frontier = contract_mix_frontier(costs[:, 0])
    term_share = _choose_term_share(frontier, risk_aversion)

    recommendation = _recommendation(
        fix_preferred, signal_strength, term_share, value_of_waiting
    )

    decision_path = [
        DecisionPoint(
            decision_date=day,
            horizon_days=(day - today).days,
            freight_median=quantiles_by_date[index].get(0.5, float("nan")),
            freight_p10=quantiles_by_date[index].get(0.10, float("nan")),
            freight_p90=quantiles_by_date[index].get(0.90, float("nan")),
            expected_cost_usd_per_t=float(costs[:, index].mean()),
            cvar_cost_usd_per_t=cvar(costs[:, index], CVAR_LEVEL),
            fix_now_preferred=fix_preferred[index],
        )
        for index, day in enumerate(calendar)
    ]

    window = EntryWindow(
        recommendation=recommendation,
        signal_strength=signal_strength,
        window_start=window_start,
        window_end=window_end,
        value_of_waiting_usd_per_t=value_of_waiting,
        value_of_waiting_usd_total=value_of_waiting * required_tonnes,
        p_better_later=p_better_later,
        regret_if_wrong_usd=regret,
        fix_now_cost_usd_per_t=fix_now_cost,
        best_expected_cost_usd_per_t=best_expected,
        decision_path=decision_path,
        contract_mix=frontier,
        recommended_term_share=term_share,
        routing={
            "vessel_class": best_routing.get("vessel_class"),
            "routing": best_routing.get("routing"),
            "discharge_port": best_routing.get("discharge_port"),
            "berth": best_routing.get("berth_name"),
            "loadable_tonnes": best_routing.get("loadable_tonnes"),
            "vessels_required": best_routing.get("vessels_required"),
            "limiting_factor": best_routing.get("limiting_factor"),
            "explanation": best_routing.get("explanation"),
        },
        assumptions={
            "demurrage_usd_per_day": demurrage_usd_per_day,
            "bunker_usd_per_tonne": bunker_usd_per_tonne,
            "inventory_carry_usd_per_t_per_day": (
                DEFAULT_INVENTORY_CARRY_USD_PER_T_PER_DAY
            ),
            "risk_aversion_lambda": risk_aversion,
            "cvar_level": CVAR_LEVEL,
            "term_premium_fraction": TERM_PREMIUM_FRACTION,
            "days_of_burn": days_of_burn,
            "n_scenarios": n_scenarios,
            "point_skill_at_horizon": point_skill,
        },
        scenario_notes=scenarios.notes,
    )
    window.explanation = _explain(window, required_tonnes, not_later_than)
    return window


def _signal_strength(point_skill: float | None, regime_changed: bool) -> str:
    """Translate measured backtest skill into a claim the UI may make.

    An unmeasured model is treated as having no edge, not as fine. A fresh
    regime change downgrades a strong signal, because the conformal guarantee
    assumes exchangeability and a break is exactly when that fails.
    """
    if point_skill is None or point_skill <= 0.0:
        return "no_edge"
    if regime_changed:
        return "weak"
    return "strong" if point_skill >= 0.15 else "weak"


def _window_from(
    calendar: list[date], fix_preferred: list[bool]
) -> tuple[date | None, date | None]:
    """The first contiguous run of dates on which fixing is optimal."""
    start = None
    for index, preferred in enumerate(fix_preferred):
        if preferred and start is None:
            start = index
        elif not preferred and start is not None:
            return calendar[start], calendar[index - 1]
    if start is not None:
        return calendar[start], calendar[-1]
    return None, None


def _recommendation(
    fix_preferred: list[bool],
    signal_strength: str,
    term_share: float,
    value_of_waiting: float,
) -> str:
    if fix_preferred and fix_preferred[0]:
        return "SPLIT" if 0.0 < term_share < 1.0 and signal_strength != "no_edge" else "FIX_NOW"
    return "WAIT"


def _explain(window: EntryWindow, tonnes: float, deadline: date) -> str:
    """Plain English, in the house style of the loadable-tonnes engine."""
    routing = window.routing or {}
    vessel = routing.get("vessel_class", "a suitable vessel")
    lines: list[str] = []

    if window.signal_strength == "no_edge":
        lines.append(
            "The rate forecaster does not beat a random walk at this horizon, so "
            "the platform will not time the market on this cargo. The "
            "recommendation below is driven by feasibility, carrying cost and "
            "the deadline alone."
        )
    else:
        lines.append(
            f"Forecast signal is {window.signal_strength} at this horizon "
            f"(measured skill against the hardest baseline)."
        )

    if window.recommendation == "WAIT":
        lines.append(
            f"Waiting is worth ${window.value_of_waiting_usd_per_t:.2f}/t "
            f"(${window.value_of_waiting_usd_total:,.0f} on {tonnes:,.0f} t). "
            f"There is a {window.p_better_later:.0%} chance a later date beats "
            f"fixing today."
        )
        if window.window_start:
            lines.append(
                f"The engine expects to fix between "
                f"{window.window_start.isoformat()} and "
                f"{window.window_end.isoformat()}."
            )
    elif window.signal_strength == "no_edge":
        # Do not dress a forced default up as a preference. The scenarios may
        # well show a cheaper later date; the point is that we have no
        # measured ability to identify which one, so acting on it would be
        # guessing with a confident face on.
        lines.append(
            f"Fixing now at an expected ${window.fix_now_cost_usd_per_t:.2f}/t is "
            f"the default, not a prediction that today is cheapest. The scenarios "
            f"put the best calendar date at ${window.best_expected_cost_usd_per_t:.2f}/t "
            f"and give a {window.p_better_later:.0%} chance that some later date "
            f"beats today - but with no demonstrated skill at this horizon we "
            f"cannot say which, and waiting on an unmeasured signal is a gamble "
            f"rather than a strategy."
        )
    else:
        lines.append(
            f"Fixing now is preferred: expected delivered cost "
            f"${window.fix_now_cost_usd_per_t:.2f}/t against a best achievable "
            f"${window.best_expected_cost_usd_per_t:.2f}/t elsewhere on the "
            f"calendar, and delay carries inventory and stockout cost."
        )

    lines.append(
        f"Risk of waiting and being wrong: ${window.regret_if_wrong_usd:,.0f} at the "
        f"95th percentile."
    )

    if window.recommended_term_share > 0:
        lines.append(
            f"On the contract-mix frontier, committing "
            f"{window.recommended_term_share:.0%} of this volume to term cover "
            f"minimises the risk-adjusted cost."
        )

    lines.append(f"Routing: {routing.get('explanation', vessel)}")
    lines.append(f"Cargo must be fixed by {deadline.isoformat()}.")
    return " ".join(lines)
