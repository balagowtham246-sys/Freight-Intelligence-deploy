"""Joint scenario generation for the timing engine.

The timing decision is taken under uncertainty in several correlated
quantities at once — freight rate, bunker cost, waiting time at the discharge
port — and the objective has a CVaR term, so what matters most is the *tail*
of total delivered cost. Getting the tail right is almost entirely a question
of getting the dependence right.

Why independent draws would be wrong
------------------------------------
Sampling each quantity independently understates the tail badly. In a firm
market freight is high *and* congestion is heavy *and* bunkers are up, because
all three are driven by the same demand surge. Independent draws let the good
and bad news cancel, which produces a tight, comfortable cost distribution and
a CVaR that is far too flattering. The whole risk-averse objective would then
be optimising against a fiction.

The method
----------
1. Draw a common shock per scenario and combine it with an idiosyncratic shock
   at a specified correlation, so the marginals stay standard normal while the
   joint carries the dependence — a Gaussian copula, built by hand rather than
   pulled in as a dependency.
2. Map each scenario's uniform through the *calibrated predictive quantiles*
   of the corresponding series, by monotone interpolation. This is the
   inverse-transform step, and it means the scenarios inherit the conformal
   calibration rather than assuming a shape.
3. Extrapolate beyond the outermost quantiles with an exponential tail rather
   than clamping. Clamping at q05/q95 would truncate exactly the tail the CVaR
   term is there to measure.

The correlation used is a stated assumption, not a fitted one — the archive is
far too short to estimate a freight/congestion correlation, and pretending
otherwise would be worse than declaring a defensible prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Assumed dependence between the drivers. Positive: a firm market means high
# freight, heavy congestion and strong bunker demand together.
DEFAULT_CORRELATION = 0.45

# Tail behaviour outside the quantile grid, as a multiple of the outer
# inter-quantile gap per unit of tail probability.
TAIL_DECAY = 1.5


@dataclass
class ScenarioSet:
    """Joint draws of the quantities the cost function needs."""

    freight_usd_per_t: np.ndarray  # (n_scenarios, n_decision_dates)
    waiting_days: np.ndarray
    bunker_multiplier: np.ndarray
    n_scenarios: int
    correlation: float
    seed: int
    notes: list[str] = field(default_factory=list)


def _quantile_interpolate(
    uniforms: np.ndarray, levels: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Inverse-CDF by interpolation, with exponential tails outside the grid.

    Inside the grid this is plain monotone interpolation of the calibrated
    quantiles. Outside it, the value decays at the rate implied by the
    outermost gap, so extreme scenarios stay finite and ordered without being
    truncated at q05/q95.
    """
    drawn = np.interp(uniforms, levels, values)

    low_mask = uniforms < levels[0]
    if low_mask.any():
        gap = max(values[1] - values[0], 1e-9)
        slope = gap / max(levels[1] - levels[0], 1e-9)
        excess = (levels[0] - uniforms[low_mask]) * slope * TAIL_DECAY
        drawn[low_mask] = values[0] - excess

    high_mask = uniforms > levels[-1]
    if high_mask.any():
        gap = max(values[-1] - values[-2], 1e-9)
        slope = gap / max(levels[-1] - levels[-2], 1e-9)
        excess = (uniforms[high_mask] - levels[-1]) * slope * TAIL_DECAY
        drawn[high_mask] = values[-1] + excess

    return drawn


def _correlated_uniforms(
    rng: np.random.Generator, n_scenarios: int, n_series: int, correlation: float
) -> np.ndarray:
    """Gaussian copula: correlated uniforms with exact standard-normal marginals."""
    from scipy.stats import norm

    weight = np.sqrt(max(min(correlation, 0.999), 0.0))
    common = rng.standard_normal((n_scenarios, 1))
    idiosyncratic = rng.standard_normal((n_scenarios, n_series))
    joint = weight * common + np.sqrt(1.0 - weight**2) * idiosyncratic
    return norm.cdf(joint)


def generate(
    *,
    freight_quantiles_by_date: list[dict[float, float]],
    waiting_median_days: float,
    waiting_p95_days: float,
    n_scenarios: int = 1000,
    correlation: float = DEFAULT_CORRELATION,
    bunker_vol: float = 0.20,
    seed: int = 20260903,
) -> ScenarioSet:
    """Draw joint scenarios across the decision calendar.

    ``freight_quantiles_by_date`` is one calibrated quantile dict per decision
    date. Freight is drawn with a *persistent* rank across dates within a
    scenario, because a scenario in which the market is firm should be firm
    throughout rather than re-rolling the dice each week — re-rolling would
    average the path out and destroy exactly the persistence that makes
    waiting a real decision.
    """
    rng = np.random.default_rng(seed)
    n_dates = len(freight_quantiles_by_date)
    if n_dates == 0:
        raise ValueError("no decision dates supplied")

    uniforms = _correlated_uniforms(rng, n_scenarios, 3, correlation)
    freight_rank = uniforms[:, 0]
    waiting_rank = uniforms[:, 1]
    bunker_rank = uniforms[:, 2]

    # Let the rank drift slowly so a scenario is persistent but not frozen.
    drift = rng.standard_normal((n_scenarios, n_dates)) * 0.08
    drifted = np.clip(
        freight_rank[:, None] + np.cumsum(drift, axis=1), 0.001, 0.999
    )

    freight = np.empty((n_scenarios, n_dates))
    for index, quantiles in enumerate(freight_quantiles_by_date):
        levels = np.array(sorted(quantiles), dtype=float)
        values = np.array([quantiles[level] for level in levels], dtype=float)
        freight[:, index] = _quantile_interpolate(drifted[:, index], levels, values)
    freight = np.maximum(freight, 0.01)

    # Waiting time: lognormal shaped to hit the model's median and p95.
    median = max(waiting_median_days, 0.1)
    p95 = max(waiting_p95_days, median * 1.01)
    sigma = max(np.log(p95 / median) / 1.645, 1e-3)
    from scipy.stats import norm

    waiting = median * np.exp(sigma * norm.ppf(waiting_rank))
    waiting = np.clip(waiting, 0.0, 60.0)

    bunker = np.exp(bunker_vol * norm.ppf(bunker_rank) - 0.5 * bunker_vol**2)

    return ScenarioSet(
        freight_usd_per_t=freight,
        waiting_days=waiting,
        bunker_multiplier=bunker,
        n_scenarios=n_scenarios,
        correlation=correlation,
        seed=seed,
        notes=[
            f"Gaussian copula, correlation {correlation:.2f} (assumed, not fitted - "
            "the archive is too short to estimate it)",
            f"freight drawn by inverse-transform through the calibrated quantiles "
            f"at {n_dates} decision dates, with persistent rank",
            f"waiting time lognormal fitted to median {median:.1f}d and p95 {p95:.1f}d",
        ],
    )


def cvar(costs: np.ndarray, level: float = 0.95) -> float:
    """Conditional value at risk: mean cost in the worst ``1 - level`` tail.

    This is the number the risk-aversion term prices, and the reason a
    chartering desk wants term cover in the first place.
    """
    values = np.asarray(costs, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    threshold = np.quantile(values, level)
    tail = values[values >= threshold]
    return float(tail.mean()) if tail.size else float(threshold)
