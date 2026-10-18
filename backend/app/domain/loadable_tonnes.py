"""Loadable-tonnes and feasibility engine.

This answers one question precisely: *given this vessel, this berth, this
tide and this cargo, how many tonnes can actually be loaded, and if the
answer is zero, exactly which constraint bit?*

It is deliberately deterministic. Nothing here is forecast or learned, so
the number it produces is auditable and can be checked against a loading
computer. That is what makes it usable for a real decision.

The physics
-----------
A vessel floats at the draft where displaced water weight equals its own.
Displaced volume is fixed by hull geometry at a given draft, so displacement
scales with water density:

    displacement(draft, rho) = volume(draft) * rho

Volume is taken from the summer condition and adjusted linearly with TPC,
which is accurate over the small draft range that matters here:

    displacement_salt(draft) = (lightship + dwt) - (summer_draft - draft) * 100 * tpc
    displacement(draft, rho) = displacement_salt(draft) * rho / 1.025

Cargo is then whatever is left after the ship itself, its fuel and its
constants:

    cargo = displacement(draft, rho) - lightship - bunkers - constants

Four separate limits can bind, and the engine reports which one did:

1. **Draft**    - berth depth, plus tide if the port works vessels on a tide
2. **Deadweight** - the vessel cannot exceed its own DWT
3. **Volume**   - low calorific value coal fills the holds before the marks
4. **Dimensional** - LOA, beam and air draft are hard rejections

Why the density term earns its place: Haldia sits in brackish Hooghly water
at about 1.008 t/m3 and the Mississippi is fresh at 1.000. Ignoring that
overstates loadable cargo on exactly the lanes where parcels are tightest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SALT_DENSITY = 1.025


class LimitingFactor(str, Enum):
    DRAFT = "draft"
    DEADWEIGHT = "deadweight"
    VOLUME = "volume"
    NONE = "none"


class RejectionReason(str, Enum):
    LOA = "loa"
    BEAM = "beam"
    AIR_DRAFT = "air_draft"
    MAX_DWT = "max_dwt"
    GEAR = "gear"
    CARGO_TYPE = "cargo_type"
    NO_CARGO_POSSIBLE = "no_cargo_possible"


@dataclass(frozen=True)
class Vessel:
    """Vessel particulars. Either a class archetype or a named candidate."""

    name: str
    dwt: float
    loa_m: float
    beam_m: float
    summer_draft_m: float
    lightship_t: float
    tpc: float
    grain_cubic_m3: float
    geared: bool = False
    constants_t: float = 500.0
    air_draft_m: float | None = None

    @property
    def summer_displacement_t(self) -> float:
        return self.lightship_t + self.dwt


@dataclass(frozen=True)
class BerthLimits:
    """The constraint set for one berth at one port."""

    port_code: str
    berth_name: str
    max_draft_m: float
    water_density: float = SALT_DENSITY
    max_loa_m: float | None = None
    max_beam_m: float | None = None
    max_dwt: float | None = None
    max_air_draft_m: float | None = None
    requires_gear: bool = False
    cargo_types: tuple[str, ...] = ()
    discharge_rate_tpd: float | None = None
    # True when the port works vessels on a tide, so tidal height adds to the
    # permissible sailing draft rather than being ignored.
    tide_constrained: bool = False


@dataclass(frozen=True)
class Cargo:
    """A cargo grade.

    ``name`` identifies the specific grade and drives stowage factor and
    calorific value. ``category`` is the generic commodity that a berth is
    equipped to handle, and is what the berth compatibility check matches on:
    a berth rated for "thermal_coal" takes any thermal grade.
    """

    name: str
    stowage_factor_m3_per_t: float
    gcv_kcal_per_kg: float = 0.0
    category: str | None = None

    @property
    def berth_category(self) -> str:
        return self.category or self.name


@dataclass
class LoadableResult:
    """The full worked answer, including the arithmetic behind it."""

    feasible: bool
    loadable_tonnes: float = 0.0
    limiting_factor: LimitingFactor = LimitingFactor.NONE
    rejections: list[RejectionReason] = field(default_factory=list)

    permissible_draft_m: float = 0.0
    tide_credit_m: float = 0.0
    density_penalty_t: float = 0.0
    displacement_t: float = 0.0

    draft_limited_t: float = 0.0
    dwt_limited_t: float = 0.0
    volume_limited_t: float | None = None

    explanation: str = ""
    # Every intermediate value, so the UI can show the working rather than
    # asking the user to trust a single number.
    workings: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "loadable_tonnes": round(self.loadable_tonnes, 1),
            "limiting_factor": self.limiting_factor.value,
            "rejections": [r.value for r in self.rejections],
            "permissible_draft_m": round(self.permissible_draft_m, 3),
            "tide_credit_m": round(self.tide_credit_m, 3),
            "density_penalty_t": round(self.density_penalty_t, 1),
            "displacement_t": round(self.displacement_t, 1),
            "draft_limited_t": round(self.draft_limited_t, 1),
            "dwt_limited_t": round(self.dwt_limited_t, 1),
            "volume_limited_t": (
                round(self.volume_limited_t, 1) if self.volume_limited_t is not None else None
            ),
            "explanation": self.explanation,
            "workings": {k: round(v, 3) for k, v in self.workings.items()},
        }


def displacement_at_draft(vessel: Vessel, draft_m: float, density: float) -> float:
    """Displacement at a given physical draft in water of a given density."""
    draft_deficit_m = vessel.summer_draft_m - draft_m
    salt_displacement = vessel.summer_displacement_t - draft_deficit_m * 100.0 * vessel.tpc
    return salt_displacement * (density / SALT_DENSITY)


def fresh_water_allowance_mm(vessel: Vessel) -> float:
    """FWA in millimetres: the standard displacement / (4 * TPC) relation."""
    return vessel.summer_displacement_t / (4.0 * vessel.tpc)


def dock_water_allowance_m(vessel: Vessel, density: float) -> float:
    """How much deeper the vessel floats in water lighter than full salt."""
    fwa_mm = fresh_water_allowance_mm(vessel)
    dwa_mm = fwa_mm * (SALT_DENSITY - density) / 0.025
    return dwa_mm / 1000.0


def calculate_loadable_tonnes(
    vessel: Vessel,
    berth: BerthLimits,
    *,
    cargo: Cargo | None = None,
    bunkers_rob_t: float = 1200.0,
    tide_height_m: float = 0.0,
    ukc_policy_m: float = 0.0,
) -> LoadableResult:
    """Compute loadable cargo, or explain precisely why the vessel cannot go.

    ``tide_height_m`` is the predicted height of tide at the planned time of
    the movement. It is credited only at ports that actually work vessels on
    a tide; elsewhere the published permissible draft already stands alone.

    ``ukc_policy_m`` is an additional under-keel clearance the charterer wants
    to hold back beyond the port requirement, in metres of draft.
    """
    result = LoadableResult(feasible=True)

    # ---------------------------------------------------------- hard limits
    if berth.max_loa_m is not None and vessel.loa_m > berth.max_loa_m:
        result.rejections.append(RejectionReason.LOA)
    if berth.max_beam_m is not None and vessel.beam_m > berth.max_beam_m:
        result.rejections.append(RejectionReason.BEAM)
    if (
        berth.max_air_draft_m is not None
        and vessel.air_draft_m is not None
        and vessel.air_draft_m > berth.max_air_draft_m
    ):
        result.rejections.append(RejectionReason.AIR_DRAFT)
    if berth.max_dwt is not None and vessel.dwt > berth.max_dwt:
        result.rejections.append(RejectionReason.MAX_DWT)
    if berth.requires_gear and not vessel.geared:
        result.rejections.append(RejectionReason.GEAR)
    if cargo and berth.cargo_types and cargo.berth_category not in berth.cargo_types:
        result.rejections.append(RejectionReason.CARGO_TYPE)

    # ------------------------------------------------------- draft envelope
    tide_credit = tide_height_m if berth.tide_constrained else 0.0
    permissible_draft = berth.max_draft_m + tide_credit - ukc_policy_m
    # A vessel never loads past its own summer marks regardless of depth.
    governing_draft = min(permissible_draft, vessel.summer_draft_m)

    result.permissible_draft_m = permissible_draft
    result.tide_credit_m = tide_credit

    if result.rejections:
        result.feasible = False
        result.explanation = _explain_rejection(vessel, berth, result.rejections)
        return result

    if governing_draft <= 0:
        result.feasible = False
        result.rejections.append(RejectionReason.NO_CARGO_POSSIBLE)
        result.explanation = (
            f"{vessel.name} rejected at {berth.port_code} {berth.berth_name}: "
            f"no positive permissible draft available."
        )
        return result

    # ---------------------------------------------------------- displacement
    displacement = displacement_at_draft(vessel, governing_draft, berth.water_density)
    displacement_if_salt = displacement_at_draft(vessel, governing_draft, SALT_DENSITY)
    density_penalty = displacement_if_salt - displacement

    deadweight_available = displacement - vessel.lightship_t
    draft_limited = deadweight_available - bunkers_rob_t - vessel.constants_t
    dwt_limited = vessel.dwt - bunkers_rob_t - vessel.constants_t

    result.displacement_t = displacement
    result.density_penalty_t = density_penalty
    result.draft_limited_t = max(draft_limited, 0.0)
    result.dwt_limited_t = max(dwt_limited, 0.0)

    # Deadweight is listed first so that ties resolve to it. At a berth deep
    # enough for the vessel to reach its summer marks the two figures are
    # mathematically identical, and "on her marks" is the honest description.
    candidates: list[tuple[float, LimitingFactor]] = [
        (dwt_limited, LimitingFactor.DEADWEIGHT),
        (draft_limited, LimitingFactor.DRAFT),
    ]

    # ------------------------------------------------------------- cubic cap
    if cargo is not None and cargo.stowage_factor_m3_per_t > 0:
        volume_limited = vessel.grain_cubic_m3 / cargo.stowage_factor_m3_per_t
        result.volume_limited_t = volume_limited
        candidates.append((volume_limited, LimitingFactor.VOLUME))

    loadable, limiting = min(candidates, key=lambda pair: pair[0])

    if loadable <= 0:
        result.feasible = False
        result.rejections.append(RejectionReason.NO_CARGO_POSSIBLE)
        result.loadable_tonnes = 0.0
        result.limiting_factor = limiting
        result.explanation = (
            f"{vessel.name} cannot lift cargo at {berth.port_code} "
            f"{berth.berth_name}: permissible draft {permissible_draft:.2f} m "
            f"leaves no deadweight after {bunkers_rob_t:,.0f} t bunkers and "
            f"{vessel.constants_t:,.0f} t constants."
        )
        return result

    result.loadable_tonnes = loadable
    result.limiting_factor = limiting
    result.workings = {
        "summer_displacement_t": vessel.summer_displacement_t,
        "governing_draft_m": governing_draft,
        "berth_max_draft_m": berth.max_draft_m,
        "tide_credit_m": tide_credit,
        "ukc_holdback_m": ukc_policy_m,
        "water_density": berth.water_density,
        "fwa_mm": fresh_water_allowance_mm(vessel),
        "dwa_m": dock_water_allowance_m(vessel, berth.water_density),
        "displacement_t": displacement,
        "lightship_t": vessel.lightship_t,
        "bunkers_rob_t": bunkers_rob_t,
        "constants_t": vessel.constants_t,
    }
    result.explanation = _explain_success(vessel, berth, result, limiting, governing_draft)
    return result


def _explain_rejection(
    vessel: Vessel, berth: BerthLimits, reasons: list[RejectionReason]
) -> str:
    parts: list[str] = []
    for reason in reasons:
        if reason is RejectionReason.LOA:
            parts.append(f"LOA {vessel.loa_m:.1f} m exceeds berth limit {berth.max_loa_m:.1f} m")
        elif reason is RejectionReason.BEAM:
            parts.append(f"beam {vessel.beam_m:.2f} m exceeds berth limit {berth.max_beam_m:.2f} m")
        elif reason is RejectionReason.MAX_DWT:
            parts.append(f"DWT {vessel.dwt:,.0f} exceeds berth limit {berth.max_dwt:,.0f}")
        elif reason is RejectionReason.GEAR:
            parts.append("berth requires geared tonnage and this vessel is gearless")
        elif reason is RejectionReason.AIR_DRAFT:
            parts.append("air draft exceeds the berth limit")
        elif reason is RejectionReason.CARGO_TYPE:
            parts.append("berth does not handle this cargo")
    return (
        f"{vessel.name} rejected at {berth.port_code} {berth.berth_name}: "
        + "; ".join(parts)
        + "."
    )


def _explain_success(
    vessel: Vessel,
    berth: BerthLimits,
    result: LoadableResult,
    limiting: LimitingFactor,
    governing_draft: float,
) -> str:
    base = (
        f"{vessel.name} at {berth.port_code} {berth.berth_name}: "
        f"{result.loadable_tonnes:,.0f} t loadable "
        f"on {governing_draft:.2f} m draft"
    )
    if result.tide_credit_m:
        base += f" (includes {result.tide_credit_m:.2f} m of tide)"

    if limiting is LimitingFactor.DRAFT:
        base += (
            f". Draft limited: the berth permits {berth.max_draft_m:.2f} m against "
            f"the vessel summer draft of {vessel.summer_draft_m:.2f} m, so "
            f"{result.dwt_limited_t - result.draft_limited_t:,.0f} t of deadweight "
            f"is unusable."
        )
    elif limiting is LimitingFactor.DEADWEIGHT:
        base += ". Deadweight limited: the berth depth is not the binding constraint."
    elif limiting is LimitingFactor.VOLUME:
        base += (
            f". Volume limited: the holds cube out at {result.volume_limited_t:,.0f} t "
            f"before the marks are reached, so a denser grade would lift more."
        )

    if result.density_penalty_t > 50:
        base += (
            f" Water density {berth.water_density:.3f} t/m3 costs "
            f"{result.density_penalty_t:,.0f} t against full salt water."
        )
    return base
