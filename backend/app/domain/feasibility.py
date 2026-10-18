"""Feasibility screening across vessel classes, berths and routings.

Given a cargo requirement, this produces the ranked set of ways to move it,
each with a worked loadable-tonnage figure and a plain-English explanation of
why it is or is not possible.

Two routings are considered for every discharge port:

**Direct** - the vessel berths and discharges alongside.

**Lightering** - the vessel discharges part or all of its cargo at a
transhipment anchorage into daughter vessels or barges, which then run into a
berth the mother vessel could never reach. This is modelled as a first-class
option rather than an exception, because for East Coast India it frequently
is the answer: Haldia is limited to roughly 8.5 m draft and 230 m LOA, while
Sandheads has 40-50 m of water a short distance away. One Capesize plus
transhipment against three Supramaxes direct is a real, priced trade-off that
moves with freight rates and floating-crane availability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .loadable_tonnes import (
    BerthLimits,
    Cargo,
    LoadableResult,
    Vessel,
    calculate_loadable_tonnes,
)
from .reference import berths_for_port, load_berths, load_cargoes, load_vessel_classes

# Anchorages that can serve as lightering points, and the berths they feed.
LIGHTERING_TOPOLOGY: dict[str, list[str]] = {
    # Mother vessel discharges at Sandheads or Sagar; daughters run to Haldia.
    "INHAL": ["INSND", "INSAG"],
}

# Indicative daughter-vessel classes used for the transhipment leg.
DAUGHTER_CLASSES = ("Handysize", "Supramax")


@dataclass
class RoutingOption:
    """One way of moving the cargo, direct or via a lightering anchorage."""

    routing: str  # "direct" | "lightering"
    vessel_class: str
    discharge_port: str
    berth_name: str
    loadable_tonnes: float
    limiting_factor: str
    feasible: bool
    explanation: str

    # Lightering only.
    transhipment_port: str | None = None
    daughter_class: str | None = None
    daughter_voyages: int | None = None

    vessels_required: int = 1
    total_lifted_t: float = 0.0
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "routing": self.routing,
            "vessel_class": self.vessel_class,
            "discharge_port": self.discharge_port,
            "berth_name": self.berth_name,
            "loadable_tonnes": round(self.loadable_tonnes, 1),
            "limiting_factor": self.limiting_factor,
            "feasible": self.feasible,
            "explanation": self.explanation,
            "transhipment_port": self.transhipment_port,
            "daughter_class": self.daughter_class,
            "daughter_voyages": self.daughter_voyages,
            "vessels_required": self.vessels_required,
            "total_lifted_t": round(self.total_lifted_t, 1),
            "detail": self.detail,
        }


def _best_berth(
    vessel: Vessel,
    berths: list[BerthLimits],
    cargo: Cargo | None,
    bunkers_rob_t: float,
    tide_height_m: float,
) -> tuple[BerthLimits | None, LoadableResult | None]:
    """The berth at a port that lifts the most for this vessel."""
    best_berth: BerthLimits | None = None
    best_result: LoadableResult | None = None

    for berth in berths:
        result = calculate_loadable_tonnes(
            vessel,
            berth,
            cargo=cargo,
            bunkers_rob_t=bunkers_rob_t,
            tide_height_m=tide_height_m,
        )
        if not result.feasible:
            # Keep the first infeasible result so the reason can be reported
            # when no berth at the port works.
            if best_result is None:
                best_berth, best_result = berth, result
            continue
        if best_result is None or not best_result.feasible:
            best_berth, best_result = berth, result
        elif result.loadable_tonnes > best_result.loadable_tonnes:
            best_berth, best_result = berth, result

    return best_berth, best_result


def screen_direct(
    discharge_port: str,
    required_tonnes: float,
    *,
    cargo_name: str = "thermal_coal_mid_cv",
    bunkers_rob_t: float = 1200.0,
    tide_height_m: float = 0.0,
    vessel_classes: list[str] | None = None,
) -> list[RoutingOption]:
    """Screen every vessel class for direct discharge at one port."""
    cargo = load_cargoes().get(cargo_name)
    classes = load_vessel_classes()
    names = vessel_classes or list(classes)
    berths = berths_for_port(discharge_port)

    options: list[RoutingOption] = []
    for name in names:
        vessel = classes.get(name)
        if vessel is None:
            continue

        berth, result = _best_berth(vessel, berths, cargo, bunkers_rob_t, tide_height_m)
        if berth is None or result is None:
            continue

        vessels_required = (
            max(1, math.ceil(required_tonnes / result.loadable_tonnes))
            if result.feasible and result.loadable_tonnes > 0
            else 0
        )

        options.append(
            RoutingOption(
                routing="direct",
                vessel_class=name,
                discharge_port=discharge_port,
                berth_name=berth.berth_name,
                loadable_tonnes=result.loadable_tonnes,
                limiting_factor=result.limiting_factor.value,
                feasible=result.feasible,
                explanation=result.explanation,
                vessels_required=vessels_required,
                total_lifted_t=result.loadable_tonnes * vessels_required,
                detail=result.as_dict(),
            )
        )
    return options


def screen_lightering(
    discharge_port: str,
    required_tonnes: float,
    *,
    cargo_name: str = "thermal_coal_mid_cv",
    bunkers_rob_t: float = 1200.0,
    tide_height_m: float = 0.0,
    vessel_classes: list[str] | None = None,
) -> list[RoutingOption]:
    """Screen mother-vessel plus transhipment routings into a shallow port.

    The mother vessel is constrained by the anchorage, not the final berth,
    which is exactly why this unlocks tonnage that direct calls cannot.
    """
    anchorages = LIGHTERING_TOPOLOGY.get(discharge_port, [])
    if not anchorages:
        return []

    cargo = load_cargoes().get(cargo_name)
    classes = load_vessel_classes()
    names = vessel_classes or list(classes)
    final_berths = berths_for_port(discharge_port)

    options: list[RoutingOption] = []

    for anchorage in anchorages:
        anchorage_berths = berths_for_port(anchorage)

        for name in names:
            vessel = classes.get(name)
            if vessel is None:
                continue

            berth, result = _best_berth(
                vessel, anchorage_berths, cargo, bunkers_rob_t, tide_height_m
            )
            if berth is None or result is None or not result.feasible:
                continue

            # Size the daughter leg against the final berth.
            best_daughter = None
            best_daughter_lift = 0.0
            for daughter_name in DAUGHTER_CLASSES:
                daughter = classes.get(daughter_name)
                if daughter is None:
                    continue
                _, daughter_result = _best_berth(
                    daughter, final_berths, cargo, 400.0, tide_height_m
                )
                if daughter_result and daughter_result.feasible:
                    if daughter_result.loadable_tonnes > best_daughter_lift:
                        best_daughter = daughter_name
                        best_daughter_lift = daughter_result.loadable_tonnes

            if best_daughter is None or best_daughter_lift <= 0:
                continue

            mother_lift = result.loadable_tonnes
            daughter_voyages = max(1, math.ceil(mother_lift / best_daughter_lift))
            vessels_required = max(1, math.ceil(required_tonnes / mother_lift))

            explanation = (
                f"{name} lifts {mother_lift:,.0f} t to {anchorage}, transhipped by "
                f"{daughter_voyages} x {best_daughter} "
                f"({best_daughter_lift:,.0f} t each) into {discharge_port}. "
                f"This routing exists because {discharge_port} cannot take the "
                f"mother vessel directly."
            )

            options.append(
                RoutingOption(
                    routing="lightering",
                    vessel_class=name,
                    discharge_port=discharge_port,
                    berth_name=berth.berth_name,
                    loadable_tonnes=mother_lift,
                    limiting_factor=result.limiting_factor.value,
                    feasible=True,
                    explanation=explanation,
                    transhipment_port=anchorage,
                    daughter_class=best_daughter,
                    daughter_voyages=daughter_voyages,
                    vessels_required=vessels_required,
                    total_lifted_t=mother_lift * vessels_required,
                    detail={**result.as_dict(), "daughter_lift_t": best_daughter_lift},
                )
            )

    return options


def apply_load_port_limit(
    options: list[RoutingOption],
    load_port: str,
    cargo: Cargo | None,
    bunkers_rob_t: float,
    required_tonnes: float,
) -> list[RoutingOption]:
    """Cap each option by what the load port can actually put on board.

    A vessel cannot discharge more than it loaded. Screening only the
    discharge side overstates lift on any lane out of a draft-restricted
    origin, so the binding figure is the lower of the two ends.
    """
    classes = load_vessel_classes()
    load_berths = berths_for_port(load_port)
    if not load_berths:
        return options

    capped: list[RoutingOption] = []
    for option in options:
        vessel = classes.get(option.vessel_class)
        if vessel is None:
            capped.append(option)
            continue

        _, load_result = _best_berth(vessel, load_berths, cargo, bunkers_rob_t, 0.0)
        if load_result is None or not load_result.feasible:
            option.feasible = False
            option.explanation = (
                f"{option.vessel_class} cannot load at {load_port}: "
                f"{load_result.explanation if load_result else 'no suitable berth'}"
            )
            option.loadable_tonnes = 0.0
            capped.append(option)
            continue

        discharge_detail = option.detail
        if load_result.loadable_tonnes < option.loadable_tonnes:
            option.explanation = (
                f"{option.explanation} Capped by the load port: {load_port} permits "
                f"{load_result.loadable_tonnes:,.0f} t against "
                f"{option.loadable_tonnes:,.0f} t at the discharge end."
            )
            option.loadable_tonnes = load_result.loadable_tonnes
            option.limiting_factor = f"load_port_{load_result.limiting_factor.value}"
            # The displayed workings must belong to whichever end actually
            # binds, otherwise the arithmetic shown will not reconcile with
            # the tonnage above it.
            option.detail = {
                **load_result.as_dict(),
                "binding_end": "load_port",
                "binding_port": load_port,
                "discharge_side": discharge_detail,
            }

        if option.loadable_tonnes > 0:
            option.vessels_required = max(
                1, math.ceil(required_tonnes / option.loadable_tonnes)
            )
            option.total_lifted_t = option.loadable_tonnes * option.vessels_required
            if option.routing == "lightering" and option.daughter_class:
                daughter_lift = option.detail.get("daughter_lift_t")
                if daughter_lift:
                    option.daughter_voyages = max(
                        1, math.ceil(option.loadable_tonnes / daughter_lift)
                    )
        capped.append(option)

    return capped


def screen_all(
    discharge_port: str,
    required_tonnes: float,
    *,
    load_port: str | None = None,
    cargo_name: str = "thermal_coal_mid_cv",
    bunkers_rob_t: float = 1200.0,
    tide_height_m: float = 0.0,
) -> dict:
    """Full screening for one requirement: direct and lightering, ranked.

    When ``load_port`` is supplied the lift is capped by whichever end binds,
    which is the honest answer for a real voyage.

    Feasible options are ranked by how few vessels are needed, then by lift
    per vessel. Infeasible options are retained and returned separately,
    because *why* a vessel was rejected is often the most useful output.
    """
    direct = screen_direct(
        discharge_port,
        required_tonnes,
        cargo_name=cargo_name,
        bunkers_rob_t=bunkers_rob_t,
        tide_height_m=tide_height_m,
    )
    lightering = screen_lightering(
        discharge_port,
        required_tonnes,
        cargo_name=cargo_name,
        bunkers_rob_t=bunkers_rob_t,
        tide_height_m=tide_height_m,
    )

    options = direct + lightering
    if load_port:
        cargo = load_cargoes().get(cargo_name)
        options = apply_load_port_limit(
            options, load_port, cargo, bunkers_rob_t, required_tonnes
        )

    feasible = [o for o in options if o.feasible]
    rejected = [o for o in options if not o.feasible and o.routing == "direct"]

    feasible.sort(key=lambda o: (o.vessels_required, -o.loadable_tonnes))

    return {
        "discharge_port": discharge_port,
        "load_port": load_port,
        "required_tonnes": required_tonnes,
        "cargo": cargo_name,
        "feasible": [o.as_dict() for o in feasible],
        "rejected": [o.as_dict() for o in rejected],
        "summary": _summarise(feasible, rejected, discharge_port, required_tonnes),
    }


def _summarise(
    feasible: list[RoutingOption],
    rejected: list[RoutingOption],
    discharge_port: str,
    required_tonnes: float,
) -> str:
    if not feasible:
        return (
            f"No feasible routing found for {required_tonnes:,.0f} t into "
            f"{discharge_port}. {len(rejected)} vessel classes were screened and "
            f"all were rejected."
        )

    best = feasible[0]
    text = (
        f"Best routing for {required_tonnes:,.0f} t into {discharge_port}: "
        f"{best.vessels_required} x {best.vessel_class} "
        f"({best.loadable_tonnes:,.0f} t each) via {best.routing}."
    )
    if rejected:
        feasible_classes = {o.vessel_class for o in feasible}
        blocked = sorted({o.vessel_class for o in rejected} - feasible_classes)
        via_other = sorted({o.vessel_class for o in rejected} & feasible_classes)
        if blocked:
            text += f" Not possible at all: {', '.join(blocked)}."
        if via_other:
            text += (
                f" Cannot discharge directly, but feasible by transhipment: "
                f"{', '.join(via_other)}."
            )
    return text
