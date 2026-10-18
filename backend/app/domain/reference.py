"""Loads the port and vessel reference data from YAML into usable objects.

The YAML files under ``data/reference`` are the editable source of truth for
constraints. They are loaded into memory here and, separately, mirrored into
the database so the API can join them against observed data.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from ..config import REFERENCE_DIR
from .loadable_tonnes import SALT_DENSITY, BerthLimits, Cargo, Vessel

PORTS_FILE = REFERENCE_DIR / "port_constraints.yaml"
VESSELS_FILE = REFERENCE_DIR / "vessel_classes.yaml"


@lru_cache(maxsize=1)
def _ports_document() -> dict:
    with PORTS_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def _vessels_document() -> dict:
    with VESSELS_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def load_ports() -> dict[str, dict]:
    """Port records keyed by code, each with its berth list."""
    return {port["code"]: port for port in _ports_document()["ports"]}


@lru_cache(maxsize=1)
def load_vessel_classes() -> dict[str, Vessel]:
    """Vessel archetypes keyed by class name."""
    classes: dict[str, Vessel] = {}
    for record in _vessels_document()["classes"]:
        classes[record["name"]] = Vessel(
            name=record["name"],
            dwt=float(record["dwt_typical"]),
            loa_m=float(record["loa_m"]),
            beam_m=float(record["beam_m"]),
            summer_draft_m=float(record["summer_draft_m"]),
            lightship_t=float(record["lightship_t"]),
            tpc=float(record["tpc"]),
            grain_cubic_m3=float(record["grain_cubic_m3"]),
            geared=bool(record.get("geared", False)),
            constants_t=float(record.get("constants_t", 500)),
        )
    return classes


@lru_cache(maxsize=1)
def load_cargoes() -> dict[str, Cargo]:
    cargoes: dict[str, Cargo] = {}
    for record in _vessels_document().get("cargoes", []):
        cargoes[record["name"]] = Cargo(
            name=record["name"],
            stowage_factor_m3_per_t=float(record["stowage_factor_m3_per_t"]),
            gcv_kcal_per_kg=float(record.get("gcv_kcal_per_kg", 0)),
            category=record.get("category"),
        )
    return cargoes


@lru_cache(maxsize=1)
def load_berths() -> list[BerthLimits]:
    """Every berth across every port, as engine-ready constraint objects."""
    berths: list[BerthLimits] = []
    for port in _ports_document()["ports"]:
        density = float(port.get("water_density", SALT_DENSITY))
        tide_constrained = bool(port.get("tide_constrained", False))

        for berth in port.get("berths", []):
            cargo_types = tuple(
                item.strip()
                for item in str(berth.get("cargo_types", "")).split(",")
                if item.strip()
            )
            berths.append(
                BerthLimits(
                    port_code=port["code"],
                    berth_name=berth["name"],
                    max_draft_m=float(berth["max_draft_m"]),
                    water_density=density,
                    max_loa_m=_optional_float(berth.get("max_loa_m")),
                    max_beam_m=_optional_float(berth.get("max_beam_m")),
                    max_dwt=_optional_float(berth.get("max_dwt")),
                    max_air_draft_m=_optional_float(berth.get("max_air_draft_m")),
                    requires_gear=bool(berth.get("requires_gear", False)),
                    cargo_types=cargo_types,
                    discharge_rate_tpd=_optional_float(berth.get("discharge_rate_tpd")),
                    tide_constrained=tide_constrained,
                )
            )
    return berths


def berths_for_port(port_code: str) -> list[BerthLimits]:
    return [berth for berth in load_berths() if berth.port_code == port_code]


def discharge_ports() -> list[dict]:
    return [
        port
        for port in _ports_document()["ports"]
        if port.get("role") in {"discharge", "transhipment"}
    ]


def load_ports_list() -> list[dict]:
    return list(_ports_document()["ports"])


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
