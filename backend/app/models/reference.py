"""Reference entities: ports, berths, vessel classes. Mostly loaded from YAML."""
from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, ProvenanceMixin


class Port(Base, ProvenanceMixin):
    __tablename__ = "ports"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)  # UN/LOCODE-ish
    name: Mapped[str] = mapped_column(String(80))
    country: Mapped[str] = mapped_column(String(4))
    role: Mapped[str] = mapped_column(String(16))  # "load" | "discharge" | "transhipment"
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)

    # Water density at the berth in t/m3 -- the Hooghly is brackish, the
    # Mississippi fresh, most others full seawater. Drives the FWA correction.
    water_density: Mapped[float] = mapped_column(Float, default=1.025)

    # Mean spring tidal range (m); used with the tide model for the window search.
    tidal_range_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Whether the port can only be worked on a tide (true for Haldia/Hooghly).
    tide_constrained: Mapped[bool] = mapped_column(Boolean, default=False)

    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

    berths: Mapped[list["Berth"]] = relationship(back_populates="port", cascade="all, delete-orphan")


class Berth(Base, ProvenanceMixin):
    __tablename__ = "berths"

    id: Mapped[int] = mapped_column(primary_key=True)
    port_id: Mapped[int] = mapped_column(ForeignKey("ports.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))

    max_draft_m: Mapped[float] = mapped_column(Float)
    max_loa_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_beam_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_dwt: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_air_draft_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    requires_gear: Mapped[bool] = mapped_column(Boolean, default=False)
    discharge_rate_tpd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cargo_types: Mapped[str] = mapped_column(String(200), default="thermal_coal")

    # Under-keel clearance policy as a fraction of draft (port/pilotage rule).
    ukc_policy_pct: Mapped[float] = mapped_column(Float, default=0.10)

    port: Mapped[Port] = relationship(back_populates="berths")


class VesselClass(Base, ProvenanceMixin):
    __tablename__ = "vessel_classes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(40), unique=True, index=True)

    dwt_typical: Mapped[int] = mapped_column(Integer)
    dwt_min: Mapped[int] = mapped_column(Integer)
    dwt_max: Mapped[int] = mapped_column(Integer)

    loa_m: Mapped[float] = mapped_column(Float)
    beam_m: Mapped[float] = mapped_column(Float)
    summer_draft_m: Mapped[float] = mapped_column(Float)
    lightship_t: Mapped[float] = mapped_column(Float)
    tpc: Mapped[float] = mapped_column(Float)  # tonnes per cm immersion

    geared: Mapped[bool] = mapped_column(Boolean, default=False)
    service_speed_kn: Mapped[float] = mapped_column(Float, default=12.5)
    # Daily fuel burn at service speed, laden (t/day VLSFO).
    consumption_laden_tpd: Mapped[float] = mapped_column(Float, default=32.0)
    consumption_ballast_tpd: Mapped[float] = mapped_column(Float, default=28.0)
    consumption_port_tpd: Mapped[float] = mapped_column(Float, default=5.0)
