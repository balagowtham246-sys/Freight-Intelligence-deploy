"""Time-series and event tables: rates, port calls, tides, cyclones, fixtures."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, ProvenanceMixin


class RateObservation(Base, ProvenanceMixin):
    """One observation of one freight/commodity/bunker series on one day.

    Deliberately generic: BDI, BSI 58 TC, NEWC coal, Singapore VLSFO and the
    route-basis series all live here, separated by ``series_code``.
    """

    __tablename__ = "rate_observations"
    __table_args__ = (UniqueConstraint("series_code", "obs_date", name="uq_series_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    series_code: Mapped[str] = mapped_column(String(40), index=True)
    # freight_index | tc_avg | commodity | bunker | route_basis | ffa
    series_kind: Mapped[str] = mapped_column(String(24), index=True)
    obs_date: Mapped[date] = mapped_column(Date, index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20))


class CoalStockObservation(Base, ProvenanceMixin):
    """CEA daily coal stock at Indian thermal plants: the demand-urgency proxy."""

    __tablename__ = "coal_stock_observations"
    __table_args__ = (UniqueConstraint("plant_name", "obs_date", name="uq_plant_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    obs_date: Mapped[date] = mapped_column(Date, index=True)
    plant_name: Mapped[str] = mapped_column(String(120), index=True)
    state: Mapped[str | None] = mapped_column(String(60), nullable=True)
    region: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    capacity_mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    stock_kt: Mapped[float | None] = mapped_column(Float, nullable=True)
    normative_stock_kt: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_of_burn: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_import_based: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class PortCallSnapshot(Base, ProvenanceMixin):
    """One vessel line from one daily berthing programme.

    Scraped daily. The port PDFs use day+month filenames and are overwritten
    annually, so this archive is only ever as deep as the date we started it.
    """

    __tablename__ = "port_call_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "port_code", "snapshot_date", "vessel_name", "status", name="uq_snapshot_vessel"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    port_code: Mapped[str] = mapped_column(String(16), index=True)
    snapshot_date: Mapped[date] = mapped_column(Date, index=True)

    vessel_name: Mapped[str] = mapped_column(String(120), index=True)
    # working | waiting | expected | sailed
    status: Mapped[str] = mapped_column(String(20), index=True)
    berth: Mapped[str | None] = mapped_column(String(60), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(120), nullable=True)
    cargo: Mapped[str | None] = mapped_column(String(120), nullable=True)
    cargo_tonnes: Mapped[float | None] = mapped_column(Float, nullable=True)
    arrival_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    berthed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    draft_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    loa_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_row: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class WaitingTimeObservation(Base, ProvenanceMixin):
    """Derived: realised pre-berthing detention, reconstructed from snapshots.

    One row per *port call*, keyed on (port, vessel, arrival date) — a vessel
    makes several calls in a season and each is a separate waiting event.

    Censoring is the crux of this table. A call that has not berthed by the
    last snapshot we hold, or that vanished from the queue without ever being
    seen working, is recorded with ``censored=True`` and ``waiting_days`` as a
    *lower bound*. Dropping those rows, or treating them as berthed, biases
    every waiting-time estimate downward, and with a 73-day archive they are a
    large share of the sample.
    """

    __tablename__ = "waiting_time_observations"
    __table_args__ = (
        UniqueConstraint(
            "port_code", "vessel_name", "arrival_date", name="uq_waiting_call"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    port_code: Mapped[str] = mapped_column(String(16), index=True)
    vessel_name: Mapped[str] = mapped_column(String(120))
    arrival_date: Mapped[date] = mapped_column(Date, index=True)
    berthed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    waiting_days: Mapped[float] = mapped_column(Float)
    # True when the vessel was still waiting at the last snapshot we hold.
    censored: Mapped[bool] = mapped_column(Boolean, default=False)
    queue_len_on_arrival: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cargo: Mapped[str | None] = mapped_column(String(120), nullable=True)
    cargo_tonnes: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Days between the last "waiting" sighting and the first "working" sighting.
    # Zero means the berthing date is exact; larger values mean a gap in the
    # snapshot archive straddled the event and the date is an upper bound.
    berth_date_uncertain_days: Mapped[int] = mapped_column(Integer, default=0)


class TidePrediction(Base, ProvenanceMixin):
    __tablename__ = "tide_predictions"
    __table_args__ = (UniqueConstraint("port_code", "timestamp", name="uq_tide_port_ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    port_code: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    height_m: Mapped[float] = mapped_column(Float)  # above chart datum


class CycloneTrackPoint(Base, ProvenanceMixin):
    __tablename__ = "cyclone_track_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    storm_id: Mapped[str] = mapped_column(String(40), index=True)
    storm_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    max_wind_kt: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    basin: Mapped[str | None] = mapped_column(String(8), nullable=True)


class Fixture(Base, ProvenanceMixin):
    """A chartering fixture.

    Real ones if a chartering desk supplies them; otherwise the generated book
    documented in SYNTHETIC_DATA.md. The ``is_synthetic`` flag separates them.
    """

    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(primary_key=True)
    fixed_date: Mapped[date] = mapped_column(Date, index=True)
    load_port_code: Mapped[str] = mapped_column(String(16), index=True)
    discharge_port_code: Mapped[str] = mapped_column(String(16), index=True)
    vessel_class: Mapped[str] = mapped_column(String(40), index=True)
    cargo: Mapped[str] = mapped_column(String(60))
    cargo_tonnes: Mapped[float] = mapped_column(Float)
    freight_usd_per_t: Mapped[float] = mapped_column(Float)
    laycan_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    laycan_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    demurrage_usd_per_day: Mapped[float | None] = mapped_column(Float, nullable=True)
    contract_type: Mapped[str] = mapped_column(String(16), default="spot")  # spot | coa | tc


class AisPosition(Base, ProvenanceMixin):
    """Down-sampled AIS positions inside the Bay of Bengal bounding box."""

    __tablename__ = "ais_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    mmsi: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    sog_kn: Mapped[float | None] = mapped_column(Float, nullable=True)
    cog_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    draught_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    ship_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    ship_type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    destination: Mapped[str | None] = mapped_column(String(120), nullable=True)


class IngestRun(Base):
    """Audit log for every adapter execution; powers the data-freshness panel."""

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    adapter: Mapped[str] = mapped_column(String(60), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20))  # ok | failed | stale | skipped
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    used_cache: Mapped[bool] = mapped_column(Boolean, default=False)
