"""Model output: forecasts, backtest metrics and alerts.

These tables deliberately do **not** carry ``ProvenanceMixin``.

A forecast is not an observation. It is a claim about the future derived from
inputs, and flagging it ``is_synthetic=False`` would imply it was fetched from
a publisher, while flagging it ``is_synthetic=True`` would trip the disclosure
guard in ``app.ingest.base`` and conflate a model with a fabricated feed.
Both labels are wrong, so model output gets its own provenance shape:
``model_name``, ``model_version``, ``trained_through`` and the list of input
series it was fitted on.

Two consequences follow, and both are intentional:

  * ``/api/meta/provenance-summary`` and the "% real" badge count ingested
    observations only. A forecast can neither inflate nor dilute that number.
  * A forecast trained on a disclosed synthetic series inherits that taint.
    ``ForecastRun.input_series`` records exactly what it was fitted on, and
    ``ForecastRun.trained_on_synthetic`` makes the check one column lookup so
    the UI can badge it without parsing anything.

See the "Three kinds of number" section of SYNTHETIC_DATA.md.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utcnow


class ForecastRun(Base):
    """One fit of one model against one target, at one point in time."""

    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(60), index=True)
    model_version: Mapped[str] = mapped_column(String(20))
    target_key: Mapped[str] = mapped_column(String(60), index=True)
    run_date: Mapped[date] = mapped_column(Date, index=True)
    # Last observation the model was allowed to see. Everything after this is
    # out of sample by construction.
    trained_through: Mapped[date] = mapped_column(Date)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    input_series: Mapped[str] = mapped_column(Text, default="")  # comma separated
    trained_on_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="ok")  # ok | failed | skipped
    message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ForecastPoint(Base):
    """One quantile of one horizon of one run.

    Long form rather than q05..q95 columns, so the quantile grid can change
    without a migration.
    """

    __tablename__ = "forecast_points"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "horizon_days", "quantile", name="uq_forecast_point"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, index=True)
    target_key: Mapped[str] = mapped_column(String(60), index=True)
    origin_date: Mapped[date] = mapped_column(Date, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    quantile: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float)


class BacktestMetric(Base):
    """Walk-forward performance of one model at one horizon.

    ``skill`` is ``1 - loss / baseline_loss``: positive means the model beat
    the baseline, zero or negative means it did not and the platform must say
    so rather than emitting a confident number.
    """

    __tablename__ = "backtest_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    target_key: Mapped[str] = mapped_column(String(60), index=True)
    model_name: Mapped[str] = mapped_column(String(60), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True)
    metric: Mapped[str] = mapped_column(String(30), index=True)
    value: Mapped[float] = mapped_column(Float)
    baseline_name: Mapped[str | None] = mapped_column(String(40), nullable=True)
    baseline_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    skill: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_folds: Mapped[int] = mapped_column(Integer, default=0)
    computed_on_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    """One risk warning, raised and later cleared.

    Alerts are persisted rather than computed on request, so that an alert
    which fired and later cleared is a testable prediction. After a season of
    operation the precision of each rule is measurable from this table, which
    is the only honest way to decide whether a rule earns its place.
    """

    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("code", "scope", "raised_date", name="uq_alert_instance"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), index=True)
    # info | watch | warning | critical
    severity: Mapped[str] = mapped_column(String(12), index=True)
    # The thing the alert is about: a port code, a series code, or an adapter.
    scope: Mapped[str] = mapped_column(String(60), index=True, default="")
    port_code: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    series_code: Mapped[str | None] = mapped_column(String(60), nullable=True)

    raised_date: Mapped[date] = mapped_column(Date, index=True)
    cleared_date: Mapped[date | None] = mapped_column(Date, index=True, nullable=True)
    last_seen_date: Mapped[date] = mapped_column(Date, index=True)

    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    # What this implies for a chartering decision. The reason the alert exists.
    action: Mapped[str] = mapped_column(Text, default="")
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
