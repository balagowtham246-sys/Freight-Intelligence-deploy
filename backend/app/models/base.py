"""SQLAlchemy base plus the provenance mixin every ingested row must carry."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProvenanceMixin:
    """Attached to every table holding externally-sourced data.

    The `is_synthetic` flag is the backbone of the project's data-honesty
    commitment: it is enforced at write time (see app.ingest.base.record_rows)
    and surfaced through the API so the UI can badge every figure it renders.
    """

    source_name: Mapped[str] = mapped_column(String(80), index=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
