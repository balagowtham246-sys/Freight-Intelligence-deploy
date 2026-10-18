"""Ingestion adapters, one per external source.

``ADAPTERS`` is the registry the scheduler and the backfill script iterate.
``BACKFILLABLE`` lists the keys whose sources retain history that can be
walked date by date.
"""
from __future__ import annotations

from .base import DisclosureError, IngestResult, SourceAdapter, assert_disclosed, upsert
from .cea_coal import CeaCoalStockAdapter
from .ibtracs import IbtracsCycloneAdapter
from .paradip import ParadipBerthingAdapter
from .sse_cdfi import SseCdfiAdapter
from .synthetic_history import SyntheticFreightHistoryAdapter
from .worldbank import WorldBankPinkSheetAdapter

ADAPTERS: dict[str, type[SourceAdapter]] = {
    "paradip": ParadipBerthingAdapter,
    "cea_coal": CeaCoalStockAdapter,
    "worldbank": WorldBankPinkSheetAdapter,
    "sse_cdfi": SseCdfiAdapter,
    "ibtracs": IbtracsCycloneAdapter,
    "synthetic_history": SyntheticFreightHistoryAdapter,
}

BACKFILLABLE = {"paradip", "cea_coal"}

# Adapters that derive from what is already held rather than fetching, and so
# must run after the feeds they depend on.
DERIVED_LAST = ("synthetic_history",)

__all__ = [
    "ADAPTERS",
    "BACKFILLABLE",
    "DERIVED_LAST",
    "SourceAdapter",
    "IngestResult",
    "DisclosureError",
    "assert_disclosed",
    "upsert",
    "ParadipBerthingAdapter",
    "CeaCoalStockAdapter",
    "WorldBankPinkSheetAdapter",
    "SseCdfiAdapter",
    "IbtracsCycloneAdapter",
    "SyntheticFreightHistoryAdapter",
]
