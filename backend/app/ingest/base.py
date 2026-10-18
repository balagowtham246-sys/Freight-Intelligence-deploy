"""Adapter framework for every external feed.

Three rules are enforced here rather than left to each adapter:

1. **Provenance is mandatory.** Every row written carries a source name, the
   URL it came from, when it was fetched, and whether it is synthetic.
2. **Synthetic data must be disclosed.** A row flagged ``is_synthetic`` cannot
   be written unless its source name appears in SYNTHETIC_DATA.md. This is a
   hard failure, not a warning, so the disclosure file cannot silently drift
   out of date.
3. **Feeds degrade gracefully.** A fetch failure falls back to the last
   known-good payload on disk and the run is recorded as ``stale`` rather than
   taking the platform down.
"""
from __future__ import annotations

import abc
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from ..config import REPO_ROOT, settings
from ..models import IngestRun

CACHE_DIR = Path(settings.database_url.split("///")[-1]).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_DISCLOSURE_FILE = REPO_ROOT / "SYNTHETIC_DATA.md"


class DisclosureError(RuntimeError):
    """Raised when synthetic data would be written without a disclosure entry."""


def _disclosed_source_names() -> set[str]:
    """Parse the source identifiers declared in SYNTHETIC_DATA.md.

    Entries are declared with a line of the form::

        - `source_name: some_identifier`
    """
    if not SYNTHETIC_DISCLOSURE_FILE.exists():
        return set()
    text = SYNTHETIC_DISCLOSURE_FILE.read_text(encoding="utf-8")
    return set(re.findall(r"`source_name:\s*([A-Za-z0-9_.\-]+)`", text))


def assert_disclosed(source_name: str) -> None:
    """Fail loudly if a synthetic source has not been documented."""
    if not settings.enforce_synthetic_disclosure:
        return
    if source_name not in _disclosed_source_names():
        raise DisclosureError(
            f"Synthetic source {source_name!r} is not declared in SYNTHETIC_DATA.md. "
            f"Add an entry containing the line: `source_name: {source_name}` "
            f"before writing synthetic rows."
        )


@dataclass
class FetchResult:
    """Payload plus the provenance needed to attribute every derived row."""

    content: bytes
    url: str
    retrieved_at: datetime
    from_cache: bool = False


@dataclass
class IngestResult:
    adapter: str
    status: str  # ok | failed | stale | skipped
    rows_written: int = 0
    message: str | None = None
    used_cache: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(abc.ABC):
    """Base class for one external data source.

    Subclasses declare their identity and implement :meth:`run`. Fetching goes
    through :meth:`fetch` so caching and failure handling stay uniform.
    """

    #: Stable identifier written to every row this adapter produces.
    name: str
    #: Human-readable description shown in the data-provenance UI.
    description: str = ""
    #: Landing page or API root, for the sources table and UI attribution.
    homepage: str = ""
    #: True only for adapters that fabricate data. Forces a disclosure check.
    produces_synthetic: bool = False

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------ http

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=settings.http_timeout_seconds,
                headers={"User-Agent": settings.user_agent},
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        return CACHE_DIR / f"{self.name}__{digest}.bin"

    def fetch(self, url: str, *, allow_cache: bool = True) -> FetchResult:
        """GET a URL, falling back to the last known-good copy on failure."""
        cache_path = self._cache_path(url)
        try:
            response = self.client.get(url)
            response.raise_for_status()
            content = response.content
            if not content:
                raise ValueError(f"empty response body from {url}")
            if allow_cache:
                cache_path.write_bytes(content)
            return FetchResult(content, url, datetime.now(timezone.utc), from_cache=False)
        except Exception as exc:
            if allow_cache and cache_path.exists():
                stamp = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
                return FetchResult(cache_path.read_bytes(), url, stamp, from_cache=True)
            raise RuntimeError(f"{self.name}: fetch failed for {url} with no cache: {exc}") from exc

    # ------------------------------------------------------------- provenance

    def provenance(self, result: FetchResult | None = None, *, url: str | None = None) -> dict:
        """Provenance kwargs to splat into any model carrying ProvenanceMixin."""
        if self.produces_synthetic:
            assert_disclosed(self.name)
        return {
            "source_name": self.name,
            "source_url": url or (result.url if result else self.homepage or None),
            "retrieved_at": result.retrieved_at if result else datetime.now(timezone.utc),
            "is_synthetic": self.produces_synthetic,
        }

    # ---------------------------------------------------------------- running

    @abc.abstractmethod
    def run(self, session: Session) -> IngestResult:
        """Fetch, parse and persist. Must be idempotent (upsert, not append)."""

    def execute(self, session: Session) -> IngestResult:
        """Run the adapter and record the attempt in the ingest_runs audit log."""
        started = datetime.now(timezone.utc)
        try:
            result = self.run(session)
        except Exception as exc:
            result = IngestResult(self.name, "failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            self.close()

        session.add(
            IngestRun(
                adapter=self.name,
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                status=result.status,
                rows_written=result.rows_written,
                message=(result.message or "")[:1000] or None,
                used_cache=result.used_cache,
            )
        )
        session.commit()
        return result


def upsert(session: Session, model, rows: list[dict], conflict_keys: list[str]) -> int:
    """Insert rows, replacing any existing record matching ``conflict_keys``.

    Written against SQLAlchemy core so it works on SQLite now and Postgres later.
    Returns the number of rows written.
    """
    if not rows:
        return 0

    written = 0
    for row in rows:
        criteria = {key: row[key] for key in conflict_keys}
        existing = session.query(model).filter_by(**criteria).one_or_none()
        if existing is None:
            session.add(model(**row))
        else:
            for key, value in row.items():
                setattr(existing, key, value)
        written += 1

    session.flush()
    return written


def dump_json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
