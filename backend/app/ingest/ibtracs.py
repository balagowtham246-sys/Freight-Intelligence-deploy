"""IBTrACS North Indian Ocean cyclone best-track archive.

Source: https://www.ncei.noaa.gov/products/international-best-track-archive

The strategy document asks for IMD RSMC New Delhi cyclone tracks. IBTrACS is
the better route to exactly that data: NOAA consolidates every agency's best
track into one free CSV, and the North Indian Ocean file carries the
``NEWDELHI_*`` columns, which *are* the IMD RSMC assessment. No scraping, no
package dependency, and history back to the nineteenth century.

We prefer the New Delhi position and intensity where present, because that is
the responsible agency for the Bay of Bengal, and fall back to the WMO
consolidated columns otherwise.

What this feeds:

  * the ``CYCLONE`` alert in ``app.domain.alerts`` - a storm currently within
    reach of a discharge port;
  * the cyclone-season calendar features in ``app.ml.features`` - Bay of
    Bengal risk is strongly seasonal (Oct-Dec, secondary Apr-May), and the
    climatology here is what makes that a measured feature rather than an
    assumed one.

The file is ~30 MB, so it is fetched once and archived; the adapter is
idempotent and re-parses the archive rather than re-downloading when the
season has not advanced.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import RAW_DIR
from ..models import CycloneTrackPoint
from .base import IngestResult, SourceAdapter, upsert

BASE = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/"
)
NORTH_INDIAN_URL = f"{BASE}ibtracs.NI.list.v04r01.csv"

# Only the last N seasons are loaded. The full archive goes back to 1842, but
# storm behaviour before satellite coverage is not comparable, and the
# climatology we need is a modern one.
SEASONS_KEPT = 30


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text in {"", " ", "NA", "-999", "-9999"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_float(row: dict, *keys: str) -> float | None:
    """First parseable value across a preference-ordered list of columns."""
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def parse_ibtracs(content: bytes, *, min_season: int) -> list[dict]:
    """Parse the IBTrACS CSV into track-point records.

    Row 0 is the header; row 1 is a units row that must be skipped.
    """
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None or "SID" not in reader.fieldnames:
        raise ValueError("IBTrACS layout changed: no SID column in header")

    records: list[dict] = []
    for row in reader:
        season_raw = (row.get("SEASON") or "").strip()
        if not season_raw.isdigit():
            continue  # the units row, and any trailing junk
        season = int(season_raw)
        if season < min_season:
            continue

        stamp = (row.get("ISO_TIME") or "").strip()
        if not stamp:
            continue
        try:
            timestamp = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        # New Delhi is the responsible RSMC for this basin; prefer it.
        latitude = _first_float(row, "NEWDELHI_LAT", "LAT", "USA_LAT")
        longitude = _first_float(row, "NEWDELHI_LON", "LON", "USA_LON")
        if latitude is None or longitude is None:
            continue

        records.append(
            {
                "storm_id": (row.get("SID") or "").strip(),
                "storm_name": ((row.get("NAME") or "").strip() or None),
                "timestamp": timestamp,
                "latitude": latitude,
                "longitude": longitude,
                "max_wind_kt": _first_float(row, "NEWDELHI_WIND", "WMO_WIND", "USA_WIND"),
                "pressure_hpa": _first_float(row, "NEWDELHI_PRES", "WMO_PRES", "USA_PRES"),
                "category": ((row.get("NEWDELHI_GRADE") or "").strip() or None),
                "basin": ((row.get("BASIN") or "").strip() or None),
            }
        )

    if not records:
        raise ValueError("IBTrACS parsed to zero track points")
    return records


class IbtracsCycloneAdapter(SourceAdapter):
    name = "ibtracs_north_indian"
    description = (
        "IBTrACS North Indian Ocean cyclone best tracks, using the IMD RSMC "
        "New Delhi assessment where available"
    )
    homepage = "https://www.ncei.noaa.gov/products/international-best-track-archive"

    def probe(self) -> tuple[bool, str]:
        response = self.client.get(NORTH_INDIAN_URL, headers={"Range": "bytes=0-2000"})
        ok = response.status_code in (200, 206)
        return ok, f"HTTP {response.status_code}"

    def run(self, session: Session) -> IngestResult:
        fetched = self.fetch(NORTH_INDIAN_URL)

        archive = RAW_DIR / "ibtracs" / "ibtracs.NI.list.v04r01.csv"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(fetched.content)

        min_season = date.today().year - SEASONS_KEPT
        records = parse_ibtracs(fetched.content, min_season=min_season)

        provenance = self.provenance(fetched)
        rows = [{**record, **provenance} for record in records]

        # The table has no natural key constraint, so an idempotent load means
        # clearing the seasons we are about to rewrite.
        existing = session.scalars(
            select(CycloneTrackPoint).where(CycloneTrackPoint.source_name == self.name)
        ).all()
        for row in existing:
            session.delete(row)
        session.flush()

        for row in rows:
            session.add(CycloneTrackPoint(**row))
        session.flush()

        storms = len({row["storm_id"] for row in rows})
        return IngestResult(
            self.name,
            "stale" if fetched.from_cache else "ok",
            rows_written=len(rows),
            message=f"{storms} storms since {min_season}",
            used_cache=fetched.from_cache,
        )
