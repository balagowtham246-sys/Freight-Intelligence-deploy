"""World Bank Pink Sheet monthly commodity prices.

Source: https://www.worldbank.org/en/research/commodity-markets

Free, and the deepest history available anywhere for coal: monthly Australian
and South African thermal coal back to 1960, plus crude and iron ore. That
depth matters because the strategy document calls for the model to see at
least two full freight cycles.

The workbook URL is versioned with a rotating document id, so the adapter
discovers the current link from the landing page rather than pinning a URL
that will silently go stale.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from ..config import RAW_DIR
from ..models import RateObservation
from .base import FetchResult, IngestResult, SourceAdapter, upsert

LANDING_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WORKBOOK_RE = re.compile(
    r"https://thedocs\.worldbank\.org[^\"'\s]*CMO-Historical-Data-Monthly\.xlsx"
)

# Known-good fallback, used only if link discovery fails.
FALLBACK_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "18675f1d1639c7a34d463f59263ba0a2-0050012025/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)

# Pink Sheet column label -> (our series code, unit). Labels are matched
# case-insensitively on a normalised form, because the sheet carries footnote
# markers such as "Coal, South African **".
SERIES_MAP = {
    "coal, australian": ("COAL_AU_NEWC", "usd_per_t"),
    "coal, south african": ("COAL_ZA_RB", "usd_per_t"),
    "coal, colombian": ("COAL_CO", "usd_per_t"),
    "crude oil, brent": ("CRUDE_BRENT", "usd_per_bbl"),
    "crude oil, average": ("CRUDE_AVG", "usd_per_bbl"),
    "iron ore, cfr spot": ("IRON_ORE_CFR", "usd_per_dmtu"),
    "natural gas, europe": ("GAS_EU", "usd_per_mmbtu"),
}

PERIOD_RE = re.compile(r"^(\d{4})M(\d{1,2})$")


def _normalise(label: str) -> str:
    """Strip footnote markers and whitespace so labels match reliably."""
    return re.sub(r"[*\s]+$", "", str(label).strip()).strip().lower()


def parse_pinksheet(content: bytes) -> list[dict]:
    """Parse the Monthly Prices sheet into observation records."""
    frame = pd.read_excel(io.BytesIO(content), sheet_name="Monthly Prices", header=None)

    header_row = 4
    labels = frame.iloc[header_row].tolist()

    wanted: dict[int, tuple[str, str]] = {}
    for index, label in enumerate(labels):
        key = _normalise(label)
        if key in SERIES_MAP:
            wanted[index] = SERIES_MAP[key]

    if not wanted:
        raise ValueError(
            "Pink Sheet layout changed: none of the expected commodity columns were found"
        )

    records: list[dict] = []
    for _, row in frame.iloc[header_row + 2 :].iterrows():
        period = PERIOD_RE.match(str(row[0]).strip())
        if not period:
            continue
        year, month = int(period.group(1)), int(period.group(2))
        if not 1 <= month <= 12:
            continue
        obs_date = date(year, month, 1)

        for column, (series_code, unit) in wanted.items():
            raw = row[column]
            try:
                value = float(str(raw).replace(",", ""))
            except (TypeError, ValueError):
                continue  # the sheet uses ".." and similar for missing months
            if value != value:  # NaN
                continue

            records.append(
                {
                    "series_code": series_code,
                    "series_kind": "commodity",
                    "obs_date": obs_date,
                    "value": value,
                    "unit": unit,
                }
            )

    if not records:
        raise ValueError("Pink Sheet parsed to zero observations")
    return records


class WorldBankPinkSheetAdapter(SourceAdapter):
    name = "worldbank_pinksheet"
    description = "World Bank Pink Sheet monthly commodity prices (coal, crude, iron ore)"
    homepage = LANDING_PAGE

    def discover_url(self) -> str:
        """Find the current workbook link from the landing page."""
        try:
            response = self.client.get(LANDING_PAGE)
            response.raise_for_status()
            match = WORKBOOK_RE.search(response.text)
            if match:
                return match.group(0)
        except Exception:
            pass
        return FALLBACK_URL

    def probe(self) -> tuple[bool, str]:
        archive = RAW_DIR / "worldbank" / "CMO-Historical-Data-Monthly.xlsx"
        url = self.discover_url()
        try:
            response = self.client.head(url)
            ok = response.status_code == 200
            discovered = "discovered" if url != FALLBACK_URL else "fallback"
            return ok, f"{discovered} url, HTTP {response.status_code}"
        except Exception:
            if archive.exists():
                return True, f"archive available on disk ({archive.name})"
            raise

    def run(self, session: Session) -> IngestResult:
        archive = RAW_DIR / "worldbank" / "CMO-Historical-Data-Monthly.xlsx"
        url = self.discover_url()
        try:
            fetched = self.fetch(url)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(fetched.content)
        except Exception:
            if archive.exists():
                stamp = datetime.fromtimestamp(archive.stat().st_mtime, tz=timezone.utc)
                fetched = FetchResult(archive.read_bytes(), url, stamp, from_cache=True)
                cache_path = self._cache_path(url)
                if not cache_path.exists():
                    cache_path.write_bytes(fetched.content)
            else:
                raise

        provenance = self.provenance(fetched)
        rows = [{**record, **provenance} for record in parse_pinksheet(fetched.content)]
        written = upsert(session, RateObservation, rows, ["series_code", "obs_date"])

        latest = max(row["obs_date"] for row in rows)
        series_count = len({row["series_code"] for row in rows})

        return IngestResult(
            self.name,
            "stale" if fetched.from_cache else "ok",
            rows_written=written,
            message=f"{series_count} series through {latest.isoformat()}",
            used_cache=fetched.from_cache,
        )
