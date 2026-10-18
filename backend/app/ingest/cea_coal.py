"""CEA daily coal stock report for Indian thermal power plants.

Source: https://npp.gov.in/public-reports/cea/daily/fuel/DD-MM-YYYY/dailyCoalN-YYYY-MM-DD.xls

Published every day by the Central Electricity Authority, Fuel Management
Division. This is the best free proxy for import urgency that exists: when
stock days-of-burn at the coastal import-designed plants fall, buyers are
forced into the spot market regardless of freight levels, and East Coast
India rates firm.

The workbook is a JasperReports export with merged cells, so it is read
positionally rather than by header name. Column positions are documented in
``COLUMNS`` below and validated on every parse, so a layout change fails
loudly instead of silently loading nonsense.

Sections in the report:

    A. PLANTS HAVING COAL LINKAGES, NO LINKAGES AND COAL BLOCK
    B. PLANTS DESIGNED ON IMPORTED COAL      <- the import demand signal
    C. NOT IN OPERATION
    D. PLANTS BASED ON WASHERY REJECTS
"""
from __future__ import annotations

import io
import re
from datetime import date, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from ..config import RAW_DIR
from ..models import CoalStockObservation
from .base import IngestResult, SourceAdapter, upsert

# Positional column map for the JasperReports layout.
COLUMNS = {
    "serial": 0,
    "state": 1,
    "genco": 2,
    "transport": 4,
    "plant": 6,
    "capacity_mw": 8,
    "plf_pct": 9,
    "normative_days": 12,
    "daily_requirement_kt": 14,
    "normative_stock_kt": 17,
    "stock_indigenous_kt": 18,
    "stock_import_kt": 21,
    "stock_total_kt": 24,
    "pct_of_normative": 27,
    "receipt_kt": 31,
    "consumption_kt": 32,
}

SECTION_RE = re.compile(r"^\s*([A-D])\.\s+(.+)$")

# Eastern and southern coastal states served by East Coast India ports.
EAST_COAST_STATES = {
    "odisha", "orissa", "west bengal", "andhra pradesh", "tamil nadu",
    "telangana", "jharkhand", "bihar", "chhattisgarh",
}


def _to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if not text or text.lower() in {"nan", "-", "*"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _region_for(state: str | None) -> str | None:
    if not state:
        return None
    return "east_coast" if state.strip().lower() in EAST_COAST_STATES else "other"


def parse_workbook(content: bytes, obs_date: date) -> list[dict]:
    """Parse the CEA workbook into one record per plant."""
    frame = pd.read_excel(io.BytesIO(content), header=None)

    if frame.shape[1] <= max(COLUMNS.values()):
        raise ValueError(
            f"CEA layout changed: expected at least {max(COLUMNS.values()) + 1} "
            f"columns, found {frame.shape[1]}"
        )

    records: list[dict] = []
    section = "A"

    for _, row in frame.iterrows():
        first = str(row[COLUMNS["serial"]]).strip()

        header = SECTION_RE.match(first)
        if header:
            section = header.group(1).upper()
            continue

        # Plant rows are numbered; subtotal rows carry text such as "HPGCL-Total".
        if not first.isdigit():
            continue

        plant = str(row[COLUMNS["plant"]]).strip()
        if not plant or plant.lower() == "nan":
            continue

        state = str(row[COLUMNS["state"]]).strip()
        state = None if state.lower() == "nan" else state

        stock_total = _to_float(row[COLUMNS["stock_total_kt"]])
        daily_requirement = _to_float(row[COLUMNS["daily_requirement_kt"]])

        days_of_burn = None
        if stock_total is not None and daily_requirement and daily_requirement > 0:
            days_of_burn = stock_total / daily_requirement

        records.append(
            {
                "obs_date": obs_date,
                "plant_name": plant,
                "state": state,
                "region": _region_for(state),
                "capacity_mw": _to_float(row[COLUMNS["capacity_mw"]]),
                "stock_kt": stock_total,
                "normative_stock_kt": _to_float(row[COLUMNS["normative_stock_kt"]]),
                "days_of_burn": days_of_burn,
                # Section B plants are designed to run on imported coal, which
                # is the demand signal that matters for chartering.
                "is_import_based": section == "B",
            }
        )

    if not records:
        raise ValueError("CEA workbook parsed to zero plant rows; layout likely changed")

    return records


class CeaCoalStockAdapter(SourceAdapter):
    name = "cea_daily_coal_stock"
    description = "CEA daily coal stock at Indian thermal power plants"
    homepage = "https://npp.gov.in/publishedReports"

    lookback_days = 10

    def url_for(self, obs_date: date) -> str:
        return (
            f"https://npp.gov.in/public-reports/cea/daily/fuel/"
            f"{obs_date.strftime('%d-%m-%Y')}/"
            f"dailyCoal1-{obs_date.isoformat()}.xls"
        )

    def probe(self) -> tuple[bool, str]:
        """Reachability check for the dry-run mode of the backfill script."""
        for offset in range(self.lookback_days):
            target = date.today() - timedelta(days=offset)
            try:
                response = self.client.head(self.url_for(target))
                if response.status_code == 200:
                    return True, f"latest report {target.isoformat()}"
            except Exception:
                continue
        return False, f"no report found in last {self.lookback_days} days"

    def ingest_date(self, session: Session, obs_date: date) -> tuple[int, bool]:
        url = self.url_for(obs_date)
        fetched = self.fetch(url)

        archive = RAW_DIR / "cea" / f"dailyCoal_{obs_date.isoformat()}.xls"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(fetched.content)

        provenance = self.provenance(fetched)
        rows = [{**record, **provenance} for record in parse_workbook(fetched.content, obs_date)]
        written = upsert(session, CoalStockObservation, rows, ["plant_name", "obs_date"])
        return written, fetched.from_cache

    def run(self, session: Session) -> IngestResult:
        today = date.today()
        errors: list[str] = []

        for offset in range(self.lookback_days):
            obs_date = today - timedelta(days=offset)
            try:
                written, cached = self.ingest_date(session, obs_date)
            except Exception as exc:
                errors.append(f"{obs_date}: {type(exc).__name__}")
                continue

            if written:
                return IngestResult(
                    self.name,
                    "stale" if cached else "ok",
                    rows_written=written,
                    message=f"report {obs_date.isoformat()}",
                    used_cache=cached,
                )

        return IngestResult(
            self.name,
            "failed",
            message=f"no report in last {self.lookback_days} days ({'; '.join(errors[:3])})",
        )

    def backfill(self, session: Session, start: date, end: date) -> dict:
        """Walk a date range.

        Note: npp.gov.in retains only a short window of daily reports, so a
        long backfill will mostly miss. Depth here accumulates going forward.
        """
        found, missing, rows = 0, 0, 0
        cursor = start
        while cursor <= end:
            try:
                written, _ = self.ingest_date(session, cursor)
                if written:
                    found += 1
                    rows += written
                else:
                    missing += 1
            except Exception:
                missing += 1
            cursor += timedelta(days=1)
            session.commit()
        return {"reports": found, "missing": missing, "rows": rows}
