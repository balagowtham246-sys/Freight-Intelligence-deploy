"""Paradip Port Authority daily traffic update.

Source: https://www.paradipport.gov.in/uploads/YYYY/MM/dtrDDMM.pdf

The PDF carries four sections:

    A. WORKING VESSELS       -- currently at berth
    B. VESSELS WAITING AT ANCHORAGE
    C. EXPECTED VESSELS
    D. BERTHING MOVEMENTS

Sections B and C are the ones that drive congestion and waiting-time
modelling, and they extract as clean single-line records. Section A uses
stacked two-line table cells that pdfplumber interleaves, so it is parsed
on a looser anchor (vessel name plus the LOA/beam/draft triple) which is
sufficient to count berth occupancy.

Note on history: the newer ``/uploads/YYYY/MM/`` path is scoped by year and
month, so past reports remain retrievable and can be backfilled. The older
``/Writereaddata/Daily_Traffic/dtrDDMM.pdf`` path reused filenames annually
and is treated as best-effort only.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date, timedelta

import pdfplumber
from sqlalchemy.orm import Session

from ..config import RAW_DIR
from ..models import PortCallSnapshot
from .base import IngestResult, SourceAdapter, upsert
from .pdf_layout import Line, cluster_lines

PORT_CODE = "INPRT"

# "MV. STAR GALAXY", "MT. POROS BAY", occasionally with a trailing suffix.
VESSEL_TOKEN = r"M[VT]\.\s+[A-Z0-9][A-Z0-9 .&/'()-]*?"

# The dimension triple is the most reliable anchor in every section: three
# decimal numbers in a row, immediately after the vessel name.
DIMS = (
    r"(?P<loa>\d{1,3}\.\d{1,2})\s+"
    r"(?P<beam>\d{1,3}\.\d{1,2})\s+"
    r"(?P<draft>\d{1,3}\.\d{1,2})"
)

SECTION_RE = re.compile(r"^\s*([A-D])\.\s+(WORKING|VESSELS|EXPECTED|BERTHING)", re.I)


@dataclass
class ParsedRow:
    vessel_name: str
    status: str
    loa_m: float | None = None
    beam_m: float | None = None
    draft_m: float | None = None
    cargo: str | None = None
    cargo_tonnes: float | None = None
    berth: str | None = None
    agent: str | None = None
    arrival_date: date | None = None
    raw_row: str | None = None


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip(" .-")
    # Strip trailing operational annotations such as "(1HMC)" or "(2HMC)".
    name = re.sub(r"\s*\((?:\d?HMC|GL)\)\s*$", "", name, flags=re.I)
    return name.upper()


def _parse_daymonth(token: str, report_date: date) -> date | None:
    """Parse '29.08' or '29-08' into a date, choosing the nearest sane year."""
    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})$", token.strip())
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))
    for year in (report_date.year, report_date.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        # Arrivals precede the report; allow a small forward window for ETAs.
        if candidate <= report_date + timedelta(days=120):
            return candidate
    return None


def parse_waiting(text: str, report_date: date) -> list[ParsedRow]:
    """Section B: vessels waiting at anchorage.

    Layout:
        <n> MV. NAME <R|NR|PICT/R|...> LOA BEAM DRAFT CARGO <D|L> QTY
            SHIPPER dd.mm HHMM AGENT STEV PRIORITY REMARKS

    The berth-group headers ("FOR CB-1 & CB-2", "FOR OJ", ...) are carried
    down as the requested berth, and the READY / NOT READY sub-headers are
    kept because a vessel that is not ready is not truly queueing for a berth.
    """
    rows: list[ParsedRow] = []
    group: str | None = None
    ready_state: str | None = None

    pattern = re.compile(
        rf"^\s*\d+\s+({VESSEL_TOKEN})\s+"
        rf"(?:(?P<flag>[A-Z/]{{1,12}})\s+)?"
        rf"{DIMS}\s+"
        rf"(?P<cargo>[A-Z0-9./ ]+?)\s+(?P<dl>[DL])\s+(?P<qty>[\d,]+)\s+"
        rf"(?P<rest>.*)$"
    )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        upper = stripped.upper()
        if upper.startswith("FOR ") and len(stripped) < 40:
            group = stripped[4:].strip()
            continue
        if upper in {"READY", "NOT READY"}:
            ready_state = upper
            continue
        if upper.startswith("OTHERS"):
            group = "Others"
            continue

        match = pattern.match(stripped)
        if not match:
            continue

        rest = match.group("rest")
        arrival = None
        agent = None
        # Arrival stamp appears as "dd.mm HHMM" inside the tail.
        tail = re.search(r"(\d{1,2}[.\-]\d{1,2})\s+(\d{3,4})\s+(\S+)", rest)
        if tail:
            arrival = _parse_daymonth(tail.group(1), report_date)
            agent = tail.group(3)

        qty = match.group("qty").replace(",", "")
        rows.append(
            ParsedRow(
                vessel_name=_clean_name(match.group(1)),
                status="waiting",
                loa_m=float(match.group("loa")),
                beam_m=float(match.group("beam")),
                draft_m=float(match.group("draft")) or None,
                cargo=match.group("cargo").strip(),
                cargo_tonnes=float(qty) if qty.isdigit() else None,
                berth=group,
                agent=agent,
                arrival_date=arrival,
                raw_row=(f"[{ready_state or ''}] {stripped}")[:1000],
            )
        )
    return rows


def parse_expected(text: str, report_date: date) -> list[ParsedRow]:
    """Section C: expected vessels, grouped under 'DD-Mon-YYYY' headers."""
    rows: list[ParsedRow] = []
    eta_group: date | None = None

    header_re = re.compile(r"^\s*(\d{1,2})-([A-Za-z]{3})-(\d{4})\s*$")
    pattern = re.compile(
        rf"^\s*\d+\s+({VESSEL_TOKEN})\s+{DIMS}\s+"
        rf"(?P<cargo>[A-Z0-9./ ]+?)\s+(?P<dl>[DL])\s+(?P<qty>[\d,]+)\s+(?P<rest>.*)$"
    )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        header = header_re.match(stripped)
        if header:
            try:
                eta_group = date(
                    int(header.group(3)),
                    [
                        "jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec",
                    ].index(header.group(2).lower()) + 1,
                    int(header.group(1)),
                )
            except ValueError:
                eta_group = None
            continue

        match = pattern.match(stripped)
        if not match:
            continue

        qty = match.group("qty").replace(",", "")
        rows.append(
            ParsedRow(
                vessel_name=_clean_name(match.group(1)),
                status="expected",
                loa_m=float(match.group("loa")),
                beam_m=float(match.group("beam")),
                draft_m=float(match.group("draft")) or None,
                cargo=match.group("cargo").strip(),
                cargo_tonnes=float(qty) if qty.isdigit() else None,
                arrival_date=eta_group,
                raw_row=stripped[:1000],
            )
        )
    return rows


# Column spans for section A, measured from the rendered PDF (points).
COL_BERTH = (18.0, 46.0)
COL_NAME = (46.0, 186.0)
COL_DIMS = (186.0, 272.0)
COL_DATES = (272.0, 360.0)
COL_AGENT = (360.0, 388.0)
COL_DL = (388.0, 400.0)

BERTH_CODE_RE = re.compile(r"^(?:[A-Z]{2,5}|[A-Z]{1,3}-\d{1,2}|SPM-\d|CB-\d|FB-\d|EQ-\d|NQ-\d|CQ-\d)$")


def parse_working_lines(lines: list[Line]) -> list[ParsedRow]:
    """Section A: vessels at berth, parsed from word coordinates.

    Each logical record spans two rendered lines::

        <berth>  MV. NAME            times...  agent D/L  quantities   remarks
        <maxdrft> CARGO - SHIPPER    LOA BEAM DRAFT  dates...

    Text extraction interleaves those two lines, so the record is rebuilt by
    pairing a line that starts with a berth code against the line below it.
    """
    rows: list[ParsedRow] = []

    for index, line in enumerate(lines):
        berth_words = line.in_span(*COL_BERTH)
        if not berth_words:
            continue
        berth = berth_words[0].text.strip()
        if not BERTH_CODE_RE.match(berth):
            continue

        name_text = line.text_in_span(*COL_NAME)
        if not re.search(r"M[VT]\.", name_text):
            continue

        # The continuation line carries the dimensions and cargo. Remarks and
        # tonnage figures can render between the two, so scan a short window
        # for the first line below that actually holds a dimension triple.
        cargo: str | None = None
        loa = beam = draft = None
        for below in lines[index + 1 : index + 5]:
            numbers = [
                w.text
                for w in below.in_span(*COL_DIMS)
                if re.fullmatch(r"\d{1,3}\.\d{1,2}", w.text)
            ]
            if len(numbers) < 3:
                continue
            loa, beam, draft = (float(n) for n in numbers[:3])
            cargo_text = below.text_in_span(*COL_NAME)
            if cargo_text:
                cargo = cargo_text.split("-")[0].strip() or None
            break

        agent = line.text_in_span(*COL_AGENT) or None
        dl = line.text_in_span(*COL_DL)

        rows.append(
            ParsedRow(
                vessel_name=_clean_name(name_text),
                status="working",
                loa_m=loa,
                beam_m=beam,
                draft_m=draft or None,
                cargo=cargo,
                berth=berth,
                agent=agent,
                raw_row=f"{line.text} || {lines[index + 1].text if index + 1 < len(lines) else ''}"[:1000],
            )
        )

    # Direction of cargo (D/L) is captured in raw_row; not modelled separately.
    _ = dl if rows else None
    return rows


def split_sections(pages: list[str]) -> dict[str, str]:
    """Split the document text into its A/B/C/D sections."""
    sections: dict[str, list[str]] = {}
    current = "A"
    for page_text in pages:
        for line in page_text.splitlines():
            match = SECTION_RE.match(line)
            if match:
                current = match.group(1).upper()
            sections.setdefault(current, []).append(line)
    return {key: "\n".join(value) for key, value in sections.items()}


def parse_pdf(content: bytes, report_date: date) -> list[ParsedRow]:
    """Parse one daily traffic PDF into snapshot rows.

    Sections B and C are line-oriented and parsed from extracted text.
    Section A needs word coordinates, so its pages are clustered into visual
    lines first.
    """
    page_texts: list[str] = []
    page_lines: list[list[Line]] = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")
            page_lines.append(cluster_lines(page.extract_words()))

    # Section A runs from the first page until the page that starts section B.
    section_b_page = next(
        (i for i, t in enumerate(page_texts) if re.search(r"^\s*B\.\s+VESSELS", t, re.M)),
        len(page_texts),
    )

    rows: list[ParsedRow] = []
    for lines in page_lines[:section_b_page + 1]:
        rows += parse_working_lines(lines)

    sections = split_sections(page_texts)
    rows += parse_waiting(sections.get("B", ""), report_date)
    rows += parse_expected(sections.get("C", ""), report_date)

    # A vessel can appear in more than one section; keep the most advanced
    # status so a berthed vessel is not also counted as waiting.
    priority = {"working": 3, "waiting": 2, "expected": 1}
    best: dict[tuple[str, str], ParsedRow] = {}
    for row in rows:
        key = (row.vessel_name, row.status)
        if key not in best:
            best[key] = row

    seen_best: dict[str, ParsedRow] = {}
    for row in best.values():
        current = seen_best.get(row.vessel_name)
        if current is None or priority[row.status] > priority[current.status]:
            seen_best[row.vessel_name] = row

    return list(seen_best.values())


class ParadipBerthingAdapter(SourceAdapter):
    name = "paradip_daily_traffic"
    description = "Paradip Port Authority daily traffic update (working, waiting, expected vessels)"
    homepage = "https://www.paradipport.gov.in/"

    #: How many days back to look for the most recent published report.
    lookback_days = 7

    def url_for(self, report_date: date) -> str:
        return (
            f"https://www.paradipport.gov.in/uploads/"
            f"{report_date.year}/{report_date.month:02d}/"
            f"dtr{report_date.day:02d}{report_date.month:02d}.pdf"
        )

    def ingest_date(self, session: Session, report_date: date) -> tuple[int, bool]:
        """Ingest one report date. Returns (rows written, whether cache was used)."""
        url = self.url_for(report_date)
        fetched = self.fetch(url)

        # Archive the raw PDF; the parser can be improved and replayed later.
        archive = RAW_DIR / "paradip" / f"dtr_{report_date.isoformat()}.pdf"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(fetched.content)

        parsed = parse_pdf(fetched.content, report_date)
        provenance = self.provenance(fetched)

        rows = [
            {
                "port_code": PORT_CODE,
                "snapshot_date": report_date,
                "vessel_name": row.vessel_name,
                "status": row.status,
                "berth": row.berth,
                "agent": row.agent,
                "cargo": row.cargo,
                "cargo_tonnes": row.cargo_tonnes,
                "arrival_date": row.arrival_date,
                "draft_m": row.draft_m,
                "loa_m": row.loa_m,
                "raw_row": row.raw_row,
                **provenance,
            }
            for row in parsed
        ]

        written = upsert(
            session,
            PortCallSnapshot,
            rows,
            ["port_code", "snapshot_date", "vessel_name", "status"],
        )
        return written, fetched.from_cache

    def run(self, session: Session) -> IngestResult:
        """Ingest the most recent report available within the lookback window."""
        today = date.today()
        errors: list[str] = []

        for offset in range(self.lookback_days):
            report_date = today - timedelta(days=offset)
            try:
                written, cached = self.ingest_date(session, report_date)
            except Exception as exc:
                errors.append(f"{report_date}: {type(exc).__name__}")
                continue

            if written:
                return IngestResult(
                    self.name,
                    "stale" if cached else "ok",
                    rows_written=written,
                    message=f"report {report_date.isoformat()}",
                    used_cache=cached,
                    extra={"report_date": report_date.isoformat()},
                )

        return IngestResult(
            self.name,
            "failed",
            message=f"no report found in last {self.lookback_days} days ({'; '.join(errors[:4])})",
        )

    def backfill(self, session: Session, start: date, end: date) -> dict:
        """Walk a date range, ingesting every report that exists.

        The month-scoped URL path means historical reports remain available,
        so a real queue history can be built immediately rather than only
        accumulating from today.
        """
        found, missing, total_rows = 0, 0, 0
        cursor = start
        while cursor <= end:
            try:
                written, _ = self.ingest_date(session, cursor)
                if written:
                    found += 1
                    total_rows += written
                else:
                    missing += 1
            except Exception:
                missing += 1
            cursor += timedelta(days=1)
            session.commit()

        return {"reports": found, "missing": missing, "rows": total_rows}
