"""Shanghai Shipping Exchange - China Import Dry Bulk Freight Index (CDFI).

Source: https://en.sse.net.cn/indices/cdfinew2.jsp
Data endpoint: https://en.sse.net.cn/currentIndex?indexName=cdfi

This is the platform's freight target variable, and it is real and free.

Why CDFI rather than a Baltic route. Baltic route assessments (S8_58, C5, C3)
are the right benchmark and are paid. CDFI is the closest free analogue that
exists, and for this project it is closer than it first looks: three of its
coal routes load at ports already in our port constraint master.

    CDFI_CD7   Newcastle (AUNTL)  -> Zhoushan    130,000 t   Capesize
    CDFI_CD9   Hay Point (AUHPT)  -> Zhoushan     85,000 t   Kamsarmax
    CDFI_CD18  Samarinda          -> Guangzhou    70,000 t   Panamax
    CDFI_CD14  Taboneo (IDTAB)    -> Guangzhou    50,000 t   Supramax
    CDFI_CD26  Taboneo (IDTAB)    -> Nantong      45,000 t   Supramax

Same load ports, same parcel sizes, same vessel classes as our own lanes, with
a different discharge region. That is exactly the shape the strategy document
section 9.1 asks for: model the East Coast India lane as a *basis* to a
published route, because the basis is far more stationary than the level and
it is the part that is specific to us.

Depth. The daily-history endpoint (``/singleIndex/cdfi``) is subscriber-gated
and returns HTTP 500 without a session, so there is no backfill: this adapter
captures forward only. Each fetch does yield two dates, because the endpoint
returns both the current assessment and the previous publication, but the
archive is only ever as deep as the date we started it. Same posture as the
CEA feed, and the reason SYNTHETIC_DATA.md carries a disclosed history series
for model development.
"""
from __future__ import annotations

import json
from datetime import date, datetime

from sqlalchemy.orm import Session

from ..config import RAW_DIR
from ..models import RateObservation
from .base import IngestResult, SourceAdapter, upsert

ENDPOINT = "https://en.sse.net.cn/currentIndex"
LANDING_PAGE = "https://en.sse.net.cn/indices/cdfinew2.jsp"

# SSE item code -> (our series code, series_kind, unit).
#
# Held explicitly rather than accepting whatever the endpoint returns, so that
# a route being renamed or withdrawn fails loudly here instead of silently
# changing the meaning of a stored series.
SERIES_MAP: dict[str, tuple[str, str, str]] = {
    # Composite and cargo sub-indices, in index points.
    "CDFI_T": ("CDFI_COMPOSITE", "freight_index", "point"),
    "CDFI_T_COAL": ("CDFI_COAL", "freight_index", "point"),
    "CDFI_T_IRON_ORE": ("CDFI_IRON_ORE", "freight_index", "point"),
    # Coal routes, in $/tonne. These are the ones the timing engine prices against.
    "CDFI_CD7": ("CDFI_NEWCASTLE_ZHOUSHAN_130K", "route", "usd_per_t"),
    "CDFI_CD9": ("CDFI_HAYPOINT_ZHOUSHAN_85K", "route", "usd_per_t"),
    "CDFI_CD18": ("CDFI_SAMARINDA_GUANGZHOU_70K", "route", "usd_per_t"),
    "CDFI_CD14": ("CDFI_TABONEO_GUANGZHOU_50K", "route", "usd_per_t"),
    "CDFI_CD26": ("CDFI_TABONEO_NANTONG_45K", "route", "usd_per_t"),
    # Iron ore benchmarks: the free analogues of Baltic C5 and C3, kept as
    # market-state features rather than as lanes we would ever fix.
    "CDFI_CD2": ("CDFI_DAMPIER_QINGDAO_170K", "route", "usd_per_t"),
    "CDFI_CD3": ("CDFI_TUBARAO_QINGDAO_170K", "route", "usd_per_t"),
}

# Routes we would actually fix, with the load port they leave from and the
# vessel class their parcel size implies. Consumed by the timing engine to
# choose which published route a candidate voyage should be priced against.
ROUTE_META: dict[str, dict] = {
    "CDFI_NEWCASTLE_ZHOUSHAN_130K": {
        "load_port_code": "AUNTL",
        "vessel_class": "Capesize",
        "parcel_tonnes": 130_000,
        "cargo": "coal",
    },
    "CDFI_HAYPOINT_ZHOUSHAN_85K": {
        "load_port_code": "AUHPT",
        "vessel_class": "Kamsarmax",
        "parcel_tonnes": 85_000,
        "cargo": "coal",
    },
    "CDFI_SAMARINDA_GUANGZHOU_70K": {
        "load_port_code": "IDTAB",
        "vessel_class": "Panamax",
        "parcel_tonnes": 70_000,
        "cargo": "coal",
    },
    "CDFI_TABONEO_GUANGZHOU_50K": {
        "load_port_code": "IDTAB",
        "vessel_class": "Supramax",
        "parcel_tonnes": 50_000,
        "cargo": "coal",
    },
    "CDFI_TABONEO_NANTONG_45K": {
        "load_port_code": "IDTAB",
        "vessel_class": "Supramax",
        "parcel_tonnes": 45_000,
        "cargo": "coal",
    },
}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_cdfi(content: bytes) -> list[dict]:
    """Parse the currentIndex payload into observation records.

    The endpoint returns each line's value on ``currentDate`` and its value on
    ``lastDate``, so one fetch yields up to two observations per series.
    """
    payload = json.loads(content.decode("utf-8"))
    if payload.get("status") != 1:
        raise ValueError(f"SSE returned status {payload.get('status')}: {payload.get('msg')}")

    data = payload.get("data") or {}
    current_date = _parse_date(data.get("currentDate"))
    last_date = _parse_date(data.get("lastDate"))
    if current_date is None:
        raise ValueError("SSE payload carried no usable currentDate")

    lines = data.get("lineDataList") or []
    if not lines:
        raise ValueError("SSE payload carried no lineDataList")

    seen = {row.get("dataItemTypeName") for row in lines}
    missing = set(SERIES_MAP) - seen
    if missing:
        raise ValueError(
            f"CDFI layout changed: expected routes absent from the feed: {sorted(missing)}"
        )

    records: list[dict] = []
    for row in lines:
        mapped = SERIES_MAP.get(row.get("dataItemTypeName"))
        if mapped is None:
            continue  # a route we do not model; not an error
        series_code, series_kind, unit = mapped

        for obs_date, value in (
            (current_date, row.get("currentContent")),
            (last_date, row.get("lastContent")),
        ):
            if obs_date is None or value is None:
                continue  # SSE publishes null when a route did not trade
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "series_code": series_code,
                    "series_kind": series_kind,
                    "obs_date": obs_date,
                    "value": numeric,
                    "unit": unit,
                }
            )

    if not records:
        raise ValueError("CDFI parsed to zero observations")
    return records


class SseCdfiAdapter(SourceAdapter):
    name = "sse_cdfi"
    description = (
        "Shanghai Shipping Exchange China Import Dry Bulk Freight Index: "
        "route-level coal and iron ore assessments in $/tonne"
    )
    homepage = LANDING_PAGE

    def _url(self) -> str:
        return f"{ENDPOINT}?indexName=cdfi"

    def probe(self) -> tuple[bool, str]:
        response = self.client.get(
            ENDPOINT, params={"indexName": "cdfi"}, headers={"Referer": LANDING_PAGE}
        )
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        try:
            payload = response.json()
            published = (payload.get("data") or {}).get("currentDate")
            return payload.get("status") == 1, f"assessment for {published}"
        except Exception as exc:
            return False, f"unparseable payload: {type(exc).__name__}"

    def run(self, session: Session) -> IngestResult:
        url = self._url()
        fetched = self.fetch(url)

        records = parse_cdfi(fetched.content)
        latest = max(record["obs_date"] for record in records)

        archive = RAW_DIR / "sse_cdfi" / f"cdfi_{latest.isoformat()}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(fetched.content)

        provenance = self.provenance(fetched, url=LANDING_PAGE)
        rows = [{**record, **provenance} for record in records]
        written = upsert(session, RateObservation, rows, ["series_code", "obs_date"])

        series_count = len({row["series_code"] for row in rows})
        return IngestResult(
            self.name,
            "stale" if fetched.from_cache else "ok",
            rows_written=written,
            message=f"{series_count} series, assessment {latest.isoformat()}",
            used_cache=fetched.from_cache,
        )
