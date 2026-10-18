# Data Sources

Every external source the platform uses, what it provides, and its access
terms. Sources marked **live** are implemented and ingesting; the rest are
identified and verified reachable but not yet wired in.

For the separate question of where the platform uses data that is *not* real,
see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md). At the time of writing that file
records no synthetic data in use.

---

## Live

| Source | Provides | Cadence | Access | Adapter |
|---|---|---|---|---|
| [Paradip Port Authority daily traffic](https://www.paradipport.gov.in/) | Working, waiting and expected vessels; berth, cargo, quantity, arrival stamp, LOA, draft | Daily | Free, public PDF | `paradip_daily_traffic` |
| [CEA daily coal report](https://npp.gov.in/publishedReports) | Coal stock, normative stock and daily requirement for 191 thermal plants, including the 15 designed on imported coal | Daily | Free, public XLS | `cea_daily_coal_stock` |
| [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Monthly Australian and South African thermal coal, crude, iron ore, gas | Monthly | Free, public XLSX | `worldbank_pinksheet` |
| [Shanghai Shipping Exchange CDFI](https://en.sse.net.cn/indices/cdfinew2.jsp) | China Import Dry Bulk Freight Index: composite, coal and iron ore sub-indices, and route-level coal assessments in $/tonne | Daily | Free, public JSON endpoint | `sse_cdfi` |
| [IBTrACS North Indian Ocean](https://www.ncei.noaa.gov/products/international-best-track-archive) | Cyclone best tracks, using the IMD RSMC New Delhi assessment where present | Per storm | Free, public CSV | `ibtracs_north_indian` |

### Notes on the live sources

**Paradip URL scheme.** Reports are published at
`/uploads/YYYY/MM/dtrDDMM.pdf`. Because the path is scoped by year and month,
past reports remain retrievable and *can* be backfilled — 73 daily reports
covering 2026-06-18 to 2026-09-01 were pulled on first run. The older
`/Writereaddata/Daily_Traffic/dtrDDMM.pdf` scheme reused filenames annually
and is treated as best-effort only. Every PDF is archived to
`backend/data/raw/paradip/` on fetch, so the parser can be improved and
replayed against history rather than re-scraped.

**Paradip parsing.** Sections B (waiting at anchorage) and C (expected) are
line-oriented and parse from extracted text. Section A (working vessels) uses
stacked two-line table cells that text extraction interleaves, so it is parsed
from word coordinates instead — see `app/ingest/pdf_layout.py`.

**CEA retention.** `npp.gov.in` keeps only a short window of daily reports, so
long backfill mostly misses. Depth accumulates going forward, which is why
the adapter runs daily.

**Pink Sheet URL rotation.** The workbook link carries a rotating document id,
so the adapter discovers the current URL from the landing page rather than
pinning one that would silently go stale. A known-good fallback is retained.

**CDFI is the freight target variable.** This is the series the forecaster
predicts and the timing engine prices against, and it is real and free. Three
of its coal routes load at ports already in our constraint master:

| CDFI route | Load port | Parcel | Class |
|---|---|---|---|
| Newcastle - Zhoushan | AUNTL | 130,000 t | Capesize |
| Hay Point - Zhoushan | AUHPT | 85,000 t | Kamsarmax |
| Samarinda - Guangzhou | (Kalimantan) | 70,000 t | Panamax |
| Taboneo - Guangzhou | IDTAB | 50,000 t | Supramax |
| Taboneo - Nantong | IDTAB | 45,000 t | Supramax |

Same load ports, same parcel sizes, same vessel classes as our own lanes, with
a different discharge region. That is what makes it usable as the published
route an East Coast India lane can be modelled as a *basis* to.

**CDFI depth.** The free `currentIndex` endpoint returns the current assessment
and the previous publication date only; the dated history endpoint
(`/singleIndex/cdfi`) is subscriber-gated and returns HTTP 500 without a
session. So there is no backfill and the archive is only as deep as the date we
started it, exactly like the CEA feed. This is the reason a disclosed synthetic
development series exists - see `SYNTHETIC_DATA.md`.

**IBTrACS rather than scraping IMD.** The strategy document asks for IMD RSMC
New Delhi cyclone tracks. IBTrACS is the better route to precisely that data:
NOAA consolidates every agency's best track into one free CSV, and the North
Indian Ocean file carries the `NEWDELHI_*` columns, which *are* the IMD
assessment. No scraping, no extra dependency, and history back to the
nineteenth century. The adapter keeps the last 30 seasons, because storm
behaviour before satellite coverage is not comparable.

---

## Verified reachable, not yet wired in

| Source | Provides | Access | Status |
|---|---|---|---|
| [Visakhapatnam Port Authority](https://vizagport.com/) | Berthing programme, working and expected vessels | Free | Reachable. Page is a base64-routed portal, so it needs its own parser rather than a copy of the Paradip one. A second congestion series is the highest-value addition left |
| [SMP Kolkata draft circulars](https://smportkolkata.shipping.gov.in/) | Monthly Hooghly permissible draft as a function of LOA | Free PDF | Reachable, not wired in |
| [AISStream.io](https://aisstream.io/) | Live global AIS over WebSocket | Free key | Reachable; per-user bandwidth limits since September 2026, so subscriptions must be bounding-box and message-type filtered |
| [Indonesian HBA](https://minerba.esdm.go.id/harga_acuan) | Government monthly reference coal price | Free | Reachable (HTTP 200), not wired in |
| [US EIA](https://www.eia.gov/opendata/) | Coal production, exports, stocks | Free API | Requires an API key; returns HTTP 403 without one |
| [Ship & Bunker](https://shipandbunker.com/) | Daily VLSFO and MGO at major bunkering ports | Free daily | **Reachable but not parseable.** The price tables render in JavaScript with no underlying JSON endpoint, so they cannot be read without driving a browser. Not worth a headless-browser dependency for one feature; the platform uses real Brent as the cost-side market signal and takes bunker price as an explicit input with a disclosed default |
| Adani Ports (Dhamra, Gangavaram) | Published berth limits | Free | Reachable, not wired in |

---

## Paid, and therefore out of scope for this build

These are the sources the strategy document identifies as production-grade,
and they are the reason certain entries appear in `SYNTHETIC_DATA.md`.

| Source | Why it matters | Why not used |
|---|---|---|
| **Baltic Exchange** route assessments (S8_58, C5, C3, P3A, S1B) and FFA curves | S8_58 is the direct benchmark for the Indonesia to East Coast India lane; the FFA curve is the honest benchmark any forecast must beat | Paid subscription; exchange-licensed |
| **Kpler** (incl. MarineTraffic, FleetMon, Spire) | Historical AIS with port-call events and congestion analytics | Enterprise licensing |
| **Argus / McCloskey** API 2/4/5/8, **globalCOAL** NEWC, **Coalindo** ICI 1-6 | The grade-specific coal benchmarks actually traded | Paid |
| **Clarksons SIN / Drewry / Veson** | Fleet, orderbook, deliveries, scrapping | Paid |
| Broker circulars (Clarksons, Braemar, SSY, Howe Robinson) | The only real source of East Coast India lane pricing | Relationship-based |

**Architectural consequence.** Every feed sits behind a `SourceAdapter` with a
cached last-known-good payload, so swapping a free proxy for a paid feed is a
new adapter class and a registry entry, not a rewrite. A fetch failure
degrades to cache and records the run as `stale` rather than taking the
platform down.

---

## Port constraint master

`backend/data/reference/port_constraints.yaml` holds 14 ports and their berth
limits. Each port entry cites its source. Key published figures in use:

| Port | Max draft | Max LOA | Notes |
|---|---|---|---|
| Gangavaram | 19.5 m | 300 m | Water depth to 21 m; fully laden Super Capesize to 200,000 DWT |
| Dhamra | 18.0 m | 290 m | 47 m beam, 180,000 DWT |
| Visakhapatnam (outer) | 18.1 m | 300 m | Deep draft coal berth, 356 m quay, 200,000 DWT |
| Paradip | 14.5 m (coal berths) | 250 m | Entrance channel 17.1 m; berth drafts read from the daily traffic PDF |
| Gopalpur | 14.5 m | 250 m | Harbour natural depth 18.5 m |
| Sandheads | 22.0 m | 340 m | Transhipment anchorage, 40-50 m water |
| Sagar | 9.5 m | 300 m | Transhipment anchorage |
| Haldia | 8.5 m | 230 m | Brackish (1.008 t/m3) and tide-dependent; draft is a function of LOA and is reissued monthly |

Two of these values do more work than the rest. Haldia's 8.5 m draft against
230 m LOA is what makes transhipment at Sandheads a real commercial option
rather than an edge case. Haldia's water density of 1.008 t/m3 is why the
fresh water allowance correction is in the loadable-tonnes engine at all:
ignoring it overstates cargo on exactly the lane where parcels are tightest.

---

## Reference data that is not a feed

`backend/data/reference/vessel_classes.yaml` holds eight vessel class
archetypes. These are representative of a modern vessel in each class rather
than measurements of a named ship, and the engine accepts real vessel
particulars in place of the archetype. Derived quantities such as TPC are
computed from documented naval architecture relations, with the derivation
stated in the file. This is discussed further in `SYNTHETIC_DATA.md`.
