# Freight Forecasting & Chartering Optimisation Platform
### End-to-end build plan for bulk cargo procurement into India's East Coast ports

---

## 0. Reframing the problem before building anything

The problem statement asks for a "freight forecasting model." Taken literally, that is the wrong thing to build — and it is the reason most projects of this type get shelved after a good-looking demo.

Dry bulk freight rates are close to a random walk in log space. A team that promises "predict Capesize rates 12 weeks out with 90% accuracy" will lose money and credibility. The market already has a forecast: the **FFA forward curve**, produced by people with more information than you, and it is available to trade against.

The actual value sits in three places, in order of size:

| Layer | What it does | Where the money is |
|---|---|---|
| **1. Forecast** | Probabilistic rate distributions, not point estimates | Modest. Real alpha is short-horizon nowcasting from vessel-supply data, plus *volatility* forecasting |
| **2. Feasibility & optimisation** | Constraint-aware vessel/port/parcel matching under draft, LOA, tide, berth and blend constraints | Large and defensible — this is deterministic engineering, not prediction |
| **3. Contract structuring** | How much annual volume to lock into COAs / period TCs vs leave to spot, and at what price | Largest. This is exactly the stated objective |

The last line of the problem statement — *"moving from multiple single spot contracts to short term / medium term multiple voyage contracts"* — is the real deliverable. That is a **stochastic portfolio problem under uncertainty**, and the forecast is just an input to it.

Build the system so that the forecast can be mediocre and the platform still pays for itself.

---

## 1. Target scope

**Cargoes:** thermal coal, coking coal (and extensible to limestone, pet coke, fertiliser raw material, iron ore backhaul).

**Origins:** Australia (Newcastle, Hay Point/Dalrymple Bay, Gladstone, Abbot Point), US (Baltimore, Norfolk/Lamberts Point, New Orleans/Mississippi, Mobile), Mozambique (Beira, Nacala, Maputo/Matola), Russia (Vostochny, Nakhodka, Vanino, Taman), Indonesia (Kalimantan anchorages — Taboneo, Muara Pantai, Adang Bay, Bunyu, Samarinda).

**Destinations:** Paradip, Visakhapatnam, Gangavaram, Gopalpur, Dhamra, Haldia, Sagar/Sandheads (lightering).

**Vessel classes:**

| Class | Indicative DWT | Gear | Typical ECI relevance |
|---|---|---|---|
| Handysize | 10–40k | Geared | Haldia direct, Gopalpur, coasters, lightering daughter vessels |
| Supramax / Ultramax | 50–65k | Geared | Indonesia → ECI workhorse; Beira; anchorage loading |
| Panamax / Kamsarmax | 76–85k | Gearless | Australia, US East Coast, Russia Far East |
| Capesize / Newcastlemax | 150–210k | Gearless | Gangavaram, Dhamra, Vizag outer harbour; Sandheads lightering |

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  L0  INGESTION                                                        │
│  AIS stream · Baltic/FFA feeds · commodity indices · bunker prices    │
│  port berthing PDFs · weather/cyclone ensembles · macro · internal ERP │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  L1  DATA PLATFORM (bronze → silver → gold, Parquet on object store)  │
│  Voyage reconstruction · port-call detection · congestion metrics      │
│  Fixture normalisation · port constraint master · tide tables          │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌───────────────────────────────────┬──────────────────────────────────┐
│  L2  FORECASTING SERVICE          │  L3  CONSTRAINT ENGINE           │
│  Rate distributions (quantiles)   │  Loadable-tonnes calculator      │
│  Congestion / waiting-time model  │  Draft·tide·density·LOA·beam     │
│  Volatility & regime model        │  Berth rate, laycan feasibility  │
│  Demurrage risk model             │  Lightering topology             │
└───────────────────────────┬───────┴──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  L4  DECISION / OPTIMISATION LAYER                                    │
│  Scenario generator → two-stage stochastic MILP                       │
│  Vessel selection · timing · contract mix (spot/COA/TC) · CVaR budget  │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  L5  APPLICATION — dashboard, alerts, explainability, audit ledger    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data — training / historical

Split into **must-have**, **high-value**, and **nice-to-have**. Depth target: 10–15 years so the model sees at least two full cycles (2008 crash, 2016 trough, 2021 spike, 2022 energy crisis, 2023–24 Red Sea/Panama disruption).

### 3.1 Freight rates — the core target variable

| Source | Content | Access | Notes |
|---|---|---|---|
| **Baltic Exchange** | BDI, BCI 5TC, BPI 82 5TC, BSI 58 10TC, BHSI 38 7TC; 200+ route assessments; searchable historical fixture database; FFA forward curves | Paid subscription; API portal at `api.balticexchange.com`; also redistributed via ICE Consolidated Feed, Bloomberg, Refinitiv | Non-negotiable for a production system. Route **S8_58 — South China via Indonesia to East Coast India** is a direct benchmark for your Indonesian coal lane. C5 (W Australia–Qingdao), C3 (Tubarão–Qingdao), P3A/P5, S1B, S1C, S10 give you the surrounding market |
| **Argus / Baltic FFA settlement + SGX / EEX cleared freight futures** | Forward curves by class and quarter | Exchange data feeds | This is the market's own forecast. Treat as both feature and benchmark |
| **Broker circulars** (Clarksons, Braemar, Fearnleys, Simpson Spence Young, Howe Robinson, local Indian brokers) | Daily fixture lists, market commentary, ECI-specific lane colour | Email PDFs / relationship-based | The only real source of Indian East Coast lane pricing. Extract with the LLM pipeline in §9.9 |
| **Your own fixture history** | Every fixture you've done: laycan, rate, vessel, demurrage, actual load/discharge times | Internal ERP / chartering desk spreadsheets | Highest-signal, smallest-volume dataset. Digitising this properly is week-one work |
| **Clarksons Shipping Intelligence Network / Drewry / Veson VesselsValue** | Fleet, orderbook, deliveries, scrapping, S&P prices, timeseries | Paid | Supply-side fundamentals |

### 3.2 Commodity prices — demand driver and cointegrated partner

| Source | Content | Access |
|---|---|---|
| **globalCOAL NEWC Index** | Australian 6,000 kcal/kg NAR FOB Newcastle benchmark | Paid (weekly weighted average of trades, bids, offers) |
| **Argus/McCloskey API 2, API 4, API 5, API 8** | ARA CIF, Richards Bay FOB, Australian and Chinese import benchmarks | Paid |
| **Argus/Coalindo ICI 1–6** | Indonesian grades, including the low-CV grades ECI actually buys | Paid |
| **Indonesian HBA** (Ministry of Energy & Mineral Resources) | Monthly government reference price | **Free** at `minerba.esdm.go.id/harga_acuan` |
| **World Bank Pink Sheet** | Monthly coal, iron ore, energy series back to 1960 | **Free** |
| **IMF Primary Commodity Price System** | Monthly, long history | **Free** |
| **US EIA** | Coal production, exports, stocks, Appalachian/PRB prices | **Free API** |
| **Platts / Argus coking coal (PLV FOB Australia, CFR India)** | Coking coal specifically — India is the swing buyer | Paid |

### 3.3 Vessel movement and supply — where the alpha is

| Source | Content | Access |
|---|---|---|
| **Kpler** (now owns MarineTraffic, FleetMon, Spire Maritime) | Global terrestrial + satellite AIS, 10+ years historical, port-call events, congestion analytics, predictive events; delivered via API, LiveDB, NMEA, custom datasets | Enterprise; single largest data line item |
| **S&P Global (ex-ORBCOMM AIS)** | Alternative satellite AIS constellation | Enterprise |
| **Datalastic** | Self-serve AIS API with history, from ~€99/month | Cheap; good for PoC |
| **AISStream.io / AISHub** | Free real-time AIS via WebSocket / community terrestrial network | Free; coverage gaps offshore — fine for prototyping, not production |

> **Note on vendor risk:** the AIS market consolidated sharply — Kpler acquired MarineTraffic and FleetMon (2023) and Spire Maritime (closed April 2025); S&P Global completed its ORBCOMM AIS acquisition in November 2025. Negotiate feed-continuity clauses, and architect the ingestion layer behind an adapter interface so a provider swap is a config change.

**What you derive from AIS (this is the crown jewel):**
- Historical voyage reconstruction: load port → discharge port → ballast leg, with dates
- Port-call detection, berth-level dwell times, anchorage queue length and waiting time per port
- Laden/ballast classification by draught reports and speed profile
- **Tonnage supply index:** count of ballast vessels by class within *N* days' steaming of each load port
- Speed distributions (slow-steaming signals a soft market; speeding up signals firming)
- Congestion-absorbed tonnage — how much of the fleet is effectively removed from supply

### 3.4 Port infrastructure and operations

| Source | Content | Access |
|---|---|---|
| **Port authority berthing programmes** — Paradip Port Authority, Visakhapatnam Port Authority, SMP Kolkata/Haldia, Dhamra, Gangavaram (Adani), Gopalpur | Daily "Vessels Working / Waiting / Expected" tables with vessel, agent, tonnage, commodity, berth, prior berthing sequence, plus tide tables in the same document | **Free** — published as daily PDFs/pages on port websites. Scrape and archive from day one; nobody keeps history of these |
| **V-Ocean Shipping, Ruzave, exportparadip** and similar aggregators | Consolidated Indian port line-ups and berth restriction summaries (Haldia, Dhamra, Gopalpur, Paradip, Vizag, Gangavaram…) | Free / freemium — useful cross-check |
| **Port draft circulars** | Monthly-revised maximum permissible draft, often as a function of LOA | Free from port sites. **Critical:** draft is not a constant. SMP Kolkata publishes draft circulars monthly; Haldia Dock is roughly 8.5 m draft / 230 m LOA and mainly takes fully laden Handysize |
| **Indian Ports Association / Ministry of Ports, Shipping & Waterways** | Monthly major-port traffic, turnaround time, output per ship-berth-day, pre-berthing detention | Free (`ipa.nic.in`, `sagar-setu`, MoPSW *Sagar Vidya Kosh*) |
| **PCS 1x / Sagar Setu (National Logistics Portal – Marine)** | Vessel and cargo declarations, berth allocation — agents already file through PCS mode | Requires access agreement; the highest-fidelity operational feed available |
| **Origin port authorities** | Newcastle Port (PWCS/NCIG stem lists and vessel queue — publicly reported), Dalrymple Bay, Gladstone, Richards Bay, Beira/CFM | Free to mixed |
| **Admiralty / IHO tide data, INCOIS** | Tidal predictions for Hooghly, Paradip, Vizag approaches | Paid/free mix; INCOIS free for Indian waters |

Build a **port constraint master table**: for each berth — max LOA, beam, DWT, draft (and its tidal/seasonal profile), air draft, gear requirement, discharge rate by cargo type, laytime terms, demurrage/despatch rates, working days convention, floating-crane availability.

### 3.5 Weather, climate, disruption

| Source | Content | Access |
|---|---|---|
| **ECMWF ERA5 reanalysis** | Hourly wind, wave, precipitation back to 1940 — for training | Free (Copernicus CDS) |
| **IMD Regional Specialised Meteorological Centre, New Delhi** | Bay of Bengal cyclone tracks, historical and live bulletins | Free |
| **NOAA GFS / NCEP, ECMWF ensembles, Copernicus Marine** | Forecast winds, waves, swell | Free / paid tiers |
| **Australian BoM** | Queensland/NSW cyclone season and flooding — historically the biggest coal supply shock source | Free |
| **Indonesian BMKG** | Kalimantan monsoon rainfall — anchorage loading stops in heavy rain | Free |

### 3.6 Macro and canal / chokepoint

- Chinese steel PMI, blast furnace utilisation, port coal stockpiles (Qinhuangdao), NBS output data
- India: CEA daily power plant coal stock report (**free, daily** — the single best proxy for your own urgency), Coal India production, DGCIS/Ministry of Commerce import statistics, UN Comtrade
- Bunker prices: Singapore, Fujairah, Rotterdam VLSFO/MGO (Ship & Bunker free daily; Platts/Argus paid), Brent/WTI
- Panama Canal Authority transit slots and draft restrictions; Suez Canal Authority; Red Sea routing status
- USD/INR, freight-relevant FX, US 10y and China credit impulse as cycle proxies

---

## 4. Data — real-time / operational feeds

These are what the live system consumes each day.

| Feed | Cadence | Latency | Purpose |
|---|---|---|---|
| AIS position stream (Kafka topic) | Continuous | Seconds–minutes | Live tonnage supply, congestion, ETA revisions |
| Baltic indices + FFA curves | Daily, ~midday UK | Same day (delayed tiers exist for clients of panellist brokers) | Nowcast anchor, benchmark, hedging prices |
| Port berthing programme scrape | Daily, ~06:00 IST | < 1 hr | Queue position, expected berthing, tide windows |
| CEA daily coal stock report | Daily | 1 day | Internal demand urgency signal |
| Bunker price feed | Daily | Same day | Voyage cost, TCE conversion |
| Weather + cyclone bulletins | 4×/day | Minutes | Berth-closure probability, ETA risk |
| Broker circulars & fixture lists | Daily | Hours (LLM-parsed) | Physical market truth for ECI lanes |
| Commodity index assessments | Daily/weekly | Same day | Demand-side driver, landed-cost calculation |
| Internal ERP: stock levels, plant burn rate, open cargo requirements | Hourly | Minutes | Demand forecast, urgency weighting |
| Vessel noon reports (for vessels on TC) | Daily | Hours | Performance claims, speed/consumption, idle detection |

**Architectural rule:** every external feed sits behind an adapter with a cached last-known-good value, and the system must degrade gracefully — a Baltic outage should not take down the recommendation engine.

---

## 5. Feature engineering

The feature set is where domain knowledge beats model architecture.

**Supply-side (AIS-derived)**
- `ballast_supply_7d`, `ballast_supply_14d`, `ballast_supply_21d` — vessels of class *v* that could reach load port *l* within the window
- `congestion_tonnage` — DWT sitting at anchorage > 3 days, per port and region
- `fleet_speed_index` — mean laden speed vs 3-year average (proxy for market tightness)
- `orderbook_to_fleet`, `deliveries_next_quarter`, `scrapping_ttm`
- `queue_length_paradip`, `queue_length_vizag`, … and their 7/30-day slopes

**Demand-side**
- Coal price spreads: NEWC − ICI4, API4 − ICI4, delivered-India arbitrage vs domestic Coal India price
- Chinese import arbitrage — when China pulls Indonesian coal, ECI Supramax rates rise
- CEA plant stock days-of-burn (national and eastern region)
- Indian import volumes by origin, 3-month rolling

**Market state**
- FFA forward curve level, slope (Q1–Q4), and curvature; contango/backwardation flag
- Realised volatility (5, 20, 60-day), GARCH conditional variance
- Cross-class spreads: BCI/BPI, BPI/BSI ratios (Cape-to-Panamax substitution signal)
- HMM regime posterior probability

**Cost-side**
- Bunker price, bunker-adjusted TCE, route distance, canal fees, ECA exposure

**Calendar / seasonality**
- Fourier terms on annual cycle; Chinese New Year offset; Indian monsoon (Jun–Sep); Bay of Bengal cyclone season (Oct–Dec, secondary Apr–May); Australian cyclone season (Dec–Apr); Indonesian wet season; Northern Hemisphere heating demand; Indian pre-monsoon stockpiling (Mar–May)

**Text-derived**
- Sentiment and topic scores from broker circulars (§9.9)

---

## 6. Modelling approach

### 6.1 Targets

Forecast **distributions**, not points. Quantiles {5, 10, 25, 50, 75, 90, 95} at horizons of 1, 2, 4, 8, 13 and 26 weeks, for:

1. Timecharter averages by class ($/day)
2. Voyage rates ($/t) on your actual lanes — modelled as *basis to a published Baltic route*, since ECI lanes are mostly unpublished
3. Waiting time at each discharge port (days)
4. Demurrage exposure per fixture ($)

### 6.2 Benchmarks you must beat (and report honestly)

| Benchmark | Why |
|---|---|
| Random walk / seasonal naive | Freight is near-unit-root; this is a brutal baseline |
| **FFA forward curve** | The market's own forecast. If you cannot beat it, your model has no information the market lacks — and you should just hedge with FFAs instead |
| Current chartering desk's decisions | The organisational baseline that actually matters |

### 6.3 Model family

| Model | Role |
|---|---|
| SARIMAX / VECM with exogenous drivers | Interpretable baseline; captures cointegration between freight, coal price and bunkers |
| **LightGBM/CatBoost, direct multi-horizon quantile regression** | Primary workhorse. Handles the wide, heterogeneous feature set; fast to retrain; SHAP-explainable |
| **N-HiTS / TFT / PatchTST** (Nixtla `neuralforecast`, PyTorch Forecasting, Darts) | Multi-series learning across classes and routes with shared seasonality; TFT gives attention-based interpretability |
| GARCH / HAR-RV | Volatility forecasting — drives option value and timing urgency |
| Hidden Markov / Bayesian change-point | Regime identification (tight / balanced / soft); models are conditioned on regime |
| **Conformal prediction** wrapper | Distribution-free, calibrated intervals. Essential — the optimiser consumes intervals, so miscalibration propagates into bad decisions |
| Hierarchical reconciliation (MinT) | Keeps class-level and route-level forecasts coherent |

**Ensemble:** stack with a regime-conditional weighting, and always blend toward the FFA curve at long horizons where model skill decays.

### 6.4 Congestion / waiting-time model

Separate model, and arguably more valuable than the rate model because it is more predictable. Queueing-theory features (arrival rate, service rate, berth count) fed into a gradient-boosted survival model, predicting the distribution of pre-berthing detention. Validate against the archived berthing programmes.

### 6.5 Evaluation

- **Strictly walk-forward** with an embargo period; no random splits, no lookahead in feature construction
- Metrics: MASE (vs seasonal naive), pinball loss, CRPS, directional accuracy, PIT histogram for calibration
- **Economic backtest is the real test:** replay the last 5 years, let the system make chartering decisions with only the information available at each date, and compute realised $/t versus (a) what the desk actually paid and (b) a naive always-spot policy
- Report performance *by regime* — a model that only works in stable markets is a liability

---

## 7. The decision layer (this is the differentiator)

### 7.1 Loadable-tonnes engine (deterministic, high ROI)

For a candidate vessel at a candidate berth:

```
loadable_tonnes = f( vessel DWT, lightship, constants,
                     berth declared draft (from current monthly circular),
                     tidal window at planned ETA,
                     water density (seawater vs brackish Hooghly / freshwater Mississippi),
                     under-keel clearance policy,
                     bunker ROB on arrival,
                     stability & trim,
                     LOA / beam / air-draft limits )
```

This alone is worth building first. It converts "which vessel fits" from tribal knowledge into an auditable number, and squeezing an extra 1,500 t onto a Panamax is real money on every voyage.

### 7.2 Scenario generation

Sample *S* joint scenarios (typically 500–2,000) from the probabilistic forecasts: freight rates, bunker prices, port waiting times, cyclone-driven berth closures, coal prices. Preserve the correlation structure — a copula or a block bootstrap of historical residuals, not independent draws.

### 7.3 Two-stage stochastic MILP

**Stage 1 (here-and-now):** how much of the next 6–18 months' volume to commit to COAs and period time charters, with which counterparties, at what rates and laycan spreads.

**Stage 2 (recourse, per scenario):** spot fixtures, vessel/port assignment, lightering decisions, cargo sequencing.

```
minimise   E[ total delivered cost ]  +  λ · CVaR_95[ total delivered cost ]

subject to
  annual tonnage requirement per plant, per grade
  blend quality: GCV, ash, sulphur, moisture within spec
  stockpile min/max at each discharge port and plant
  vessel feasibility: draft(tide, density), LOA, beam, DWT, gear
  berth capacity and discharge rate; laycan windows
  COA volume tolerance and nomination rules
  lightering capacity (floating cranes, barges) at Sandheads/Sagar
  budget / credit limits per counterparty
```

Cost terms: freight + bunkers + port dues + expected demurrage − expected despatch + lightering + inventory carrying + stockout penalty + demurrage risk premium.

Solvers: **HiGHS** (free) or OR-Tools CP-SAT for the PoC; Gurobi/CPLEX if the model grows.

The λ·CVaR term is the honest way to express "we want low cost but we cannot tolerate a tail blowout" — which is exactly why a chartering manager wants term contracts in the first place.

### 7.4 Contract-mix output

The headline screen is an **efficient frontier**: expected annual freight cost on one axis, 95th-percentile cost on the other, with each point a different spot/COA/TC split. The manager picks a point on the frontier; the system tells them the implied contracting programme. This directly answers the stated objective.

---

## 8. Technology stack, MLOps and delivery

### 8.1 Stack

| Layer | Choice | Rationale |
|---|---|---|
| Orchestration | **Dagster** (or Airflow) | Asset-based lineage matters when 20 feeds converge |
| Streaming | Kafka / Redpanda | AIS volume |
| Lake | S3/MinIO + Parquet, bronze/silver/gold | Cheap, replayable |
| Warehouse | Postgres + **TimescaleDB** + PostGIS; DuckDB for analyst ad-hoc | Time series + geospatial in one place |
| AIS store | ClickHouse with H3 spatial indexing | Billions of rows, sub-second aggregation |
| Feature store | Feast | Prevents train/serve skew — the #1 silent killer here |
| Data quality | Great Expectations / Soda + explicit data contracts | Feeds break constantly |
| ML | Python, LightGBM, PyTorch, Nixtla stack, MLflow registry | |
| Optimisation | Pyomo + HiGHS / OR-Tools | |
| Serving | FastAPI + Redis | |
| Frontend | React + TypeScript, ECharts/Plotly, deck.gl for the AIS map | |
| Monitoring | Evidently (drift), Prometheus + Grafana (ops) | |
| Deploy | Docker → Kubernetes; single VM acceptable for PoC | |

### 8.2 Retraining and governance

- Weekly scheduled retrain; automatic rollback if walk-forward metrics degrade beyond threshold
- Drift monitors on inputs *and* on the residual distribution
- Every recommendation is versioned and stored with its inputs, model hash, and the counterfactual "what the desk did instead" — an **immutable decision ledger**
- Model cards documenting known failure modes (structural breaks, thin-liquidity routes, new port infrastructure)

### 8.3 Dashboard — six screens

1. **Market cockpit** — fan charts by class, FFA curve overlaid, regime badge, "signal strength" indicator that goes grey when the model has no edge
2. **Chartering advisor** — input cargo volume, grade, origin options, discharge port options, required window → ranked feasible options with cost distribution and a plain-English feasibility explanation ("Kamsarmax rejected: Haldia max draft 8.5 m against required 12.4 m; feasible via Sandheads lightering at +$4.10/t")
3. **Contract portfolio planner** — the efficient frontier and the implied contracting programme
4. **Port watch** — live queue, predicted waiting time, tidal window calendar, cyclone risk gauge
5. **Fleet & idle** — vessels on TC, ballast positions, backhaul candidates, deadheading cost
6. **Decision ledger & explainability** — SHAP attributions, past recommendations vs outcomes, realised savings

### 8.4 Roadmap

| Phase | Duration | Output |
|---|---|---|
| **0. Discovery** | 2 weeks | Data audit, digitise internal fixture history, port constraint master, success metrics agreed with the chartering desk |
| **1. Data foundation** | 5 weeks | Ingestion pipelines live, AIS voyage reconstruction, berthing-programme archive started, baseline naive + FFA benchmarks |
| **2. Forecasting** | 6 weeks | Quantile models, congestion model, conformal calibration, walk-forward backtest report |
| **3. Constraint & optimisation** | 5 weeks | Loadable-tonnes engine, feasibility screening, deterministic then stochastic MILP |
| **4. Application** | 4 weeks | Dashboard, alerts, decision ledger |
| **5. Shadow mode** | 8 weeks | System runs in parallel; recommendations logged but not executed; measured against desk |
| **6. Assisted live** | ongoing | Human-in-the-loop, then progressive autonomy on low-value decisions |

**Hackathon / 48-hour MVP subset:** free AIS (AISStream.io) + archived Baltic proxy + HBA + World Bank coal + scraped Paradip/Vizag berthing programmes → a LightGBM quantile forecaster on BDI/BSI, the loadable-tonnes calculator (fully demoable and genuinely impressive), and a deterministic vessel-recommendation MILP with a Streamlit front end. Skip the stochastic layer; show the frontier on synthetic scenarios.

### 8.5 Team

1 product owner **who has actually chartered ships** (non-negotiable), 2 data scientists, 1 operations-research engineer, 2 data engineers, 1 full-stack developer, 0.5 designer. The domain hire is the highest-leverage role in the project.

### 8.6 Success metrics

*Model:* MASE < 1.0 vs seasonal naive at 4-week horizon; calibrated 80% intervals (78–82% empirical coverage); beat FFA curve at ≤ 4 weeks.

*Business:* realised $/t versus a published index benchmark; share of annual volume on term contracts (the stated objective); demurrage spend; average pre-berthing detention; idle/ballast days; number of vessel-size mismatches.

---

## 9. Novel ideas — where to go beyond the brief

Ordered roughly by value-to-effort.

### 9.1 Predict the *basis*, not the level — and benchmark against the FFA curve
Rather than forecasting the Supramax rate, forecast the spread between your actual achievable ECI rate and the published Baltic route (e.g. S8_58). The spread is far more stationary and far more predictable than the level, and it is the part that is specific to *you* — your ports, your parcel sizes, your counterparties. Bonus: this makes the FFA market usable as a hedge. If a physical COA is priced above fair value, buy the paper instead. **Very few industrial charterers in India do this, and it converts the model from "advice" into an executable financial strategy.**

### 9.2 A proprietary Bay of Bengal tonnage-supply index
From AIS, compute daily: ballast vessels by class within 7/14/21 days' steaming of each of your load regions, and congestion-absorbed tonnage across the Indian Ocean. Publish it internally as a leading indicator. This is the one genuinely non-public signal you can build, it is cheap once AIS is flowing, and it is the kind of thing brokers will start asking *you* about.

### 9.3 Tidal- and density-aware dynamic draft optimisation
Treat draft as a time-varying, stochastic quantity rather than a constant in a spreadsheet. Optimise the declared sailing draft against tide tables, seasonal siltation in the Hooghly, and freshwater allowance on the Mississippi. Squeezing 1–2% more cargo onto every vessel is a permanent, compounding saving that requires no forecasting skill at all — it is pure engineering. Start here; it funds the rest of the project.

### 9.4 Model lightering as a first-class decision, not an exception
SMP Kolkata invested in transloading infrastructure at Sagar and Sandheads specifically so that Capesize parcels can be broken down by floating crane into barges for Haldia — one such operation discharged 66,000 t of steam coal from a wide-beam vessel working to a 9.2 m draft. Sandheads has 40–50 m of water; Sagar anchorage 9–10 m; Haldia Dock itself is roughly 8.5 m and 230 m LOA. That means for any east-coast parcel there is a genuine **economic trade-off between one Capesize plus lightering versus three Supramaxes direct** — and it depends on freight rates, floating-crane availability, barge cycle time and monsoon conditions, all of which move. Nobody optimises this systematically. Make it an explicit branch in the MILP and you have a lever competitors do not have.

### 9.5 Cyclone-conditioned stochastic port availability
The Bay of Bengal cyclone season (Oct–Dec, with a secondary Apr–May peak) closes east-coast berths every year, and Gopalpur, Dhamra and Paradip are directly exposed. Take IMD and ECMWF *ensemble* forecasts, map track probability onto each port's historical closure response, and produce a probability distribution of berth downtime 5–10 days ahead. Feed that into the laycan and demurrage terms of the optimiser. Extend the same logic to Kalimantan rainfall (anchorage loading stops in heavy rain) and Queensland cyclones. This turns weather from an excuse into a priced risk.

### 9.6 Optimise landed cost per Gcal, jointly with blending — not freight per tonne
This is the biggest conceptual upgrade available. For thermal coal, the objective is not cheapest freight; it is cheapest usable energy delivered to the boiler, subject to blend specifications on GCV, ash, sulphur and moisture. Low-CV Indonesian coal has a shorter voyage and cheaper freight per tonne but you need more tonnes per Gcal — and more tonnes means more berth-hours, more handling cost, and more stockpile. Coupling the freight optimiser to the coal-blending LP means the system can say: *"Australian high-CV via Gangavaram is $3/t more freight but $0.31/Gcal cheaper delivered, and frees 1.4 berth-days."* Freight teams and fuel teams almost never optimise jointly. **This is where the real money is, and it is a strictly larger prize than the problem statement asks for.**

### 9.7 Decision-focused learning
Train the forecaster with a loss aligned to downstream *decision regret* rather than MSE — the Smart Predict-then-Optimise (SPO+) framework. A 5% rate error that doesn't change which vessel you fix costs nothing; a 1% error right at a decision boundary costs a lot. Standard training treats these identically. Implementing SPO+ over the MILP is genuinely research-grade work and is a strong differentiator for a hackathon submission.

### 9.8 Price the optionality inside a COA using real-options methods
A COA is a bundle of options: volume tolerance, laycan spread, port options, extension rights. Monte Carlo those options over your simulated freight paths and you can put a dollar value on each clause — giving negotiators a defensible walk-away number instead of a gut feel. *"Widening laycan from 10 to 15 days is worth $0.42/t to us; the owner is asking $0.90."* This turns the model into a negotiating instrument, which is what actually gets it adopted by the chartering desk.

### 9.9 LLM extraction layer for unstructured market intelligence
The single biggest data gap is that ECI lane fixtures are not in any structured database. Build an LLM pipeline that parses broker circulars (PDF/email), daily port berthing programme PDFs, agent line-up reports, WhatsApp broker chatter and trade press into a structured fixtures and events table, with confidence scores and human review for low-confidence extractions. Layer a sentiment/topic index on top. This is the cheapest way to build a proprietary dataset that no vendor sells, and after 12 months you have something genuinely valuable. Pair with RAG so a manager can ask "what have brokers said about Paradip congestion this month?"

### 9.10 Discrete-event digital twin of the East Coast port system
Simulate berths, tides, cranes, barges, rakes and stockyards. Two payoffs: counterfactual policy testing ("what if we shifted 20% of Paradip volume to Dhamra?") without touching live operations, and synthetic generation of rare-event training data that history is too short to provide.

### 9.11 Backhaul and round-voyage TCE optimisation
The brief's "idle scenario management" is usually solved badly by looking at legs in isolation. India's east coast has natural outbound cargo — iron ore, alumina, bauxite, granite, rice, steel. Build a cargo-matching engine that optimises *round-voyage* TCE, not per-leg freight. On a period time charter this is where the surplus is, and the same engine can be exposed as an internal marketplace between business units.

### 9.12 Consortium fixture data pool with other Indian importers
Individually, no Indian buyer has enough ECI fixture history to train a good lane model. Collectively — power utilities, steel producers, cement — they have more than any broker sees. A privacy-preserving pool (secure aggregation, or a trusted third party publishing only aggregate indices) would create an ECI freight benchmark that does not currently exist. Ambitious, but it is the kind of structural move that changes the market rather than just navigating it.

### 9.13 CII- and decarbonisation-aware vessel screening
Vessel CII ratings increasingly constrain speed and availability; poorly rated tonnage is cheaper but slower and carries redelivery risk. Screening candidates on CII, and modelling the resulting speed/consumption profile in the TCE calculation, will matter more each year and costs little to add now.

### 9.14 A permanent counterfactual ledger
Every recommendation the system makes is stored alongside what the desk actually did and what it cost. After six months you have an unarguable, auditable record of value added or destroyed. This is both the honest way to evaluate the project and, practically, the thing that secures the next budget cycle.

---

## 10. Principal risks

| Risk | Mitigation |
|---|---|
| **Data licensing cost** — Baltic + Kpler + Argus can run to six figures annually | Start with free proxies and your own fixture history; build the constraint engine (which needs no paid data) first to prove value and unlock budget |
| **Overpromising forecast accuracy** | Report against the FFA curve from day one; surface a "no edge" state in the UI rather than always emitting a confident number |
| **Structural breaks** (Red Sea, Panama drought, sanctions, Indonesian export policy, a new Indian import duty) | Regime-conditional models; scenario overrides; explicit human override with logged reasoning |
| **Thin internal fixture data** | LLM extraction pipeline (§9.9) plus transfer learning from published Baltic routes |
| **AIS vendor consolidation and pricing power** | Adapter-pattern ingestion; maintain a fallback free feed; contractual continuity clauses |
| **Port data is unstructured and changes without notice** | Scrape and archive daily from day one; version the port constraint master; alert on circular changes |
| **Adoption failure** — the desk ignores it | Shadow mode, explainability, the negotiation-support framing (§9.8), and a domain expert on the build team |
| **Gaming and over-trust** | Decision ledger, mandatory human sign-off above a value threshold, periodic red-team review |

---

## 11. Recommended first 30 days

1. Digitise every fixture the chartering desk has done in the last 5 years into a single normalised table.
2. Build the port constraint master and the loadable-tonnes calculator. Demo it to the desk in week 3.
3. Start the daily berthing-programme scraper and archive — the historical value of this compounds and cannot be bought later.
4. Stand up free-tier AIS ingestion and reconstruct 12 months of voyages into and out of the seven ECI ports.
5. Produce the honest benchmark report: how well does a random walk and the FFA curve predict the rates you actually paid? Everything downstream is judged against that number.
