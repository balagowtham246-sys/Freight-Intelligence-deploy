/** Typed client for the platform API. */

export interface Provenance {
  source_name: string;
  source_url: string | null;
  retrieved_at: string | null;
  is_synthetic: boolean;
}

export interface Berth {
  name: string;
  max_draft_m: number;
  max_loa_m?: number;
  max_beam_m?: number;
  max_dwt?: number;
  requires_gear?: boolean;
  discharge_rate_tpd?: number;
  cargo_types?: string;
  source?: string;
}

export interface Port {
  code: string;
  name: string;
  country: string;
  role: string;
  latitude: number;
  longitude: number;
  water_density: number | null;
  tidal_range_m: number | null;
  tide_constrained: boolean;
  source: string | null;
  notes: string | null;
  berths: Berth[];
}

export interface VesselClass {
  name: string;
  dwt: number;
  loa_m: number;
  beam_m: number;
  summer_draft_m: number;
  lightship_t: number;
  tpc: number;
  grain_cubic_m3: number;
  geared: boolean;
}

export interface CargoGrade {
  name: string;
  category: string;
  stowage_factor_m3_per_t: number;
  gcv_kcal_per_kg: number;
}

export interface RoutingOption {
  routing: "direct" | "lightering";
  vessel_class: string;
  discharge_port: string;
  berth_name: string;
  loadable_tonnes: number;
  limiting_factor: string;
  feasible: boolean;
  explanation: string;
  transhipment_port: string | null;
  daughter_class: string | null;
  daughter_voyages: number | null;
  vessels_required: number;
  total_lifted_t: number;
  detail: Record<string, unknown> & { workings?: Record<string, number> };
}

export interface FeasibilityResult {
  discharge_port: string;
  load_port: string | null;
  required_tonnes: number;
  cargo: string;
  feasible: RoutingOption[];
  rejected: RoutingOption[];
  summary: string;
  provenance: Provenance;
}

export interface QueueEntry {
  vessel_name: string;
  status: string;
  cargo: string | null;
  cargo_tonnes: number | null;
  berth: string | null;
  arrival_date: string | null;
  waiting_days: number | null;
  loa_m: number | null;
  draft_m: number | null;
}

export interface PortWatch {
  port_code: string;
  snapshot_date: string;
  working: number;
  waiting: number;
  expected: number;
  median_waiting_days: number | null;
  queue: QueueEntry[];
  provenance: Provenance;
}

export interface QueueHistory {
  port_code: string;
  points: { date: string; working: number; waiting: number; expected: number }[];
}

export interface SeriesInfo {
  series_code: string;
  series_kind: string;
  unit: string;
  observations: number;
  first: string;
  last: string;
  is_synthetic: boolean;
}

export interface Series {
  series_code: string;
  unit: string;
  points: { date: string; value: number }[];
  provenance: Provenance;
}

export interface CoalStock {
  obs_date: string;
  plant_count: number;
  median_days_of_burn: number | null;
  critical_count: number;
  plants: {
    plant_name: string;
    state: string | null;
    region: string | null;
    capacity_mw: number | null;
    stock_kt: number | null;
    days_of_burn: number | null;
  }[];
  provenance: Provenance;
}

export interface ProvenanceSummary {
  total_rows: number;
  real_rows: number;
  synthetic_rows: number;
  pct_real: number | null;
  tables: Record<string, { source_name: string; is_synthetic: boolean; rows: number }[]>;
}

export interface Freshness {
  adapter: string;
  status: string;
  rows_written: number;
  finished_at: string | null;
  used_cache: boolean;
  message: string | null;
}


/** Attribution for a modelled figure. Deliberately distinct from Provenance:
 *  a forecast is neither a fetched observation nor a fabricated feed. */
export interface ModelProvenance {
  model_name: string;
  model_version: string;
  trained_through: string;
  n_train: number;
  input_series: string[];
  trained_on_synthetic: boolean;
}

export interface ForecastBand {
  horizon_days: number;
  date: string;
  quantiles: Record<string, number>;
}

export interface HorizonSkill {
  horizon_days: number;
  n_folds: number;
  point_skill: number | null;
  distributional_skill: number | null;
  mase: number | null;
  coverage_80: number | null;
  has_edge: boolean;
  computed_on_synthetic: boolean;
}

export interface Forecast {
  target_key: string;
  unit: string;
  history: { date: string; value: number }[];
  bands: ForecastBand[];
  skill: HorizonSkill[];
  signal_strength: string;
  caveat: string | null;
  model: ModelProvenance;
  provenance: Provenance;
}

export interface ForecastTarget {
  series_code: string;
  series_kind: string;
  unit: string;
  observations: number;
  first: string;
  last: string;
  is_synthetic: boolean;
  forecastable: boolean;
  reason: string | null;
}

export interface BacktestDetail {
  horizon_days: number;
  metric: string;
  value: number;
  baseline_name: string | null;
  baseline_value: number | null;
  skill: number | null;
  n_folds: number;
}

export interface Backtest {
  target_key: string;
  computed_on_synthetic: boolean;
  ffa_benchmark: string;
  horizons: HorizonSkill[];
  detail: BacktestDetail[];
}

export interface WaitingForecast {
  port_code: string;
  queue_length: number;
  expected_days: number;
  median_days: number;
  p80_days: number;
  p95_days: number;
  survival: number[];
  model: string;
  n_train_calls: number;
  explanation: string;
  provenance: Provenance;
}

export interface DecisionPoint {
  date: string;
  horizon_days: number;
  freight_median: number;
  freight_p10: number;
  freight_p90: number;
  expected_cost_usd_per_t: number;
  cvar_cost_usd_per_t: number;
  fix_now_preferred: boolean;
}

export interface ContractMixPoint {
  term_share: number;
  expected_cost_usd_per_t: number;
  cvar_cost_usd_per_t: number;
}

export interface EntryWindow {
  recommendation: string;
  signal_strength: string;
  window_start: string | null;
  window_end: string | null;
  value_of_waiting_usd_per_t: number;
  value_of_waiting_usd_total: number;
  p_better_later: number;
  regret_if_wrong_usd: number;
  fix_now_cost_usd_per_t: number;
  best_expected_cost_usd_per_t: number;
  recommended_term_share: number;
  decision_path: DecisionPoint[];
  contract_mix: ContractMixPoint[];
  routing: Record<string, unknown> | null;
  assumptions: Record<string, number | null>;
  scenario_notes: string[];
  explanation: string;
  target_series: string;
  waiting_forecast: WaitingForecast | null;
  model: ModelProvenance;
}

export interface Alert {
  id: number;
  code: string;
  severity: string;
  scope: string;
  port_code: string | null;
  series_code: string | null;
  raised_date: string;
  cleared_date: string | null;
  last_seen_date: string;
  title: string;
  body: string;
  action: string;
  evidence: Record<string, unknown>;
}

export interface Monitor {
  code: string;
  label: string;
  value: number | null;
  display: string;
  threshold: number | null;
  threshold_label: string;
  triggered: boolean;
}

export interface RiskBoard {
  as_of: string;
  active: Alert[];
  recently_cleared: Alert[];
  monitors: Monitor[];
}

const rawBase =
  (import.meta.env.VITE_API_URL as string | undefined) ||
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  "/api";
const BASE = rawBase.endsWith("/") ? rawBase.slice(0, -1) : rawBase;

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${path}`);
  }
  return response.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  ports: () => get<Port[]>("/reference/ports"),
  vessels: () => get<VesselClass[]>("/reference/vessels"),
  cargoes: () => get<CargoGrade[]>("/reference/cargoes"),
  screen: (body: {
    discharge_port: string;
    required_tonnes: number;
    load_port?: string | null;
    cargo: string;
    tide_height_m?: number;
    bunkers_rob_t?: number;
  }) => post<FeasibilityResult>("/feasibility/screen", body),
  portWatch: (code: string) => get<PortWatch>(`/portwatch/${code}`),
  queueHistory: (code: string, days = 120) =>
    get<QueueHistory>(`/portwatch/${code}/history?days=${days}`),
  seriesList: () => get<SeriesInfo[]>("/series"),
  series: (code: string, since?: string) =>
    get<Series>(`/series/${code}${since ? `?since=${since}` : ""}`),
  coalStock: () => get<CoalStock>("/demand/coal-stock"),
  provenanceSummary: () => get<ProvenanceSummary>("/meta/provenance-summary"),
  freshness: () => get<Freshness[]>("/meta/freshness"),

  forecastTargets: () => get<ForecastTarget[]>("/forecast/targets"),
  forecast: (code: string, historyDays = 365) =>
    get<Forecast>(`/forecast/${code}?history_days=${historyDays}`),
  backtest: (code: string) => get<Backtest>(`/forecast/${code}/backtest`),
  waitingForecast: (code: string, tonnes = 50000) =>
    get<WaitingForecast>(`/portwatch/${code}/waiting-forecast?tonnes=${tonnes}`),
  timingWindow: (body: {
    discharge_port: string;
    required_tonnes: number;
    cargo: string;
    load_port?: string | null;
    not_later_than: string;
    target_series?: string | null;
    risk_aversion?: number;
    demurrage_usd_per_day?: number;
    bunker_usd_per_tonne?: number;
    n_scenarios?: number;
  }) => post<EntryWindow>("/timing/window", body),
  risk: () => get<RiskBoard>("/risk/alerts"),
};
