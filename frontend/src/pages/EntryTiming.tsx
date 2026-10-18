import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  api,
  type CargoGrade,
  type EntryWindow,
  type Forecast,
  type ForecastTarget,
  type Port,
} from "../api";
import { ProvenanceLine } from "../components/Provenance";
import { ModelProvenanceLine } from "../components/ModelProvenance";

const CHART_BASE = {
  backgroundColor: "transparent",
  grid: { left: 56, right: 20, top: 30, bottom: 40 },
  textStyle: { color: "#8b949e", fontFamily: "Inter, sans-serif" },
  tooltip: { trigger: "axis" as const },
};

const USD = (v: number, d = 2) =>
  `$${v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`;

const defaultDeadline = () => {
  const d = new Date();
  d.setDate(d.getDate() + 75);
  return d.toISOString().slice(0, 10);
};

/**
 * The fan chart. History as a line, then the calibrated quantile bands
 * stacked forward from the last observation.
 *
 * The bands are drawn as a stacked area with a transparent floor, which is the
 * standard ECharts idiom for a fan: the q05 series is invisible and each band
 * above it is the *increment* to the next quantile.
 */
function FanChart({ forecast }: { forecast: Forecast }) {
  // A time axis, not a category axis. The horizon grid is 7/14/28/56/91/182
  // days, so on a category axis the step from 91 to 182 days would render the
  // same width as the step from 7 to 14 - making the fan look like it widens
  // far faster than it does. Time-scaling is the honest rendering.
  const history = forecast.history.slice(-200);
  const last = history.length ? history[history.length - 1] : null;

  const at = (level: string): [string, number][] => {
    const points: [string, number][] = forecast.bands.map((b) => [b.date, b.quantiles[level]]);
    // Anchor the fan to the last observation so it does not float free.
    return last ? [[last.date, last.value], ...points] : points;
  };
  const incFrom = (upper: string, lower: string): [string, number][] => {
    const points: [string, number][] = forecast.bands.map((b) => [
      b.date,
      b.quantiles[upper] - b.quantiles[lower],
    ]);
    return last ? [[last.date, 0], ...points] : points;
  };

  const bands = [
    { name: "q05 floor", data: at("0.05"), color: "transparent", floor: true },
    { name: "5-25%", data: incFrom("0.25", "0.05"), color: "rgba(68,147,248,0.10)" },
    { name: "25-75%", data: incFrom("0.75", "0.25"), color: "rgba(68,147,248,0.22)" },
    { name: "75-95%", data: incFrom("0.95", "0.75"), color: "rgba(68,147,248,0.10)" },
  ].map((s) => ({
    name: s.name,
    type: "line" as const,
    stack: "fan",
    smooth: true,
    showSymbol: false,
    lineStyle: { opacity: 0 },
    areaStyle: { color: s.color },
    silent: true,
    data: s.data,
  }));

  const option = {
    ...CHART_BASE,
    tooltip: { trigger: "axis" as const },
    legend: {
      data: ["Observed", "Median forecast", "25-75%", "5-25%"],
      textStyle: { color: "#8b949e" },
      top: 0,
    },
    xAxis: {
      type: "time" as const,
      axisLine: { lineStyle: { color: "#2a3038" } },
    },
    yAxis: {
      type: "value",
      scale: true,
      splitLine: { lineStyle: { color: "#21262d" } },
      axisLabel: { formatter: (v: number) => `$${v}` },
    },
    series: [
      ...bands,
      {
        name: "Observed",
        type: "line",
        smooth: true,
        showSymbol: false,
        data: history.map((p) => [p.date, p.value]),
        itemStyle: { color: "#8b949e" },
        lineStyle: { width: 1.5 },
      },
      {
        name: "Median forecast",
        type: "line",
        smooth: true,
        showSymbol: true,
        symbolSize: 5,
        data: at("0.5"),
        itemStyle: { color: "#4493f8" },
        lineStyle: { width: 2, type: "dashed" as const },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 320 }} notMerge />;
}

/** The spot/term efficient frontier: the stated objective of the brief. */
function FrontierChart({ window }: { window: EntryWindow }) {
  const points = window.contract_mix;
  const option = {
    ...CHART_BASE,
    grid: { left: 64, right: 24, top: 30, bottom: 48 },
    tooltip: {
      trigger: "item" as const,
      formatter: (p: { data: number[] }) =>
        `${(p.data[2] * 100).toFixed(0)}% term<br/>expected ${USD(p.data[0])}/t<br/>CVaR95 ${USD(
          p.data[1]
        )}/t`,
    },
    xAxis: {
      type: "value",
      scale: true,
      name: "Expected cost $/t",
      nameLocation: "middle" as const,
      nameGap: 30,
      splitLine: { lineStyle: { color: "#21262d" } },
    },
    yAxis: {
      type: "value",
      scale: true,
      name: "95th percentile cost $/t",
      nameLocation: "middle" as const,
      nameGap: 44,
      splitLine: { lineStyle: { color: "#21262d" } },
    },
    series: [
      {
        type: "line",
        smooth: true,
        symbolSize: 11,
        data: points.map((p) => [
          p.expected_cost_usd_per_t,
          p.cvar_cost_usd_per_t,
          p.term_share,
        ]),
        itemStyle: {
          color: (p: { data: number[] }) =>
            Math.abs(p.data[2] - window.recommended_term_share) < 1e-6
              ? "#3fb950"
              : "#4493f8",
        },
        lineStyle: { color: "#4493f8", width: 2 },
        label: {
          show: true,
          formatter: (p: { data: number[] }) => `${(p.data[2] * 100).toFixed(0)}%`,
          color: "#8b949e",
          position: "top" as const,
          fontSize: 11,
        },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 300 }} notMerge />;
}

export default function EntryTiming() {
  const [ports, setPorts] = useState<Port[]>([]);
  const [cargoes, setCargoes] = useState<CargoGrade[]>([]);
  const [targets, setTargets] = useState<ForecastTarget[]>([]);

  const [dischargePort, setDischargePort] = useState("INPRT");
  const [loadPort, setLoadPort] = useState("IDTAB");
  const [tonnes, setTonnes] = useState(70000);
  const [cargo, setCargo] = useState("thermal_coal_mid_cv");
  const [deadline, setDeadline] = useState(defaultDeadline());
  const [risk, setRisk] = useState(0.35);

  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [window_, setWindow] = useState<EntryWindow | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.ports(), api.cargoes(), api.forecastTargets()])
      .then(([p, c, t]) => {
        setPorts(p);
        setCargoes(c);
        setTargets(t);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.timingWindow({
        discharge_port: dischargePort,
        load_port: loadPort || null,
        required_tonnes: tonnes,
        cargo,
        not_later_than: deadline,
        risk_aversion: risk,
      });
      setWindow(result);
      setForecast(await api.forecast(result.target_series, 400));
    } catch (e) {
      setError(String(e));
      setWindow(null);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (ports.length && targets.length) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ports.length, targets.length]);

  const discharge = ports.filter((p) => p.role === "discharge");
  const loadPorts = ports.filter((p) => p.role === "load");
  const verdict = (window_?.recommendation ?? "").toLowerCase().replace("_", "");
  const verdictClass =
    verdict === "fixnow" ? "fix" : verdict === "wait" ? "wait" : verdict === "split" ? "split" : "infeasible";

  return (
    <div>
      <div className="page-head">
        <h2>Entry Timing</h2>
        <p>
          When to fix, not just what the rate will be. Fixing is an optimal stopping problem: hold
          today's price, or keep the option and face a distribution next week. The engine solves it
          by backward induction over scenarios, and refuses to time the market at all where the
          forecast has no measured edge over a random walk.
        </p>
      </div>

      <div className="panel">
        <h3>Cargo requirement</h3>
        <div className="form-row">
          <div className="field">
            <label>Load port</label>
            <select value={loadPort} onChange={(e) => setLoadPort(e.target.value)}>
              <option value="">Not specified</option>
              {loadPorts.map((p) => (
                <option key={p.code} value={p.code}>{p.name}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Discharge port</label>
            <select value={dischargePort} onChange={(e) => setDischargePort(e.target.value)}>
              {discharge.map((p) => (
                <option key={p.code} value={p.code}>{p.name}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Quantity (t)</label>
            <input type="number" value={tonnes} step={10000}
              onChange={(e) => setTonnes(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Grade</label>
            <select value={cargo} onChange={(e) => setCargo(e.target.value)}>
              {cargoes.map((c) => (
                <option key={c.name} value={c.name}>{c.name.replace(/_/g, " ")}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Must fix by</label>
            <input type="date" value={deadline} onChange={(e) => setDeadline(e.target.value)} />
          </div>
          <div className="field">
            <label>Risk aversion (lambda)</label>
            <input type="number" value={risk} step={0.05} min={0} max={3}
              onChange={(e) => setRisk(Number(e.target.value))} />
          </div>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Solving..." : "Recommend"}
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {window_ && (
        <>
          <div className="panel">
            <div className={`recommend ${verdictClass}`}>
              <div className={`verdict ${verdictClass}`}>
                {window_.recommendation.replace("_", " ")}
              </div>
              {window_.window_start && window_.recommendation !== "INFEASIBLE" && (
                <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
                  Fixing window {window_.window_start}
                  {window_.window_end !== window_.window_start ? ` to ${window_.window_end}` : ""}
                </div>
              )}
              <div className="rationale">{window_.explanation}</div>
            </div>

            <div className="grid grid-4" style={{ marginTop: 16 }}>
              <div className="stat">
                <div className="label">Fix now, delivered</div>
                <div className="value">{USD(window_.fix_now_cost_usd_per_t)}</div>
                <div className="sub">per tonne, freight + bunkers + demurrage</div>
              </div>
              <div className="stat">
                <div className="label">Value of waiting</div>
                <div className="value">{USD(window_.value_of_waiting_usd_per_t)}</div>
                <div className="sub">
                  {window_.value_of_waiting_usd_total >= 0
                    ? `$${Math.round(window_.value_of_waiting_usd_total).toLocaleString()} on the parcel`
                    : ""}
                </div>
              </div>
              <div className="stat">
                <div className="label">Chance a later date is cheaper</div>
                <div className="value">{(window_.p_better_later * 100).toFixed(0)}%</div>
                <div className="sub">across {window_.assumptions.n_scenarios} scenarios</div>
              </div>
              <div className="stat">
                <div className="label">Regret if wrong</div>
                <div className="value">
                  ${Math.round(window_.regret_if_wrong_usd).toLocaleString()}
                </div>
                <div className="sub">95th percentile of waiting vs fixing today</div>
              </div>
            </div>
            <ModelProvenanceLine m={window_.model} signal={window_.signal_strength} />
          </div>

          {window_.routing && (
            <div className="panel">
              <h3>Routing this prices</h3>
              <div className="option best">
                <div className="option-head">
                  <span className="option-title">{String(window_.routing.vessel_class)}</span>
                  <span className={`badge badge-${String(window_.routing.routing)}`}>
                    {String(window_.routing.routing)}
                  </span>
                  <span className="option-tonnes">
                    {Number(window_.routing.loadable_tonnes ?? 0).toLocaleString()} t
                  </span>
                </div>
                <div className="option-explain">{String(window_.routing.explanation ?? "")}</div>
              </div>
              {window_.waiting_forecast && (
                <div className="option">
                  <div className="option-head">
                    <span className="option-title">Discharge-port detention</span>
                    <span className="badge badge-modelled">modelled</span>
                  </div>
                  <div className="option-explain">{window_.waiting_forecast.explanation}</div>
                </div>
              )}
            </div>
          )}

          {forecast && (
            <div className="panel">
              <h3>Rate outlook &mdash; {forecast.target_key}</h3>
              {forecast.caveat && (
                <div className="option-explain" style={{ marginBottom: 10 }}>
                  {forecast.caveat}
                </div>
              )}
              <FanChart forecast={forecast} />
              <ModelProvenanceLine m={forecast.model} signal={forecast.signal_strength} />
              <ProvenanceLine p={forecast.provenance} />
            </div>
          )}

          <div className="grid grid-2">
            <div className="panel">
              <h3>Contract mix: spot against term</h3>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 0, lineHeight: 1.6 }}>
                The problem statement's actual objective: how much volume to move off single spot
                fixtures onto term cover. Each point is a split; going right costs more on average
                and buys a lower tail. The green point minimises the risk-adjusted objective at
                lambda&nbsp;=&nbsp;{String(window_.assumptions.risk_aversion_lambda)}.
              </p>
              <FrontierChart window={window_} />
            </div>

            <div className="panel">
              <h3>Decision calendar</h3>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th className="num">Median rate</th>
                      <th className="num">Expected cost</th>
                      <th className="num">CVaR 95</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {window_.decision_path.map((p) => (
                      <tr key={p.date}>
                        <td>{p.date}</td>
                        <td className="num">{USD(p.freight_median)}</td>
                        <td className="num">{USD(p.expected_cost_usd_per_t)}</td>
                        <td className="num">{USD(p.cvar_cost_usd_per_t)}</td>
                        <td>
                          {p.fix_now_preferred ? (
                            <span className="badge badge-strong">fix</span>
                          ) : (
                            <span className="badge badge-noedge">hold</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="panel">
            <h3>Assumptions and method</h3>
            <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.7, marginTop: 0 }}>
              Every figure below is an explicit input with a disclosed default rather than an
              invented number, in the same way the loadable-tonnes engine takes tide height. See
              SYNTHETIC_DATA.md.
            </p>
            <div className="workings">
              {Object.entries(window_.assumptions).map(([k, v]) => (
                <div key={k}>
                  <span>{k.replace(/_/g, " ")}</span>
                  <br />
                  {v === null
                    ? "not measured"
                    : typeof v === "number" && !Number.isInteger(v)
                      ? v.toFixed(4)
                      : String(v)}
                </div>
              ))}
            </div>
            <ul className="muted" style={{ fontSize: 12.5, lineHeight: 1.7 }}>
              {window_.scenario_notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}
