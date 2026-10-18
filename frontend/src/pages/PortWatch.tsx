import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { api, type Port, type PortWatch as PortWatchData, type QueueHistory } from "../api";
import { ProvenanceLine } from "../components/Provenance";

const CHART_BASE = {
  backgroundColor: "transparent",
  grid: { left: 48, right: 20, top: 30, bottom: 40 },
  textStyle: { color: "#8b949e", fontFamily: "Inter, sans-serif" },
  tooltip: { trigger: "axis" as const },
};

export default function PortWatch() {
  const [ports, setPorts] = useState<Port[]>([]);
  const [code, setCode] = useState("INPRT");
  const [watch, setWatch] = useState<PortWatchData | null>(null);
  const [history, setHistory] = useState<QueueHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.ports().then(setPorts).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([api.portWatch(code), api.queueHistory(code, 180)])
      .then(([w, h]) => {
        setWatch(w);
        setHistory(h);
      })
      .catch((e) => {
        setError(String(e));
        setWatch(null);
        setHistory(null);
      })
      .finally(() => setLoading(false));
  }, [code]);

  const historyOption = history && {
    ...CHART_BASE,
    legend: { data: ["Working", "Waiting", "Expected"], textStyle: { color: "#8b949e" }, top: 0 },
    xAxis: {
      type: "category",
      data: history.points.map((p) => p.date),
      axisLine: { lineStyle: { color: "#2a3038" } },
    },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#21262d" } } },
    series: [
      {
        name: "Working", type: "line", smooth: true, showSymbol: false,
        data: history.points.map((p) => p.working), itemStyle: { color: "#3fb950" },
      },
      {
        name: "Waiting", type: "line", smooth: true, showSymbol: false,
        data: history.points.map((p) => p.waiting), itemStyle: { color: "#d29922" },
        areaStyle: { color: "rgba(210,153,34,0.12)" },
      },
      {
        name: "Expected", type: "line", smooth: true, showSymbol: false,
        data: history.points.map((p) => p.expected), itemStyle: { color: "#4493f8" },
      },
    ],
  };

  const waiting = watch?.queue.filter((q) => q.status === "waiting") ?? [];

  return (
    <div>
      <div className="page-head">
        <h2>Port Watch</h2>
        <p>
          Live queue from the port authority daily berthing programme, scraped and archived every
          day. The history below is real observed data, not a model.
        </p>
      </div>

      <div className="panel">
        <div className="form-row">
          <div className="field">
            <label>Port</label>
            <select value={code} onChange={(e) => setCode(e.target.value)}>
              {ports
                .filter((p) => p.role === "discharge")
                .map((p) => (
                  <option key={p.code} value={p.code}>{p.name}</option>
                ))}
            </select>
          </div>
        </div>
      </div>

      {loading && <div className="loading">Loading...</div>}
      {error && (
        <div className="panel">
          <div className="error">No archived berthing programme for this port yet.</div>
          <div className="muted" style={{ fontSize: 12 }}>
            Only Paradip is currently scraped. The archive begins the day the scraper starts, so
            depth accumulates rather than being backfillable for ports whose publisher overwrites
            filenames.
          </div>
        </div>
      )}

      {watch && (
        <>
          <div className="grid grid-4" style={{ marginBottom: 16 }}>
            <div className="stat">
              <div className="label">At berth</div>
              <div className="value" style={{ color: "var(--good)" }}>{watch.working}</div>
              <div className="sub">working now</div>
            </div>
            <div className="stat">
              <div className="label">Waiting</div>
              <div className="value" style={{ color: "var(--warn)" }}>{watch.waiting}</div>
              <div className="sub">at anchorage</div>
            </div>
            <div className="stat">
              <div className="label">Expected</div>
              <div className="value" style={{ color: "var(--accent)" }}>{watch.expected}</div>
              <div className="sub">declared ETA</div>
            </div>
            <div className="stat">
              <div className="label">Median wait</div>
              <div className="value">
                {watch.median_waiting_days !== null ? `${watch.median_waiting_days}d` : "n/a"}
              </div>
              <div className="sub">pre-berthing detention</div>
            </div>
          </div>

          {historyOption && (
            <div className="panel">
              <h3>Queue history - {history?.points.length} observed days</h3>
              <ReactECharts option={historyOption} style={{ height: 300 }} />
              <ProvenanceLine p={watch.provenance} />
            </div>
          )}

          <div className="panel">
            <h3>Vessels waiting at anchorage ({waiting.length})</h3>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Vessel</th>
                    <th>Cargo</th>
                    <th className="num">Quantity (t)</th>
                    <th>For berth</th>
                    <th>Arrived</th>
                    <th className="num">Waiting (days)</th>
                    <th className="num">LOA (m)</th>
                  </tr>
                </thead>
                <tbody>
                  {waiting.map((q, i) => (
                    <tr key={`${q.vessel_name}-${i}`}>
                      <td>{q.vessel_name}</td>
                      <td className="muted">{q.cargo ?? "-"}</td>
                      <td className="num">
                        {q.cargo_tonnes ? q.cargo_tonnes.toLocaleString() : "-"}
                      </td>
                      <td className="muted">{q.berth ?? "-"}</td>
                      <td className="muted">{q.arrival_date ?? "-"}</td>
                      <td className="num" style={{
                        color: (q.waiting_days ?? 0) > 7 ? "var(--bad)" : undefined,
                      }}>
                        {q.waiting_days ?? "-"}
                      </td>
                      <td className="num muted">{q.loa_m ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ProvenanceLine p={watch.provenance} />
          </div>
        </>
      )}
    </div>
  );
}
