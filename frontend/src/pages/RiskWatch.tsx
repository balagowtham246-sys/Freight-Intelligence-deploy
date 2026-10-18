import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { api, type Alert, type Monitor, type QueueHistory, type RiskBoard, type WaitingForecast } from "../api";
import { ProvenanceLine } from "../components/Provenance";

const CHART_BASE = {
  backgroundColor: "transparent",
  grid: { left: 48, right: 20, top: 30, bottom: 40 },
  textStyle: { color: "#8b949e", fontFamily: "Inter, sans-serif" },
  tooltip: { trigger: "axis" as const },
};

const SEVERITY_LABEL: Record<string, string> = {
  critical: "critical",
  warning: "warning",
  watch: "watch",
  info: "info",
};

function AlertCard({ alert }: { alert: Alert }) {
  return (
    <div className={`alert ${alert.severity}`}>
      <div className="alert-head">
        <span className="alert-title">{alert.title}</span>
        <span className={`badge badge-${alert.severity}`}>
          {SEVERITY_LABEL[alert.severity] ?? alert.severity}
        </span>
        <span className="badge badge-noedge">{alert.code.replace(/_/g, " ").toLowerCase()}</span>
        <span className="muted" style={{ marginLeft: "auto", fontSize: 11.5 }}>
          raised {alert.raised_date}
          {alert.cleared_date ? ` · cleared ${alert.cleared_date}` : ""}
        </span>
      </div>
      <div className="alert-body">{alert.body}</div>
      <div className="alert-action">
        <b>What this means for chartering:</b> {alert.action}
      </div>
    </div>
  );
}

function MonitorRow({ monitor }: { monitor: Monitor }) {
  return (
    <div className="monitor">
      <div className="m-label">{monitor.label}</div>
      <div className="m-value">{monitor.display}</div>
      <div className="m-thresh">trips at {monitor.threshold_label}</div>
      <div>
        {monitor.triggered ? (
          <span className="badge badge-warning">firing</span>
        ) : (
          <span className="badge badge-real">clear</span>
        )}
      </div>
    </div>
  );
}

export default function RiskWatch() {
  const [board, setBoard] = useState<RiskBoard | null>(null);
  const [history, setHistory] = useState<QueueHistory | null>(null);
  const [waiting, setWaiting] = useState<WaitingForecast | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.risk(),
      api.queueHistory("INPRT", 180),
      api.waitingForecast("INPRT", 70000).catch(() => null),
    ])
      .then(([r, h, w]) => {
        setBoard(r);
        setHistory(h);
        setWaiting(w);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const queueOption = history && {
    ...CHART_BASE,
    legend: { data: ["Waiting", "Working"], textStyle: { color: "#8b949e" }, top: 0 },
    xAxis: {
      type: "category",
      data: history.points.map((p) => p.date),
      axisLine: { lineStyle: { color: "#2a3038" } },
    },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#21262d" } } },
    series: [
      {
        name: "Waiting",
        type: "line",
        smooth: true,
        showSymbol: false,
        data: history.points.map((p) => p.waiting),
        itemStyle: { color: "#d29922" },
        areaStyle: { color: "rgba(210,153,34,0.12)" },
      },
      {
        name: "Working",
        type: "line",
        smooth: true,
        showSymbol: false,
        data: history.points.map((p) => p.working),
        itemStyle: { color: "#3fb950" },
      },
    ],
  };

  const survivalOption = waiting && {
    ...CHART_BASE,
    tooltip: {
      trigger: "axis" as const,
      formatter: (params: { dataIndex: number; data: number }[]) =>
        `Day ${params[0].dataIndex + 1}<br/>${(params[0].data * 100).toFixed(0)}% still waiting`,
    },
    xAxis: {
      type: "category",
      data: Array.from({ length: 30 }, (_, i) => String(i + 1)),
      name: "days at anchorage",
      nameLocation: "middle" as const,
      nameGap: 26,
      axisLine: { lineStyle: { color: "#2a3038" } },
    },
    yAxis: {
      type: "value",
      max: 1,
      splitLine: { lineStyle: { color: "#21262d" } },
      axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
    },
    series: [
      {
        name: "Still waiting",
        type: "line",
        smooth: true,
        showSymbol: false,
        // Defensive: a missing optional field should degrade to an empty
        // chart, never white-screen the whole risk board.
        data: waiting.survival?.slice(0, 30) ?? [],
        itemStyle: { color: "#4493f8" },
        areaStyle: { color: "rgba(68,147,248,0.15)" },
      },
    ],
  };

  const active = board?.active ?? [];
  const critical = active.filter((a) => a.severity === "critical").length;
  const warnings = active.filter((a) => a.severity === "warning").length;

  return (
    <div>
      <div className="page-head">
        <h2>Risk Watch</h2>
        <p>
          Early warning of market volatility, port congestion and disruption. Every alert carries
          the numbers behind it and what it implies for a chartering decision &mdash; an alert
          without an action is noise. Alerts are stored and cleared rather than recomputed, so each
          one is a prediction whose accuracy can be checked later.
        </p>
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Evaluating risk rules...</div>}

      {board && (
        <>
          <div className="grid grid-4">
            <div className="stat">
              <div className="label">Active alerts</div>
              <div className="value">{active.length}</div>
              <div className="sub">as of {board.as_of}</div>
            </div>
            <div className="stat">
              <div className="label">Critical</div>
              <div className="value" style={{ color: critical ? "var(--bad)" : undefined }}>
                {critical}
              </div>
              <div className="sub">require a decision today</div>
            </div>
            <div className="stat">
              <div className="label">Warnings</div>
              <div className="value" style={{ color: warnings ? "var(--warn)" : undefined }}>
                {warnings}
              </div>
              <div className="sub">price into the next fixture</div>
            </div>
            <div className="stat">
              <div className="label">Monitors running</div>
              <div className="value">{board.monitors.length}</div>
              <div className="sub">
                {board.monitors.filter((m) => m.triggered).length} firing
              </div>
            </div>
          </div>

          <div className="panel">
            <h3>Active alerts</h3>
            {active.length === 0 ? (
              <div className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
                Nothing is firing. That is a reading, not an absence of one &mdash; every monitor
                below is live and below its threshold. The board is what distinguishes "nothing is
                wrong" from "nothing is being watched".
              </div>
            ) : (
              active.map((a) => <AlertCard key={a.id} alert={a} />)
            )}
          </div>

          <div className="panel">
            <h3>Monitor board</h3>
            {board.monitors.map((m, i) => (
              <MonitorRow key={`${m.code}-${i}`} monitor={m} />
            ))}
          </div>

          <div className="grid grid-2">
            <div className="panel">
              <h3>Paradip queue &mdash; observed</h3>
              {queueOption && <ReactECharts option={queueOption} style={{ height: 260 }} notMerge />}
              <p className="muted" style={{ fontSize: 12, lineHeight: 1.6 }}>
                Real, scraped daily from the port authority berthing programme. The congestion
                z-score is measured against this series' own history, not an absolute level.
              </p>
            </div>

            <div className="panel">
              <h3>Pre-berthing detention &mdash; modelled</h3>
              {survivalOption && (
                <ReactECharts option={survivalOption} style={{ height: 260 }} notMerge />
              )}
              {waiting && (
                <>
                  <p className="muted" style={{ fontSize: 12, lineHeight: 1.6 }}>
                    {waiting.explanation}
                  </p>
                  <ProvenanceLine p={waiting.provenance} />
                </>
              )}
            </div>
          </div>

          {board.recently_cleared.length > 0 && (
            <div className="panel">
              <h3>Recently cleared ({board.recently_cleared.length})</h3>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 0 }}>
                Kept so the precision of each rule is measurable. A rule that fires often and
                clears quickly with nothing happening is a rule to retune.
              </p>
              {board.recently_cleared.map((a) => (
                <AlertCard key={a.id} alert={a} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
