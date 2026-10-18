import { useEffect, useState } from "react";
import { api, type Freshness, type Port, type ProvenanceSummary } from "../api";

export default function DataProvenance() {
  const [summary, setSummary] = useState<ProvenanceSummary | null>(null);
  const [freshness, setFreshness] = useState<Freshness[]>([]);
  const [ports, setPorts] = useState<Port[]>([]);

  useEffect(() => {
    api.provenanceSummary().then(setSummary).catch(() => setSummary(null));
    api.freshness().then(setFreshness).catch(() => setFreshness([]));
    api.ports().then(setPorts).catch(() => setPorts([]));
  }, []);

  return (
    <div>
      <div className="page-head">
        <h2>Data Provenance</h2>
        <p>
          A live count of what the platform actually holds. The ingestion layer refuses to write a
          synthetic row whose source is not declared in SYNTHETIC_DATA.md, so this page is an
          audit rather than an assertion.
        </p>
      </div>

      {summary && (
        <div className="grid grid-3" style={{ marginBottom: 16 }}>
          <div className="stat">
            <div className="label">Total rows</div>
            <div className="value">{summary.total_rows.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="label">Real</div>
            <div className="value" style={{ color: "var(--good)" }}>
              {summary.real_rows.toLocaleString()}
            </div>
            <div className="sub">{summary.pct_real}% of all data</div>
          </div>
          <div className="stat">
            <div className="label">Synthetic</div>
            <div className="value" style={{ color: summary.synthetic_rows ? "var(--warn)" : "var(--muted)" }}>
              {summary.synthetic_rows.toLocaleString()}
            </div>
            <div className="sub">
              {summary.synthetic_rows === 0 ? "none in use" : "see SYNTHETIC_DATA.md"}
            </div>
          </div>
        </div>
      )}

      {summary && (
        <div className="panel">
          <h3>Rows by source</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Table</th>
                  <th>Source</th>
                  <th>Kind</th>
                  <th className="num">Rows</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.tables).flatMap(([table, entries]) =>
                  entries.map((e) => (
                    <tr key={`${table}-${e.source_name}`}>
                      <td className="muted">{table}</td>
                      <td>{e.source_name}</td>
                      <td>
                        <span className={e.is_synthetic ? "badge badge-synthetic" : "badge badge-real"}>
                          {e.is_synthetic ? "synthetic" : "real"}
                        </span>
                      </td>
                      <td className="num">{e.rows.toLocaleString()}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="panel">
        <h3>Feed freshness</h3>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Adapter</th>
                <th>Status</th>
                <th className="num">Rows</th>
                <th>Last run</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {freshness.map((f) => (
                <tr key={f.adapter}>
                  <td>{f.adapter}</td>
                  <td>
                    <span className={`badge ${f.status === "ok" ? "badge-real" : f.status === "failed" ? "badge-reject" : "badge-synthetic"}`}>
                      {f.status}
                      {f.used_cache ? " (cached)" : ""}
                    </span>
                  </td>
                  <td className="num">{f.rows_written.toLocaleString()}</td>
                  <td className="muted">
                    {f.finished_at ? new Date(f.finished_at).toLocaleString() : "-"}
                  </td>
                  <td className="muted">{f.message ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h3>Port constraint master</h3>
        <p className="muted" style={{ fontSize: 12.5, marginTop: -6, marginBottom: 12 }}>
          Declared drafts are not constants. SMP Kolkata reissues Hooghly draft circulars monthly
          and permissible draft there is a function of LOA, so these are current published figures
          that the tide model and circular adapter override at runtime.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Port</th>
                <th>Role</th>
                <th className="num">Berths</th>
                <th className="num">Deepest (m)</th>
                <th className="num">Density (t/m3)</th>
                <th>Tidal</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {ports.map((p) => (
                <tr key={p.code}>
                  <td>{p.name} <span className="muted">({p.code})</span></td>
                  <td className="muted">{p.role}</td>
                  <td className="num">{p.berths.length}</td>
                  <td className="num">
                    {p.berths.length ? Math.max(...p.berths.map((b) => b.max_draft_m)).toFixed(1) : "-"}
                  </td>
                  <td className="num" style={{ color: (p.water_density ?? 1.025) < 1.02 ? "var(--warn)" : undefined }}>
                    {p.water_density?.toFixed(3) ?? "-"}
                  </td>
                  <td className="muted">{p.tide_constrained ? "yes" : "no"}</td>
                  <td className="muted" style={{ maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {p.source?.startsWith("http") ? (
                      <a href={p.source} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
                        {new URL(p.source).hostname}
                      </a>
                    ) : (p.source ?? "-")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
