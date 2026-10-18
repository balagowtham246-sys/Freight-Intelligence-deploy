import { useEffect, useState } from "react";
import CharteringAdvisor from "./pages/CharteringAdvisor";
import EntryTiming from "./pages/EntryTiming";
import RiskWatch from "./pages/RiskWatch";
import PortWatch from "./pages/PortWatch";
import MarketCockpit from "./pages/MarketCockpit";
import DataProvenance from "./pages/DataProvenance";
import { api, type ProvenanceSummary } from "./api";

type PageKey = "advisor" | "timing" | "risk" | "portwatch" | "market" | "provenance";

const PAGES: { key: PageKey; label: string }[] = [
  { key: "advisor", label: "Chartering Advisor" },
  { key: "timing", label: "Entry Timing" },
  { key: "risk", label: "Risk Watch" },
  { key: "portwatch", label: "Port Watch" },
  { key: "market", label: "Market Cockpit" },
  { key: "provenance", label: "Data Provenance" },
];

export default function App() {
  const [page, setPage] = useState<PageKey>("advisor");
  const [summary, setSummary] = useState<ProvenanceSummary | null>(null);

  useEffect(() => {
    api.provenanceSummary().then(setSummary).catch(() => setSummary(null));
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>Freight Platform</h1>
          <p>Bulk chartering into East Coast India</p>
        </div>
        <nav className="nav">
          {PAGES.map((p) => (
            <button
              key={p.key}
              className={page === p.key ? "active" : ""}
              onClick={() => setPage(p.key)}
            >
              {p.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          {summary ? (
            <>
              <div>
                <span className={summary.synthetic_rows === 0 ? "badge badge-real" : "badge badge-synthetic"}>
                  {summary.pct_real}% real
                </span>
              </div>
              <div style={{ marginTop: 6 }}>
                {summary.total_rows.toLocaleString()} rows held
                <br />
                {summary.synthetic_rows === 0
                  ? "no synthetic data in use"
                  : `${summary.synthetic_rows.toLocaleString()} synthetic`}
              </div>
            </>
          ) : (
            <div>connecting to API...</div>
          )}
        </div>
      </aside>

      <main className="main">
        {page === "advisor" && <CharteringAdvisor />}
        {page === "timing" && <EntryTiming />}
        {page === "risk" && <RiskWatch />}
        {page === "portwatch" && <PortWatch />}
        {page === "market" && <MarketCockpit />}
        {page === "provenance" && <DataProvenance />}
      </main>
    </div>
  );
}
