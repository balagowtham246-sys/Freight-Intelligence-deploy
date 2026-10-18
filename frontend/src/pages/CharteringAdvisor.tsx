import { useEffect, useState } from "react";
import { api, type CargoGrade, type FeasibilityResult, type Port, type RoutingOption } from "../api";
import { ProvenanceLine } from "../components/Provenance";

const NUM = (v: number, digits = 0) =>
  v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

function LimitBadge({ factor }: { factor: string }) {
  const label = factor.replace(/_/g, " ");
  return <span className="badge badge-reject" style={{ background: "rgba(139,148,158,0.15)", color: "var(--muted)", borderColor: "var(--border)" }}>{label} limited</span>;
}

function OptionCard({ option, best }: { option: RoutingOption; best: boolean }) {
  const workings = option.detail?.workings as Record<string, number> | undefined;
  return (
    <div className={`option${best ? " best" : ""}`}>
      <div className="option-head">
        <span className="option-title">{option.vessel_class}</span>
        <span className={`badge badge-${option.routing}`}>{option.routing}</span>
        <LimitBadge factor={option.limiting_factor} />
        <span className="option-tonnes">
          {NUM(option.loadable_tonnes)} t
          <span className="muted" style={{ fontSize: 12, fontWeight: 400 }}>
            {" "}&times; {option.vessels_required}
          </span>
        </span>
      </div>
      <div className="option-explain">{option.explanation}</div>
      {workings && (
        <div className="workings">
          <div><span>governing draft</span><br />{workings.governing_draft_m?.toFixed(2)} m</div>
          <div><span>berth max draft</span><br />{workings.berth_max_draft_m?.toFixed(2)} m</div>
          <div><span>tide credit</span><br />{workings.tide_credit_m?.toFixed(2)} m</div>
          <div><span>water density</span><br />{workings.water_density?.toFixed(3)} t/m3</div>
          <div><span>displacement</span><br />{NUM(workings.displacement_t ?? 0)} t</div>
          <div><span>bunkers ROB</span><br />{NUM(workings.bunkers_rob_t ?? 0)} t</div>
        </div>
      )}
    </div>
  );
}

export default function CharteringAdvisor() {
  const [ports, setPorts] = useState<Port[]>([]);
  const [cargoes, setCargoes] = useState<CargoGrade[]>([]);
  const [dischargePort, setDischargePort] = useState("INHAL");
  const [loadPort, setLoadPort] = useState("IDTAB");
  const [tonnes, setTonnes] = useState(150000);
  const [cargo, setCargo] = useState("thermal_coal_low_cv");
  const [tide, setTide] = useState(0);
  const [result, setResult] = useState<FeasibilityResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.ports(), api.cargoes()])
      .then(([p, c]) => {
        setPorts(p);
        setCargoes(c);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.screen({
          discharge_port: dischargePort,
          load_port: loadPort || null,
          required_tonnes: tonnes,
          cargo,
          tide_height_m: tide,
        })
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (ports.length) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ports.length]);

  const discharge = ports.filter((p) => p.role === "discharge");
  const loading = ports.filter((p) => p.role === "load");

  return (
    <div>
      <div className="page-head">
        <h2>Chartering Advisor</h2>
        <p>
          Every way of moving a cargo, ranked, with the constraint arithmetic shown. A rejection
          reason is as useful as a recommendation, so both are reported.
        </p>
      </div>

      <div className="panel">
        <h3>Cargo requirement</h3>
        <div className="form-row">
          <div className="field">
            <label>Load port</label>
            <select value={loadPort} onChange={(e) => setLoadPort(e.target.value)}>
              <option value="">Not specified</option>
              {loading.map((p) => (
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
                <option key={c.name} value={c.name}>
                  {c.name.replace(/_/g, " ")} ({c.stowage_factor_m3_per_t} m3/t)
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Tide at ETA (m)</label>
            <input type="number" value={tide} step={0.5} min={0}
              onChange={(e) => setTide(Number(e.target.value))} />
          </div>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Screening..." : "Screen options"}
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="panel">
            <h3>Recommendation</h3>
            <div style={{ fontSize: 15, marginBottom: 4 }}>{result.summary}</div>
            <ProvenanceLine p={result.provenance} />
          </div>

          <div className="panel">
            <h3>Feasible routings ({result.feasible.length})</h3>
            {result.feasible.length === 0 && (
              <div className="muted">No feasible routing for this requirement.</div>
            )}
            {result.feasible.map((o, i) => (
              <OptionCard key={`${o.routing}-${o.vessel_class}-${o.transhipment_port ?? ""}-${i}`}
                option={o} best={i === 0} />
            ))}
          </div>

          {result.rejected.length > 0 && (
            <div className="panel">
              <h3>Rejected, and why ({result.rejected.length})</h3>
              {result.rejected.map((o, i) => (
                <div className="option rejected" key={`${o.vessel_class}-${i}`}>
                  <div className="option-head">
                    <span className="option-title">{o.vessel_class}</span>
                    <span className="badge badge-reject">rejected</span>
                  </div>
                  <div className="option-explain">{o.explanation}</div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
