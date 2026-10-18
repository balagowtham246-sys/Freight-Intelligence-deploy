import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { api, type CoalStock, type Series, type SeriesInfo } from "../api";
import { ProvenanceLine } from "../components/Provenance";

export default function MarketCockpit() {
  const [list, setList] = useState<SeriesInfo[]>([]);
  const [code, setCode] = useState("COAL_AU_NEWC");
  const [series, setSeries] = useState<Series | null>(null);
  const [stock, setStock] = useState<CoalStock | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.seriesList().then(setList).catch((e) => setError(String(e)));
    api.coalStock().then(setStock).catch(() => setStock(null));
  }, []);

  useEffect(() => {
    api.series(code, "2010-01-01").then(setSeries).catch((e) => setError(String(e)));
  }, [code]);

  const priceOption = series && {
    backgroundColor: "transparent",
    grid: { left: 56, right: 20, top: 24, bottom: 40 },
    textStyle: { color: "#8b949e", fontFamily: "Inter, sans-serif" },
    tooltip: { trigger: "axis" as const },
    xAxis: {
      type: "category",
      data: series.points.map((p) => p.date),
      axisLine: { lineStyle: { color: "#2a3038" } },
    },
    yAxis: {
      type: "value",
      scale: true,
      splitLine: { lineStyle: { color: "#21262d" } },
      name: series.unit,
      nameTextStyle: { color: "#8b949e" },
    },
    series: [
      {
        type: "line",
        smooth: true,
        showSymbol: false,
        data: series.points.map((p) => p.value),
        itemStyle: { color: "#4493f8" },
        areaStyle: {
          color: {
            type: "linear", x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(68,147,248,0.28)" },
              { offset: 1, color: "rgba(68,147,248,0.01)" },
            ],
          },
        },
      },
    ],
  };

  const critical = stock?.plants.filter((p) => (p.days_of_burn ?? 99) < 7) ?? [];

  return (
    <div>
      <div className="page-head">
        <h2>Market Cockpit</h2>
        <p>
          Commodity price history and the import-demand signal &mdash; observed data only. The
          forecaster now runs, and it is measured on walk-forward folds against a random walk and a
          seasonal naive baseline; at any horizon where it does not beat them the platform reports
          no edge rather than a number. The fan chart and the per-horizon skill table are on Entry
          Timing.
        </p>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="panel">
        <h3>Price series</h3>
        <div className="form-row" style={{ marginBottom: 14 }}>
          <div className="field">
            <label>Series</label>
            <select value={code} onChange={(e) => setCode(e.target.value)}>
              {list.map((s) => (
                <option key={s.series_code} value={s.series_code}>
                  {s.series_code} ({s.observations} obs, from {s.first.slice(0, 4)})
                </option>
              ))}
            </select>
          </div>
        </div>
        {priceOption && <ReactECharts option={priceOption} style={{ height: 320 }} />}
        {series && <ProvenanceLine p={series.provenance} />}
      </div>

      {stock && (
        <div className="grid grid-3" style={{ marginBottom: 16 }}>
          <div className="stat">
            <div className="label">Import-designed plants</div>
            <div className="value">{stock.plant_count}</div>
            <div className="sub">CEA daily report, {stock.obs_date}</div>
          </div>
          <div className="stat">
            <div className="label">Median days of burn</div>
            <div className="value">
              {stock.median_days_of_burn ? stock.median_days_of_burn.toFixed(1) : "n/a"}
            </div>
            <div className="sub">stock divided by daily requirement</div>
          </div>
          <div className="stat">
            <div className="label">Below 5 days</div>
            <div className="value" style={{ color: stock.critical_count > 0 ? "var(--bad)" : undefined }}>
              {stock.critical_count}
            </div>
            <div className="sub">forced buyers, regardless of freight</div>
          </div>
        </div>
      )}

      {stock && (
        <div className="panel">
          <h3>Tightest import-coal plant stocks</h3>
          <p className="muted" style={{ fontSize: 12.5, marginTop: -6, marginBottom: 12 }}>
            When these fall, buyers enter the spot market whatever the freight level. This is the
            best free proxy for import urgency that exists.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Plant</th>
                  <th>State</th>
                  <th className="num">Capacity (MW)</th>
                  <th className="num">Stock (kt)</th>
                  <th className="num">Days of burn</th>
                </tr>
              </thead>
              <tbody>
                {(critical.length ? critical : stock.plants.slice(0, 10)).map((p) => (
                  <tr key={p.plant_name}>
                    <td>{p.plant_name}</td>
                    <td className="muted">{p.state ?? "-"}</td>
                    <td className="num muted">{p.capacity_mw?.toLocaleString() ?? "-"}</td>
                    <td className="num">{p.stock_kt?.toLocaleString() ?? "-"}</td>
                    <td className="num" style={{
                      color: (p.days_of_burn ?? 99) < 5 ? "var(--bad)"
                        : (p.days_of_burn ?? 99) < 10 ? "var(--warn)" : undefined,
                      fontWeight: 600,
                    }}>
                      {p.days_of_burn?.toFixed(1) ?? "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ProvenanceLine p={stock.provenance} />
        </div>
      )}
    </div>
  );
}
