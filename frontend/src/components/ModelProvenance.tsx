import type { ModelProvenance } from "../api";

/**
 * Attribution for a *modelled* figure.
 *
 * Sibling of ProvenanceLine, and deliberately a different badge. A forecast is
 * neither a fetched observation nor a fabricated feed: it is a claim derived
 * from inputs. Labelling it "real" would imply a publisher stands behind it;
 * labelling it "synthetic" would conflate a model with invented data. So it
 * gets its own word, and the inputs it was fitted on are named.
 *
 * When any input was a disclosed synthetic series the forecast inherits that
 * taint and says so, because a model trained on generated history is only as
 * trustworthy as that history.
 */
export function ModelProvenanceLine({
  m,
  signal,
}: {
  m: ModelProvenance;
  signal?: string;
}) {
  const signalClass =
    signal === "strong" ? "badge-strong" : signal === "weak" ? "badge-weak" : "badge-noedge";

  return (
    <div className="provenance">
      <span className="badge badge-modelled">modelled</span>
      {signal && (
        <span className={`badge ${signalClass}`}>
          {signal === "no_edge" ? "no edge" : `${signal} signal`}
        </span>
      )}
      {m.trained_on_synthetic && (
        <span className="badge badge-synthetic">synthetic inputs</span>
      )}
      <span>
        {m.model_name} v{m.model_version} &middot; trained through {m.trained_through}
        {m.n_train ? ` on ${m.n_train.toLocaleString()} rows` : ""}
      </span>
    </div>
  );
}

/** Skill badge used inline next to a horizon. */
export function EdgeBadge({ hasEdge, skill }: { hasEdge: boolean; skill: number | null }) {
  if (!hasEdge) return <span className="badge badge-noedge">no edge</span>;
  return (
    <span className="badge badge-strong">
      beats baseline{skill !== null ? ` by ${(skill * 100).toFixed(0)}%` : ""}
    </span>
  );
}
