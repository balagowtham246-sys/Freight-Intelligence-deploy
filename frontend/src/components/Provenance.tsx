import type { Provenance } from "../api";

/**
 * Attribution shown under every figure the platform displays.
 *
 * The synthetic badge is deliberately prominent: a reader should never have
 * to wonder whether a number came from a real source.
 */
export function ProvenanceLine({ p }: { p: Provenance }) {
  return (
    <div className="provenance">
      <span className={p.is_synthetic ? "badge badge-synthetic" : "badge badge-real"}>
        {p.is_synthetic ? "synthetic" : "real data"}
      </span>
      <span>
        {p.source_name}
        {p.retrieved_at ? ` - retrieved ${new Date(p.retrieved_at).toLocaleString()}` : ""}
      </span>
      {p.source_url && p.source_url.startsWith("http") && (
        <a href={p.source_url} target="_blank" rel="noreferrer">
          source
        </a>
      )}
    </div>
  );
}
