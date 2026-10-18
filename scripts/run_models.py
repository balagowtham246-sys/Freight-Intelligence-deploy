"""Fit the models, write forecasts, and evaluate the risk rules.

Run after ingestion. The intended daily sequence is:

    python scripts/backfill.py --all
    python scripts/run_models.py

Usage:
    python scripts/run_models.py                      # everything
    python scripts/run_models.py --no-backtest        # skip the walk-forward
    python scripts/run_models.py --target SYN_CDFI_TABONEO_GUANGZHOU_50K
    python scripts/run_models.py --alerts-only
    python scripts/run_models.py --folds 12

Kept as a script invoked by cron or Task Scheduler rather than an in-process
scheduler, matching how ``backfill.py`` already works.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for path in (str(BACKEND), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend.app.db import init_db, session_scope  # noqa: E402
    from backend.app.ml import pipeline  # noqa: E402
    from backend.app.ml.waiting import rebuild_waiting_times  # noqa: E402
except ImportError:
    from app.db import init_db, session_scope  # noqa: E402
    from app.ml import pipeline  # noqa: E402
    from app.ml.waiting import rebuild_waiting_times  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="series code to forecast")
    parser.add_argument(
        "--no-backtest", action="store_true", help="skip walk-forward evaluation"
    )
    parser.add_argument(
        "--alerts-only", action="store_true", help="evaluate risk rules only"
    )
    parser.add_argument(
        "--folds", type=int, default=8, help="walk-forward folds per horizon"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # LightGBM and statsmodels are chatty on small samples; the numbers that
    # matter are reported explicitly below.
    warnings.filterwarnings("ignore")

    init_db()

    if args.alerts_only:
        with session_scope() as session:
            summary = pipeline.run_alerts(session)
        print(json.dumps({"alerts": summary}, indent=2))
        return 0

    with session_scope() as session:
        waiting = rebuild_waiting_times(session)
    if not args.quiet:
        print(
            f"[waiting] {waiting['calls']} port calls "
            f"({waiting['events']} berthed, {waiting['censored']} censored)"
        )

    with session_scope() as session:
        models = pipeline.run_models(
            session,
            target_key=args.target,
            backtest=not args.no_backtest,
            n_folds=args.folds,
        )

    if not args.quiet:
        print(f"[models] target {models.get('target')} status {models.get('status')}")
        if models.get("trained_on_synthetic"):
            print(
                "[models] NOTE: trained on a disclosed synthetic series. "
                "See SYNTHETIC_DATA.md. Metrics are labelled accordingly."
            )
        for row in models.get("backtest", []):
            edge = "EDGE" if row["has_edge"] else "no edge"
            skill = row["point_skill"]
            print(
                f"  h={row['horizon_days']:>4}d  folds={row['n_folds']:>2}  "
                f"point_skill={skill if skill is None else f'{skill:+.3f}'}  "
                f"cov80={row['coverage_80']:.2f}  {edge}"
            )

    with session_scope() as session:
        alerts = pipeline.run_alerts(session)
    if not args.quiet:
        print(
            f"[alerts] {alerts['evaluated']} evaluated: "
            f"{alerts['raised']} raised, {alerts['refreshed']} refreshed, "
            f"{alerts['cleared']} cleared"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
