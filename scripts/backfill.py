"""Backfill and refresh every ingestion adapter.

Usage:
    python scripts/backfill.py --dry-run
    python scripts/backfill.py --adapter paradip --start 2026-05-01 --end 2026-09-03
    python scripts/backfill.py --all
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for path in (str(BACKEND), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend.app.db import init_db, session_scope  # noqa: E402
    from backend.app.ingest import ADAPTERS, BACKFILLABLE  # noqa: E402
except ImportError:
    from app.db import init_db, session_scope  # noqa: E402
    from app.ingest import ADAPTERS, BACKFILLABLE  # noqa: E402


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="run a single adapter by key")
    parser.add_argument("--all", action="store_true", help="run every adapter once")
    parser.add_argument("--dry-run", action="store_true", help="check reachability only")
    parser.add_argument("--start", type=parse_date, help="backfill start date")
    parser.add_argument("--end", type=parse_date, help="backfill end date")
    args = parser.parse_args()

    init_db()

    if args.dry_run:
        return dry_run(args.adapter)

    keys = [args.adapter] if args.adapter else list(ADAPTERS)
    exit_code = 0

    for key in keys:
        if key not in ADAPTERS:
            print(f"unknown adapter {key!r}; known: {', '.join(ADAPTERS)}")
            return 2

        adapter = ADAPTERS[key]()

        if args.start and key in BACKFILLABLE:
            end = args.end or date.today()
            print(f"[{key}] backfilling {args.start} -> {end}")
            with session_scope() as session:
                stats = adapter.backfill(session, args.start, end)
            print(f"[{key}] {stats}")
            continue

        with session_scope() as session:
            result = adapter.execute(session)
        flag = "OK " if result.status in {"ok", "stale"} else "ERR"
        print(f"[{flag}] {key}: {result.status} rows={result.rows_written} {result.message or ''}")
        if result.status == "failed":
            exit_code = 1

    return exit_code


def dry_run(only: str | None) -> int:
    """Confirm each adapter can reach its live source, without writing rows."""
    keys = [only] if only else list(ADAPTERS)
    failures = 0

    for key in keys:
        adapter = ADAPTERS[key]()
        probe = getattr(adapter, "probe", None)
        try:
            if probe is None:
                print(f"[--- ] {key}: no probe implemented")
                continue
            ok, detail = probe()
            print(f"[{'OK ' if ok else 'ERR'}] {key}: {detail}")
            failures += 0 if ok else 1
        except Exception as exc:
            print(f"[ERR] {key}: {type(exc).__name__}: {exc}")
            failures += 1
        finally:
            adapter.close()

    print(f"\n{len(keys) - failures}/{len(keys)} sources reachable")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
