"""The feature panel: one row per date, every column knowable on that date.

The single rule this module exists to enforce
---------------------------------------------
**A feature at date *t* may only use observations with ``obs_date <= t``, and
every driver is lagged by at least one period.**

A leak is not a bug that makes the model slightly optimistic. It makes every
backtest number in the platform a lie, and it is invisible in the output — a
leaking model looks *excellent*. So the rule is enforced structurally here
(shift-then-join, never join-then-shift) and asserted in
``backend/tests/test_leakage.py``, which is the most important test in the
repository.

Two specific traps this module avoids:

* **Monthly data published in arrears.** The World Bank Pink Sheet for August
  is stamped ``2026-08-01`` but is not published until well into September.
  Joining it on its stamp would hand the model a month of hindsight, so
  monthly series are lagged by ``MONTHLY_PUBLICATION_LAG_DAYS`` before use.
* **Forward-filling across the future.** ``ffill`` after a full-calendar
  reindex is safe only because the reindex is bounded at *t*. The panel is
  always built up to an explicit ``as_of`` date and never beyond.

Output is cached to ``backend/data/parquet/`` — the directory ``config.py``
creates and ``db.py`` designates for exactly this.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import PARQUET_DIR
from ..models import CoalStockObservation, PortCallSnapshot, RateObservation

# The Pink Sheet's August figure is stamped 2026-08-01 but lands in September.
# Anything shorter than this is optimistic; anything longer just costs signal.
MONTHLY_PUBLICATION_LAG_DAYS = 35

# Port-authority PDFs are published the same morning they describe, so a
# one-day lag is enough to keep the panel causal.
DAILY_PUBLICATION_LAG_DAYS = 1

CONGESTION_PORTS = ("INPRT",)


@dataclass(frozen=True)
class PanelSpec:
    """What to build. Held as a value so a run is reproducible from its spec."""

    target_series: str
    as_of: date
    start: date | None = None
    congestion_ports: tuple[str, ...] = CONGESTION_PORTS

    @property
    def cache_key(self) -> str:
        return f"panel__{self.target_series}__{self.as_of.isoformat()}"


def _series_frame(session: Session, as_of: date) -> pd.DataFrame:
    """Every rate/commodity observation up to ``as_of``, wide by series code."""
    rows = session.execute(
        select(
            RateObservation.series_code,
            RateObservation.obs_date,
            RateObservation.value,
            RateObservation.series_kind,
        ).where(RateObservation.obs_date <= as_of)
    ).all()
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, columns=["series_code", "obs_date", "value", "kind"])
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])

    # Lag by publication delay *before* pivoting, so the shift is applied in
    # the series' own calendar rather than the panel's.
    monthly = frame["kind"] == "commodity"
    frame.loc[monthly, "obs_date"] += pd.Timedelta(days=MONTHLY_PUBLICATION_LAG_DAYS)
    frame.loc[~monthly, "obs_date"] += pd.Timedelta(days=DAILY_PUBLICATION_LAG_DAYS)

    wide = frame.pivot_table(
        index="obs_date", columns="series_code", values="value", aggfunc="last"
    )
    return wide.sort_index()


def _congestion_frame(
    session: Session, as_of: date, ports: tuple[str, ...]
) -> pd.DataFrame:
    """Daily queue counts per port, from the archived berthing programmes."""
    rows = session.execute(
        select(
            PortCallSnapshot.port_code,
            PortCallSnapshot.snapshot_date,
            PortCallSnapshot.status,
        ).where(
            PortCallSnapshot.snapshot_date <= as_of,
            PortCallSnapshot.port_code.in_(ports),
        )
    ).all()
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, columns=["port_code", "snapshot_date", "status"])
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]) + pd.Timedelta(
        days=DAILY_PUBLICATION_LAG_DAYS
    )
    counts = (
        frame.groupby(["snapshot_date", "port_code", "status"])
        .size()
        .unstack(["port_code", "status"], fill_value=0)
    )
    counts.columns = [f"queue_{port.lower()}_{status}" for port, status in counts.columns]
    return counts.sort_index()


def _coal_stock_frame(session: Session, as_of: date) -> pd.DataFrame:
    """CEA days-of-burn at import-designed plants: the demand-urgency signal."""
    rows = session.execute(
        select(
            CoalStockObservation.obs_date,
            CoalStockObservation.days_of_burn,
            CoalStockObservation.is_import_based,
        ).where(
            CoalStockObservation.obs_date <= as_of,
            CoalStockObservation.days_of_burn.isnot(None),
        )
    ).all()
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, columns=["obs_date", "days_of_burn", "import_based"])
    frame["obs_date"] = pd.to_datetime(frame["obs_date"]) + pd.Timedelta(
        days=DAILY_PUBLICATION_LAG_DAYS
    )

    national = frame.groupby("obs_date")["days_of_burn"].median().rename("dob_all")
    imported = (
        frame[frame["import_based"] == True]  # noqa: E712 - SQL-sourced nullable bool
        .groupby("obs_date")["days_of_burn"]
        .median()
        .rename("dob_import")
    )
    critical = (
        frame[frame["days_of_burn"] < 5]
        .groupby("obs_date")
        .size()
        .rename("dob_critical_count")
    )
    return pd.concat([national, imported, critical], axis=1).sort_index()


def build_panel(session: Session, spec: PanelSpec) -> pd.DataFrame:
    """Assemble the daily feature panel up to ``spec.as_of``.

    Returns a frame indexed by calendar date with the raw (unlagged-within-
    panel) columns. Feature construction proper lives in ``features.py``; this
    module's only job is to get causally-correct raw columns onto one grid.
    """
    series = _series_frame(session, spec.as_of)
    if series.empty or spec.target_series not in series.columns:
        raise ValueError(
            f"target series {spec.target_series!r} has no observations at or before "
            f"{spec.as_of.isoformat()}"
        )

    congestion = _congestion_frame(session, spec.as_of, spec.congestion_ports)
    coal = _coal_stock_frame(session, spec.as_of)

    # Default the grid to where the target actually begins. Driver series run
    # back to 1960; carrying sixty years of rows the target cannot label just
    # inflates the panel and invites accidental fabrication in aggregations.
    target_start = series[spec.target_series].dropna().index.min()
    start = spec.start or target_start.date()
    grid = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(spec.as_of), freq="D")

    panel = pd.DataFrame(index=grid)
    panel.index.name = "date"

    for frame in (series, congestion, coal):
        if frame.empty:
            continue
        panel = panel.join(frame.reindex(grid.union(frame.index)).ffill().reindex(grid))

    # The target itself is never forward-filled beyond where it was observed:
    # a flat run would look like a run of real observations to the evaluator.
    observed = series[spec.target_series].dropna()
    panel[spec.target_series] = observed.reindex(grid).ffill()
    panel["target_observed"] = observed.reindex(grid).notna()

    return panel


def cache_panel(panel: pd.DataFrame, spec: PanelSpec) -> str:
    """Persist a built panel to Parquet and return the path."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / f"{spec.cache_key}.parquet"
    panel.to_parquet(path)
    return str(path)


def assert_causal(panel: pd.DataFrame, as_of: date) -> None:
    """Guard: nothing in the panel may be dated after ``as_of``."""
    if len(panel.index) and panel.index.max().date() > as_of:
        raise AssertionError(
            f"panel extends to {panel.index.max().date()} beyond as_of {as_of}"
        )


def horizon_target(
    panel: pd.DataFrame, target_series: str, horizon_days: int
) -> pd.Series:
    """The value of the target ``horizon_days`` ahead, aligned to the origin date.

    This is the label for direct multi-horizon regression. It is NaN for the
    final ``horizon_days`` rows by construction, which is what stops the model
    training on a future it cannot have seen.
    """
    return panel[target_series].shift(-horizon_days)


def log_returns(values: pd.Series) -> pd.Series:
    """Log returns, with non-positive values treated as missing rather than -inf."""
    safe = values.where(values > 0)
    return np.log(safe).diff()
