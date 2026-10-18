"""Disclosed synthetic freight history, for model development only.

Read SYNTHETIC_DATA.md before changing anything here. This is the only adapter
in the platform that fabricates data, and it exists for one narrow reason.

The problem it solves
---------------------
``sse_cdfi`` gives us a real, free, route-level freight assessment in $/tonne
on the exact load ports we model. It is the right target variable. But its
history endpoint is subscriber-gated, so on the day it is switched on the
platform holds two observations. Two observations will not train, calibrate or
backtest anything.

So the modelling machinery needs a history to be built and tested against, and
the only honest way to have one today is to generate it and say so loudly.

How the series are kept separate from real data
-----------------------------------------------
Synthetic history is written under its own ``SYN_`` series codes, never mixed
into the real CDFI series. That means:

  * ``/api/series`` reports ``is_synthetic`` per series with no ambiguity;
  * the provenance badge on any chart is either real or synthetic, never both;
  * the real CDFI series accumulates alongside from day one and takes over as
    the training set the moment it is deep enough, with no migration.

What is real inside the generator
---------------------------------
Two things, and they do a lot of work:

  1. **The anchor.** Each path is rescaled so its final value equals the real
     CDFI assessment observed today. The level is not invented.
  2. **The driver.** The systematic component is a regression on the real
     World Bank monthly Australian thermal coal price already held in this
     database (``COAL_AU_NEWC``, back to 1970), interpolated to daily. The
     shape of the cycle comes from a real price series.

What is assumed
---------------
The stochastic component and the parameters below. They are chosen to
reproduce documented properties of dry bulk freight rather than to fit
anything:

  * ``PHI = 0.97`` - near-unit-root persistence. Dry bulk freight is close to
    a random walk in log space; this is the single most important property to
    get right, because it is what makes the random-walk baseline hard to beat.
  * ``BETA = 0.35`` - elasticity of log freight to log coal price.
  * ``TARGET_ANNUAL_VOL = 0.55`` - annualised volatility, within the published
    range for dry bulk freight.
  * GARCH(1,1)-shaped innovation variance, so volatility clusters. Without
    this the volatility model and the ``MARKET_VOL`` alert would have nothing
    to detect and would test vacuously.
  * A Fourier annual seasonal term, amplitude 8%, peaking into the northern
    hemisphere winter restocking season.

The hard constraint
-------------------
This series may be used as a *training and development* target. It may never
be used as a *benchmark*. Beating a curve we generated ourselves proves
nothing, which is the point SYNTHETIC_DATA.md makes about synthetic FFA
curves. ``app.ml.evaluate`` carries that restriction in code, and every
metric computed on a ``SYN_`` series is labelled as such through the API.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import RateObservation
from .base import IngestResult, SourceAdapter, upsert
from .sse_cdfi import ROUTE_META

# Generator parameters. See the module docstring for why each is set this way.
YEARS_OF_HISTORY = 8
PHI = 0.97                 # AR(1) persistence of the idiosyncratic component
BETA = 0.35                # elasticity of log freight to log coal price
TARGET_ANNUAL_VOL = 0.55   # annualised volatility of log returns
SEASONAL_AMPLITUDE = 0.08  # peak-to-mean seasonal swing in log space
SEASONAL_PEAK_DOY = 330    # late November: northern hemisphere restocking
GARCH_OMEGA = 0.05
GARCH_ALPHA = 0.12
GARCH_BETA = 0.83

DRIVER_SERIES = "COAL_AU_NEWC"
SYNTHETIC_PREFIX = "SYN_"

TRADING_DAYS = 252


def synthetic_code(real_code: str) -> str:
    """The synthetic counterpart of a real series code."""
    return f"{SYNTHETIC_PREFIX}{real_code}"


def _seed_for(series_code: str) -> int:
    """Deterministic per-series seed, so a disclosed dataset is reproducible."""
    digest = hashlib.sha256(series_code.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _daily_driver(
    monthly: list[tuple[date, float]], start: date, end: date
) -> np.ndarray:
    """Interpolate a monthly price series onto a daily calendar grid.

    Returns standardised log prices, so the regression coefficient is in
    units the module docstring can talk about.
    """
    if len(monthly) < 2:
        raise ValueError(
            f"driver series {DRIVER_SERIES} has too few observations to interpolate"
        )

    ordinals = np.array([point.toordinal() for point, _ in monthly], dtype=float)
    values = np.array([value for _, value in monthly], dtype=float)
    grid = np.arange(start.toordinal(), end.toordinal() + 1, dtype=float)

    interpolated = np.interp(grid, ordinals, values)
    log_price = np.log(np.maximum(interpolated, 1e-6))
    centred = log_price - log_price.mean()
    spread = centred.std()
    return centred / spread if spread > 0 else centred


def _garch_innovations(rng: np.random.Generator, n: int) -> np.ndarray:
    """GARCH(1,1)-shaped innovations, normalised to unit unconditional variance."""
    shocks = rng.standard_normal(n)
    variance = np.empty(n)
    long_run = GARCH_OMEGA / max(1.0 - GARCH_ALPHA - GARCH_BETA, 1e-6)
    variance[0] = long_run
    innovations = np.empty(n)
    innovations[0] = np.sqrt(variance[0]) * shocks[0]

    for t in range(1, n):
        variance[t] = (
            GARCH_OMEGA
            + GARCH_ALPHA * innovations[t - 1] ** 2
            + GARCH_BETA * variance[t - 1]
        )
        innovations[t] = np.sqrt(variance[t]) * shocks[t]

    realised = innovations.std()
    return innovations / realised if realised > 0 else innovations


def generate_path(
    series_code: str,
    anchor_value: float,
    driver: np.ndarray,
    calendar: list[date],
) -> list[float]:
    """Generate one synthetic freight path ending exactly at ``anchor_value``.

    The path is a deterministic function of the series code, the driver and
    the anchor, so re-running the adapter reproduces it exactly.
    """
    n = len(calendar)
    rng = np.random.default_rng(_seed_for(series_code))

    # Scale the innovations, not the level. With PHI close to one the first
    # difference of the AR(1) is approximately the innovation itself, so this
    # is what sets the volatility of log *returns* - which is the quantity
    # TARGET_ANNUAL_VOL is quoted in, and the quantity that decides how hard
    # the random-walk baseline is to beat.
    daily_vol = TARGET_ANNUAL_VOL / np.sqrt(TRADING_DAYS)
    innovations = _garch_innovations(rng, n) * daily_vol

    # AR(1) idiosyncratic component, started from its stationary distribution.
    stationary_sd = daily_vol / np.sqrt(1.0 - PHI**2)
    idiosyncratic = np.empty(n)
    idiosyncratic[0] = rng.standard_normal() * stationary_sd
    for t in range(1, n):
        idiosyncratic[t] = PHI * idiosyncratic[t - 1] + innovations[t]

    day_of_year = np.array([point.timetuple().tm_yday for point in calendar], dtype=float)
    seasonal = SEASONAL_AMPLITUDE * np.cos(
        2.0 * np.pi * (day_of_year - SEASONAL_PEAK_DOY) / 365.25
    )

    log_path = BETA * driver + seasonal + idiosyncratic

    # Anchor: shift the whole path so it ends on the real observed level.
    log_path = log_path - log_path[-1] + np.log(anchor_value)
    return [round(float(value), 4) for value in np.exp(log_path)]


class SyntheticFreightHistoryAdapter(SourceAdapter):
    """Generates disclosed synthetic history for the real CDFI coal routes."""

    name = "synthetic_freight_history"
    description = (
        "Disclosed synthetic freight history for CDFI coal routes, anchored to the "
        "real current assessment and driven by the real World Bank coal price. "
        "Training and development only; never a benchmark."
    )
    homepage = "SYNTHETIC_DATA.md"
    produces_synthetic = True

    def probe(self) -> tuple[bool, str]:
        """No network. Reachability here means the real inputs are present."""
        return True, "generated locally from real anchors held in the warehouse"

    def run(self, session: Session) -> IngestResult:
        # provenance() runs assert_disclosed() because produces_synthetic is
        # True, so an undeclared source fails here before anything is written.
        provenance = self.provenance(url="SYNTHETIC_DATA.md")

        driver_rows = session.execute(
            select(RateObservation.obs_date, RateObservation.value)
            .where(RateObservation.series_code == DRIVER_SERIES)
            .order_by(RateObservation.obs_date)
        ).all()
        if not driver_rows:
            return IngestResult(
                self.name,
                "skipped",
                message=(
                    f"driver series {DRIVER_SERIES} not loaded; "
                    "run the worldbank adapter first"
                ),
            )
        monthly = [(row[0], float(row[1])) for row in driver_rows]

        anchors: dict[str, tuple[date, float]] = {}
        for real_code in ROUTE_META:
            row = session.execute(
                select(RateObservation.obs_date, RateObservation.value)
                .where(
                    RateObservation.series_code == real_code,
                    RateObservation.is_synthetic.is_(False),
                )
                .order_by(RateObservation.obs_date.desc())
                .limit(1)
            ).first()
            if row is not None:
                anchors[real_code] = (row[0], float(row[1]))

        if not anchors:
            return IngestResult(
                self.name,
                "skipped",
                message=(
                    "no real CDFI assessment to anchor to; run the sse_cdfi "
                    "adapter first. Refusing to invent a level as well as a path."
                ),
            )

        end = max(obs_date for obs_date, _ in anchors.values())
        start = end - timedelta(days=int(365.25 * YEARS_OF_HISTORY))
        # Do not extrapolate the driver beyond where the real coal price ends.
        start = max(start, min(point for point, _ in monthly))
        calendar = [
            date.fromordinal(o) for o in range(start.toordinal(), end.toordinal() + 1)
        ]
        driver = _daily_driver(monthly, start, end)

        rows: list[dict] = []
        for real_code, (_, anchor_value) in sorted(anchors.items()):
            code = synthetic_code(real_code)
            path = generate_path(code, anchor_value, driver, calendar)
            for obs_date, value in zip(calendar, path):
                rows.append(
                    {
                        "series_code": code,
                        "series_kind": "route",
                        "obs_date": obs_date,
                        "value": value,
                        "unit": "usd_per_t",
                        **provenance,
                    }
                )

        written = upsert(session, RateObservation, rows, ["series_code", "obs_date"])
        return IngestResult(
            self.name,
            "ok",
            rows_written=written,
            message=(
                f"{len(anchors)} routes, {start.isoformat()} to {end.isoformat()}, "
                f"anchored to real assessments"
            ),
        )
