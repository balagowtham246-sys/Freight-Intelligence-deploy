"""Feature construction over the causal panel.

The strategy document's position — feature engineering beats model
architecture here — is the one this module takes. Every feature below is a
transform of a column already in the panel, and every one is lagged, so a
feature row dated *t* is knowable on the morning of *t*.

What is here, and why each earns its place:

**Market state.** Momentum and realised volatility of the target at several
windows. Freight is near-unit-root, so the level tells you little and the
*change* and its dispersion tell you most of what is learnable.

**Cross-route spreads.** The Capesize/Panamax and Panamax/Supramax ratios are
the classic substitution signal: when the ratio stretches, cargo migrates
between classes and the smaller class firms. Free from CDFI, which publishes
all three.

**Demand.** CEA days-of-burn at the import-designed plants, its level and its
slope. When the coastal plants run down, buyers take freight regardless of
price, and this is the only free daily read on that in existence.

**Congestion.** Queue length at Paradip, its slope and its z-score against a
trailing baseline. Congestion absorbs tonnage, which tightens supply.

**Commodity.** Coal price level and the NEWC-Richards Bay spread, which is the
arbitrage that decides whether Atlantic or Pacific tonnage gets pulled.

**Calendar.** Fourier annual terms plus explicit monsoon and cyclone-season
flags. These are cheap and they encode real, repeating physical constraints on
the Bay of Bengal.

Missing data is left missing. LightGBM handles NaN natively and splits on
missingness, which is more honest than imputing a number nobody observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import log_returns

# Windows in calendar days. The panel is daily, so these are literal.
MOMENTUM_WINDOWS = (5, 10, 20, 60)
VOLATILITY_WINDOWS = (5, 20, 60)
SLOPE_WINDOWS = (7, 30)
ZSCORE_WINDOW = 60

# Bay of Bengal cyclone season: primary Oct-Dec, secondary Apr-May.
CYCLONE_MONTHS_PRIMARY = (10, 11, 12)
CYCLONE_MONTHS_SECONDARY = (4, 5)
# Indian southwest monsoon.
MONSOON_MONTHS = (6, 7, 8, 9)


def _safe_zscore(values: pd.Series, window: int) -> pd.Series:
    rolling = values.rolling(window, min_periods=max(5, window // 4))
    spread = rolling.std()
    return (values - rolling.mean()) / spread.where(spread > 0)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Seasonality that encodes real physical constraints, not fitted wiggles."""
    day_of_year = index.dayofyear.to_numpy(dtype=float)
    frame = pd.DataFrame(index=index)

    # Two harmonics is enough for an annual cycle; more starts fitting noise.
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * day_of_year / 365.25
        frame[f"cal_sin{harmonic}"] = np.sin(angle)
        frame[f"cal_cos{harmonic}"] = np.cos(angle)

    month = index.month
    frame["cal_cyclone_primary"] = month.isin(CYCLONE_MONTHS_PRIMARY).astype(float)
    frame["cal_cyclone_secondary"] = month.isin(CYCLONE_MONTHS_SECONDARY).astype(float)
    frame["cal_monsoon"] = month.isin(MONSOON_MONTHS).astype(float)
    # Indian pre-monsoon stockpiling, when import demand front-loads.
    frame["cal_prestock"] = month.isin((3, 4, 5)).astype(float)
    return frame


def target_features(panel: pd.DataFrame, target_series: str) -> pd.DataFrame:
    """Momentum, volatility and level features of the forecast target."""
    values = panel[target_series]
    returns = log_returns(values)
    frame = pd.DataFrame(index=panel.index)

    frame["tgt_level"] = values
    frame["tgt_log_level"] = np.log(values.where(values > 0))

    for window in MOMENTUM_WINDOWS:
        frame[f"tgt_mom_{window}"] = np.log(
            values.where(values > 0) / values.where(values > 0).shift(window)
        )

    for window in VOLATILITY_WINDOWS:
        frame[f"tgt_vol_{window}"] = returns.rolling(
            window, min_periods=max(3, window // 3)
        ).std() * np.sqrt(252)

    # Where the level sits inside its own recent range: a cheap, robust
    # mean-reversion signal that does not assume a distribution.
    frame["tgt_z_60"] = _safe_zscore(values, ZSCORE_WINDOW)
    rolling_min = values.rolling(252, min_periods=60).min()
    rolling_max = values.rolling(252, min_periods=60).max()
    span = (rolling_max - rolling_min).where(lambda s: s > 0)
    frame["tgt_pct_of_range_252"] = (values - rolling_min) / span

    return frame


def cross_route_features(panel: pd.DataFrame, target_series: str = "") -> pd.DataFrame:
    """Substitution signals between vessel classes.

    When the Capesize/Panamax ratio stretches, cargo that could go either way
    migrates to the cheaper class and firms it. CDFI publishes all three sizes
    on comparable coal lanes, so this costs nothing to compute.

    The spreads are taken from the *same namespace* as the target. A forecast
    of a ``SYN_`` development series must be built from ``SYN_`` companions:
    mixing a disclosed synthetic target with real companion routes would pair
    an eight-year path against a two-day one, and the ratio would be almost
    entirely missing. It would also quietly mix the two kinds of data inside a
    single feature, which is the thing this codebase is built not to do.
    """
    frame = pd.DataFrame(index=panel.index)
    prefix = "SYN_" if target_series.startswith("SYN_") else ""
    pairs = {
        "cape_kmax": ("CDFI_NEWCASTLE_ZHOUSHAN_130K", "CDFI_HAYPOINT_ZHOUSHAN_85K"),
        "pmax_smax": ("CDFI_SAMARINDA_GUANGZHOU_70K", "CDFI_TABONEO_GUANGZHOU_50K"),
    }
    for name, (numerator, denominator) in pairs.items():
        top, bottom_code = f"{prefix}{numerator}", f"{prefix}{denominator}"
        if top in panel.columns and bottom_code in panel.columns:
            bottom = panel[bottom_code].where(panel[bottom_code] > 0)
            ratio = panel[top] / bottom
            frame[f"spread_{name}"] = ratio
            frame[f"spread_{name}_z"] = _safe_zscore(ratio, ZSCORE_WINDOW)
    return frame


def commodity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Coal price level and the Pacific/Atlantic arbitrage spread."""
    frame = pd.DataFrame(index=panel.index)

    if "COAL_AU_NEWC" in panel.columns:
        newc = panel["COAL_AU_NEWC"]
        frame["coal_newc"] = newc
        frame["coal_newc_mom_60"] = np.log(
            newc.where(newc > 0) / newc.where(newc > 0).shift(60)
        )
        if "COAL_ZA_RB" in panel.columns:
            richards = panel["COAL_ZA_RB"]
            frame["coal_newc_rb_spread"] = newc - richards
            frame["coal_newc_rb_z"] = _safe_zscore(newc - richards, 180)

    if "CRUDE_BRENT" in panel.columns:
        brent = panel["CRUDE_BRENT"]
        frame["brent"] = brent
        frame["brent_mom_60"] = np.log(
            brent.where(brent > 0) / brent.where(brent > 0).shift(60)
        )
        # Bunkers are the cost side of every voyage; Brent is the real,
        # long-history proxy the platform holds for them.
        if "COAL_AU_NEWC" in panel.columns:
            frame["coal_brent_ratio"] = panel["COAL_AU_NEWC"] / brent.where(brent > 0)

    return frame


def congestion_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Queue depth, its slope and how unusual it is.

    Congestion is a supply signal: tonnage sitting at anchorage is tonnage
    removed from the market. It is also the input to the demurrage term in the
    timing engine, which is where it earns most of its value.
    """
    frame = pd.DataFrame(index=panel.index)
    queue_columns = [column for column in panel.columns if column.startswith("queue_")]
    if not queue_columns:
        return frame

    for column in queue_columns:
        frame[column] = panel[column]

    waiting = [column for column in queue_columns if column.endswith("_waiting")]
    if waiting:
        # min_count=1 is load-bearing. Without it pandas sums an all-NaN row to
        # 0.0, which would put a fabricated "queue was empty" reading on every
        # date before the berthing-programme archive begins - a made-up feature
        # that looks exactly like a real one.
        total = panel[waiting].sum(axis=1, min_count=1)
        frame["queue_waiting_total"] = total
        for window in SLOPE_WINDOWS:
            frame[f"queue_slope_{window}"] = total - total.shift(window)
        frame["queue_z_60"] = _safe_zscore(total, ZSCORE_WINDOW)

    return frame


def demand_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Indian coal-stock urgency, from the CEA daily report."""
    frame = pd.DataFrame(index=panel.index)
    for column in ("dob_all", "dob_import", "dob_critical_count"):
        if column in panel.columns:
            frame[column] = panel[column]
            for window in SLOPE_WINDOWS:
                frame[f"{column}_slope_{window}"] = (
                    panel[column] - panel[column].shift(window)
                )
    return frame


def build_features(
    panel: pd.DataFrame, target_series: str, *, lag_days: int = 1
) -> pd.DataFrame:
    """Assemble the design matrix.

    Every column is shifted by ``lag_days`` at the end. This is the structural
    guarantee: a feature row dated *t* contains only information from *t - 1*
    and earlier, so nothing downstream has to remember to lag anything.

    Calendar features are deliberately exempt — the day of the year on date
    *t* is known on date *t*, and lagging it would just blur it.
    """
    parts = [
        target_features(panel, target_series),
        cross_route_features(panel, target_series),
        commodity_features(panel),
        congestion_features(panel),
        demand_features(panel),
    ]
    design = pd.concat([part for part in parts if not part.empty], axis=1)
    design = design.shift(lag_days)

    calendar = calendar_features(panel.index)
    design = pd.concat([design, calendar], axis=1)

    # Drop columns that are entirely missing: with a young archive some feeds
    # have no overlap with the target yet, and an all-NaN column tells the
    # model nothing while making the feature list misleading.
    return design.dropna(axis=1, how="all")


def feature_names(design: pd.DataFrame) -> list[str]:
    return [str(column) for column in design.columns]
