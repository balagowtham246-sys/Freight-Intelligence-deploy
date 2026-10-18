"""Volatility forecasting: GARCH(1,1) and HAR-RV.

Volatility deserves its own model because it is genuinely more forecastable
than the level. Freight returns are close to unforecastable; freight
*variance* is strongly autocorrelated and clusters. That asymmetry is why the
strategy document puts volatility forecasting next to nowcasting as the place
where real skill exists.

Three things consume this module:

1. **The ``MARKET_VOL`` alert.** Conditional volatility high against its own
   history is the early warning the brief asks for under "market volatility".
2. **The timing engine's urgency term.** High volatility raises the option
   value of waiting *and* the cost of being wrong. The optimal-stopping
   solution needs the variance, not just the mean.
3. **Scenario generation.** Scenario dispersion is set from the conditional
   variance, so a calm market produces tight scenarios and a violent one does
   not.

Two estimators, deliberately:

* **GARCH(1,1)** on daily log returns. The workhorse, with a persistence
  parameter (alpha + beta) that is itself informative — close to one means
  shocks decay slowly and today's turbulence will still be here next month.
* **HAR-RV**, a linear regression of realised variance on its own 1, 5 and 22
  day averages. Crude, transparent, and famously hard to beat. It is the
  baseline that keeps the GARCH fit honest.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# HAR-RV component windows: daily, weekly, monthly.
HAR_WINDOWS = (1, 5, 22)

MIN_OBSERVATIONS = 120


@dataclass
class VolatilityState:
    """Current and forecast volatility, annualised."""

    current_annual_vol: float
    forecast_annual_vol: float
    long_run_annual_vol: float
    percentile_of_history: float
    persistence: float
    model_name: str
    n_observations: int

    @property
    def is_elevated(self) -> bool:
        """High against its own history, not against an absolute threshold.

        An absolute threshold would fire permanently on a violent lane and
        never on a quiet one.
        """
        return self.percentile_of_history >= 0.90

    def as_dict(self) -> dict:
        return {
            "current_annual_vol": round(self.current_annual_vol, 4),
            "forecast_annual_vol": round(self.forecast_annual_vol, 4),
            "long_run_annual_vol": round(self.long_run_annual_vol, 4),
            "percentile_of_history": round(self.percentile_of_history, 4),
            "persistence": round(self.persistence, 4),
            "model": self.model_name,
            "n_observations": self.n_observations,
            "is_elevated": self.is_elevated,
        }


def realised_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling standard deviation of log returns."""
    return returns.rolling(window, min_periods=max(3, window // 3)).std() * np.sqrt(
        TRADING_DAYS
    )


def har_rv(returns: pd.Series, horizon_days: int = 1) -> float | None:
    """HAR-RV forecast of annualised volatility.

    Regresses realised variance on its own daily, weekly and monthly averages.
    Returns None rather than a number when the history is too short.
    """
    clean = returns.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return None

    variance = clean.pow(2)
    frame = pd.DataFrame({"y": variance})
    for window in HAR_WINDOWS:
        frame[f"rv_{window}"] = variance.rolling(window).mean()

    frame["target"] = (
        variance.rolling(horizon_days).mean().shift(-horizon_days)
        if horizon_days > 1
        else variance.shift(-1)
    )
    frame = frame.dropna()
    if len(frame) < 60:
        return None

    columns = [f"rv_{window}" for window in HAR_WINDOWS]
    design = np.column_stack(
        [np.ones(len(frame))] + [frame[column].to_numpy() for column in columns]
    )
    coefficients, *_ = np.linalg.lstsq(design, frame["target"].to_numpy(), rcond=None)

    latest = np.array(
        [1.0] + [float(variance.rolling(window).mean().iloc[-1]) for window in HAR_WINDOWS]
    )
    predicted_variance = float(latest @ coefficients)
    if not np.isfinite(predicted_variance) or predicted_variance <= 0:
        return None
    return float(np.sqrt(predicted_variance * TRADING_DAYS))


def fit_garch(returns: pd.Series, horizon_days: int = 20) -> VolatilityState | None:
    """Fit GARCH(1,1) and summarise the current volatility state.

    The ``arch`` package works in percentage returns, so returns are scaled by
    100 going in and the resulting volatility scaled back on the way out.
    """
    from arch import arch_model

    clean = returns.dropna()
    clean = clean[np.isfinite(clean)]
    if len(clean) < MIN_OBSERVATIONS:
        return None

    scaled = clean * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Constant", dist="t")
            result = model.fit(disp="off", show_warning=False)
            forecast = result.forecast(horizon=horizon_days, reindex=False)
    except Exception:
        return None

    conditional = result.conditional_volatility / 100.0
    current = float(conditional.iloc[-1]) * np.sqrt(TRADING_DAYS)

    # Average variance over the horizon, then annualise.
    variance_path = np.asarray(forecast.variance.iloc[-1]) / 10_000.0
    forecast_vol = float(np.sqrt(variance_path.mean() * TRADING_DAYS))

    params = result.params
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    omega = float(params.get("omega", 0.0)) / 10_000.0
    persistence = alpha + beta

    long_run = (
        float(np.sqrt(omega / max(1.0 - persistence, 1e-6) * TRADING_DAYS))
        if persistence < 1.0
        else float("nan")
    )

    annualised = conditional * np.sqrt(TRADING_DAYS)
    percentile = float((annualised <= annualised.iloc[-1]).mean())

    return VolatilityState(
        current_annual_vol=current,
        forecast_annual_vol=forecast_vol,
        long_run_annual_vol=long_run,
        percentile_of_history=percentile,
        persistence=persistence,
        model_name="garch(1,1)-t",
        n_observations=int(len(clean)),
    )


def volatility_state(returns: pd.Series, horizon_days: int = 20) -> VolatilityState | None:
    """GARCH where it fits, HAR-RV where it does not, None where neither does.

    Falling back rather than failing matters: the timing engine needs *a*
    variance to size its scenarios, and a transparent HAR-RV number is far
    better than silently assuming a constant.
    """
    state = fit_garch(returns, horizon_days=horizon_days)
    if state is not None:
        return state

    har = har_rv(returns, horizon_days=horizon_days)
    if har is None:
        return None

    realised = realised_volatility(returns, window=20).dropna()
    if realised.empty:
        return None

    return VolatilityState(
        current_annual_vol=float(realised.iloc[-1]),
        forecast_annual_vol=har,
        long_run_annual_vol=float(realised.mean()),
        percentile_of_history=float((realised <= realised.iloc[-1]).mean()),
        persistence=float("nan"),
        model_name="har-rv",
        n_observations=int(returns.dropna().size),
    )
