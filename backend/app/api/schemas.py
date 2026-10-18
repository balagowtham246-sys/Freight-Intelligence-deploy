"""Request and response models for the API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ProvenanceOut(BaseModel):
    """Attribution for any figure the API returns.

    Present on every data-bearing response so the UI can badge each number
    with where it came from and whether it is synthetic.
    """

    source_name: str
    source_url: str | None = None
    retrieved_at: str | None = None
    is_synthetic: bool = False


class FeasibilityRequest(BaseModel):
    discharge_port: str = Field(..., description="Port code, e.g. INHAL")
    required_tonnes: float = Field(..., gt=0)
    load_port: str | None = Field(None, description="Optional origin port code")
    cargo: str = Field("thermal_coal_mid_cv")
    bunkers_rob_t: float = Field(1200.0, ge=0)
    tide_height_m: float = Field(0.0, description="Predicted height of tide at ETA")


class LoadableRequest(BaseModel):
    vessel_class: str
    port_code: str
    berth_name: str | None = None
    cargo: str | None = None
    bunkers_rob_t: float = 1200.0
    tide_height_m: float = 0.0
    ukc_policy_m: float = 0.0


class QueueEntry(BaseModel):
    vessel_name: str
    status: str
    cargo: str | None = None
    cargo_tonnes: float | None = None
    berth: str | None = None
    arrival_date: date | None = None
    waiting_days: float | None = None
    loa_m: float | None = None
    draft_m: float | None = None


class PortWatchOut(BaseModel):
    port_code: str
    snapshot_date: date | None
    working: int
    waiting: int
    expected: int
    median_waiting_days: float | None
    queue: list[QueueEntry]
    provenance: ProvenanceOut


class SeriesPoint(BaseModel):
    date: date
    value: float


class SeriesOut(BaseModel):
    series_code: str
    unit: str
    points: list[SeriesPoint]
    provenance: ProvenanceOut


class DataFreshnessOut(BaseModel):
    adapter: str
    status: str
    rows_written: int
    finished_at: str | None
    used_cache: bool
    message: str | None = None


# --------------------------------------------------------------- model output


class ModelProvenanceOut(BaseModel):
    """Attribution for a *modelled* figure.

    Deliberately a different shape from ``ProvenanceOut``. A forecast is not an
    observation: it has no publisher and no retrieval time, and flagging it
    synthetic would conflate a model with a fabricated feed. See the "Three
    kinds of number" section of SYNTHETIC_DATA.md.
    """

    model_name: str
    model_version: str
    trained_through: date
    n_train: int = 0
    input_series: list[str] = Field(default_factory=list)
    # True when any input was a disclosed synthetic series, so the UI can badge
    # a forecast that inherits that taint.
    trained_on_synthetic: bool = False


class ForecastBand(BaseModel):
    """One horizon of a fan chart."""

    horizon_days: int
    date: date
    quantiles: dict[str, float]


class SkillOut(BaseModel):
    """Measured performance at one horizon, and whether it earns an edge."""

    horizon_days: int
    n_folds: int
    point_skill: float | None = None
    distributional_skill: float | None = None
    mase: float | None = None
    coverage_80: float | None = None
    has_edge: bool = False
    computed_on_synthetic: bool = False


class ForecastOut(BaseModel):
    """History plus the calibrated predictive distribution."""

    target_key: str
    unit: str
    history: list[SeriesPoint]
    bands: list[ForecastBand]
    skill: list[SkillOut]
    signal_strength: str
    model: ModelProvenanceOut
    # Present when the forecaster has no measured edge, explaining the gap
    # rather than leaving a bare chart to imply confidence.
    caveat: str | None = None


class ForecastTargetOut(BaseModel):
    series_code: str
    series_kind: str
    unit: str
    observations: int
    first: date
    last: date
    is_synthetic: bool
    forecastable: bool
    reason: str | None = None


class WaitingForecastOut(BaseModel):
    port_code: str
    queue_length: float
    expected_days: float
    median_days: float
    p80_days: float
    p95_days: float
    survival: list[float] = Field(default_factory=list)
    model: str
    n_train_calls: int
    explanation: str
    provenance: ProvenanceOut


# ------------------------------------------------------------------- timing


class TimingRequest(BaseModel):
    """A cargo requirement to time the market for."""

    discharge_port: str = Field(..., description="Port code, e.g. INPRT")
    required_tonnes: float = Field(..., gt=0)
    cargo: str = Field("thermal_coal_mid_cv")
    load_port: str | None = None
    not_later_than: date = Field(..., description="Latest acceptable fixing date")
    target_series: str | None = Field(
        None, description="Freight series to price against; defaults to the deepest"
    )
    risk_aversion: float = Field(
        0.35, ge=0.0, le=5.0, description="Weight on the CVaR term"
    )
    # Disclosed defaults rather than invented figures. See SYNTHETIC_DATA.md
    # entries 4 and 8.
    demurrage_usd_per_day: float = Field(22_000.0, gt=0)
    bunker_usd_per_tonne: float = Field(560.0, gt=0)
    n_scenarios: int = Field(1000, ge=100, le=5000)


# --------------------------------------------------------------------- risk


class AlertOut(BaseModel):
    id: int
    code: str
    severity: str
    scope: str
    port_code: str | None = None
    series_code: str | None = None
    raised_date: date
    cleared_date: date | None = None
    last_seen_date: date
    title: str
    body: str
    action: str
    evidence: dict = Field(default_factory=dict)


class MonitorOut(BaseModel):
    """A watch and its current reading, whether or not it has fired."""

    code: str
    label: str
    value: float | None = None
    display: str
    threshold: float | None = None
    threshold_label: str
    triggered: bool


class RiskOut(BaseModel):
    as_of: date
    active: list[AlertOut]
    recently_cleared: list[AlertOut]
    monitors: list[MonitorOut]
