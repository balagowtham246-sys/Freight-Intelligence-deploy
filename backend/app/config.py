"""Central configuration. Secrets come from env; everything else has a sane default."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PARQUET_DIR = DATA_DIR / "parquet"
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = f"sqlite:///{DATA_DIR / 'warehouse.db'}"

    # --- optional API keys; adapters degrade gracefully when absent ---
    eia_api_key: str | None = None
    aisstream_api_key: str | None = None

    # Bay of Bengal + ECI approaches bounding box for AIS filtering.
    # AISStream applies per-user bandwidth limits, so we never subscribe globally.
    ais_bbox: tuple[float, float, float, float] = (5.0, 78.0, 23.0, 95.0)

    http_timeout_seconds: float = 45.0
    user_agent: str = "SIH-FreightPlatform/0.1 (academic research; contact via repo)"

    # Refuse to start if a synthetic source is undisclosed. See SYNTHETIC_DATA.md.
    enforce_synthetic_disclosure: bool = True


settings = Settings()

for _d in (DATA_DIR, RAW_DIR, REFERENCE_DIR, PARQUET_DIR):
    _d.mkdir(parents=True, exist_ok=True)
