"""Central configuration for the rain prediction project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

DEFAULT_RAW_DATA = RAW_DATA_DIR / "weatherAUS.csv"
DEFAULT_DEMO_DATA = RAW_DATA_DIR / "weatherAUS_demo.csv"
TARGET_COLUMN = "RainTomorrow"
DATE_COLUMN = "Date"


def ensure_directories() -> None:
    """Create all required folders if they do not already exist."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EDA_DIR,
        METRICS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
