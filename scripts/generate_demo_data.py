"""Generate a demo weather dataset so the project can run without Kaggle data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rain_prediction.config import DEFAULT_DEMO_DATA, ensure_directories
from rain_prediction.data import save_demo_dataset


def main() -> None:
    """Create the demo CSV file used for local testing."""
    ensure_directories()
    path = save_demo_dataset(DEFAULT_DEMO_DATA)
    print(f"Demo dataset saved to: {path}")


if __name__ == "__main__":
    main()
