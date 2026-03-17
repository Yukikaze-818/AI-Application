"""Run exploratory data analysis and export plots/reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rain_prediction.config import DEFAULT_DEMO_DATA, DEFAULT_RAW_DATA, EDA_DIR, TARGET_COLUMN, ensure_directories
from rain_prediction.data import load_weather_data, prepare_target
from rain_prediction.plots import (
    plot_correlation_heatmap,
    plot_missingness,
    plot_numeric_distributions,
    plot_target_distribution,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run EDA for rain prediction.")
    parser.add_argument("--data", type=str, default=None, help="Path to the input CSV file.")
    return parser.parse_args()


def resolve_data_path(user_path: str | None) -> Path:
    """Choose the best available dataset path."""
    if user_path:
        return Path(user_path)
    if DEFAULT_RAW_DATA.exists():
        return DEFAULT_RAW_DATA
    return DEFAULT_DEMO_DATA


def main() -> None:
    """Load data, create plots, and save a compact JSON profile."""
    args = parse_args()
    ensure_directories()
    data_path = resolve_data_path(args.data)

    df = load_weather_data(data_path)
    df = prepare_target(df)

    plot_missingness(df, EDA_DIR / "missingness.png")
    plot_target_distribution(df, TARGET_COLUMN, EDA_DIR / "target_distribution.png")
    plot_numeric_distributions(df, EDA_DIR / "numeric_distributions.png")
    plot_correlation_heatmap(df, EDA_DIR / "correlation_heatmap.png")

    profile = {
        "data_path": str(data_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_rate": float(df[TARGET_COLUMN].mean()),
        "missing_percentage": df.isna().mean().sort_values(ascending=False).round(4).to_dict(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "summary": json.loads(df.describe(include="all").to_json(date_format="iso")),
    }
    with open(EDA_DIR / "eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, ensure_ascii=False)

    print(f"EDA completed. Outputs saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()

