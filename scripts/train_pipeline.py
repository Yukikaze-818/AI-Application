"""Train, tune, evaluate, and export the rain prediction models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rain_prediction.config import (
    DEFAULT_DEMO_DATA,
    DEFAULT_RAW_DATA,
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_directories,
)
from rain_prediction.data import load_weather_data, prepare_target, train_valid_test_split
from rain_prediction.modeling import (
    fit_and_tune_models,
    maybe_calibrate_best_model,
    plot_confusion_matrix_and_report,
    plot_partial_dependence,
    plot_permutation_importance,
    plot_roc_pr_curves,
    run_error_analysis,
    save_best_model,
    save_model_comparison,
    save_training_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train the rain prediction pipeline.")
    parser.add_argument("--data", type=str, default=None, help="Path to the input CSV file.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for row count during testing.")
    return parser.parse_args()


def resolve_data_path(user_path: str | None) -> Path:
    """Choose the best available dataset path."""
    if user_path:
        return Path(user_path)
    if DEFAULT_RAW_DATA.exists():
        return DEFAULT_RAW_DATA
    return DEFAULT_DEMO_DATA


def main() -> None:
    """Execute the full model training workflow."""
    args = parse_args()
    ensure_directories()
    data_path = resolve_data_path(args.data)

    df = load_weather_data(data_path)
    df = prepare_target(df)

    if args.max_rows is not None and len(df) > args.max_rows:
        df = (
            df.groupby("RainTomorrow", group_keys=False)
            .apply(lambda frame: frame.sample(n=max(1, int(args.max_rows * len(frame) / len(df))), random_state=42))
            .reset_index(drop=True)
        )
    x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(df)

    results, best_result = fit_and_tune_models(x_train, y_train, x_valid, y_valid)
    best_result = maybe_calibrate_best_model(best_result, x_train, y_train, x_valid, y_valid)

    comparison = save_model_comparison(results + [best_result], METRICS_DIR / "model_comparison.csv")
    save_best_model(best_result.estimator, MODELS_DIR / "best_model.joblib")

    test_metrics = plot_roc_pr_curves(best_result.estimator, x_test, y_test, METRICS_DIR)
    plot_confusion_matrix_and_report(best_result.estimator, x_test, y_test, METRICS_DIR)
    plot_permutation_importance(
        best_result.estimator,
        x_test,
        y_test,
        REPORTS_DIR / "feature_importance.png",
    )
    plot_partial_dependence(
        best_result.estimator,
        x_test,
        REPORTS_DIR / "partial_dependence.png",
    )
    run_error_analysis(
        best_result.estimator,
        x_test,
        y_test,
        REPORTS_DIR / "error_analysis.csv",
    )
    save_training_summary(best_result, test_metrics, MODELS_DIR / "training_summary.json")

    pd.DataFrame({"xgboost_available": [comparison["model"].str.contains("xgboost").any()]}).to_csv(
        REPORTS_DIR / "environment_flags.csv",
        index=False,
    )

    print(f"Training complete. Best model: {best_result.name}")
    print(f"Model saved to: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
