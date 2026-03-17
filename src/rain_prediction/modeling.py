"""Model building, tuning, evaluation, and explainability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class TrainingResult:
    """Simple container for tracking a fitted model and its metrics."""

    name: str
    estimator: object
    cv_roc_auc: float
    valid_roc_auc: float
    valid_average_precision: float
    valid_precision: float
    valid_recall: float
    valid_brier: float


def make_model_candidates(df: pd.DataFrame, random_state: int = 42) -> Dict[str, Tuple[Pipeline, Dict]]:
    """Create pipelines and hyperparameter spaces for each candidate model."""
    from .features import make_feature_transformer, make_preprocessor

    feature_transformer = make_feature_transformer()
    preprocessor = make_preprocessor(df)

    candidates: Dict[str, Tuple[Pipeline, Dict]] = {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("features", feature_transformer),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {
                "model__C": np.logspace(-2, 1.2, 12),
                "model__penalty": ["l1", "l2"],
            },
        ),
        "decision_tree": (
            Pipeline(
                steps=[
                    ("features", feature_transformer),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        DecisionTreeClassifier(
                            random_state=random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [3, 5, 8, 12, 18, None],
                "model__min_samples_split": [2, 5, 10, 20, 40],
                "model__min_samples_leaf": [1, 2, 4, 8, 12],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("features", feature_transformer),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            random_state=random_state,
                            class_weight="balanced",
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["xgboost"] = (
            Pipeline(
                steps=[
                    ("features", feature_transformer),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBClassifier(
                            random_state=random_state,
                            n_estimators=250,
                            learning_rate=0.05,
                            max_depth=5,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
        )

    return candidates


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute the key metrics required for the challenge."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def fit_and_tune_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int = 42,
) -> Tuple[List[TrainingResult], TrainingResult]:
    """Tune each model using randomized search and keep the best candidate."""
    candidates = make_model_candidates(pd.concat([x_train, y_train], axis=1), random_state=random_state)

    tune_x = x_train
    tune_y = y_train
    cv_splits = 5
    search_iterations = 10
    large_dataset = len(x_train) > 50000

    if large_dataset:
        tune_size = min(20000, len(x_train))
        sampled_index = (
            pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True).rename("target")], axis=1)
            .groupby("target", group_keys=False)
            .apply(lambda frame: frame.sample(frac=min(1.0, tune_size / len(x_train)), random_state=random_state))
            .index
        )
        sampled_index = sampled_index[:tune_size]
        tune_x = x_train.iloc[sampled_index].copy()
        tune_y = y_train.iloc[sampled_index].copy()
        cv_splits = 3
        search_iterations = 2

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results: List[TrainingResult] = []

    for name, (pipeline, param_grid) in candidates.items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(search_iterations, int(np.prod([len(v) for v in param_grid.values()]))),
            scoring="roc_auc",
            n_jobs=1,
            cv=cv,
            random_state=random_state,
            refit=True,
            verbose=0,
        )
        search.fit(tune_x, tune_y)

        best_estimator = clone(pipeline).set_params(**search.best_params_)

        fit_x = x_train
        fit_y = y_train
        if large_dataset and name in {"decision_tree", "random_forest", "xgboost"}:
            fit_x = tune_x
            fit_y = tune_y

        best_estimator.fit(fit_x, fit_y)

        cv_score = float(search.best_score_)
        y_prob = best_estimator.predict_proba(x_valid)[:, 1]
        metrics = evaluate_predictions(y_valid, y_prob)

        result = TrainingResult(
            name=name,
            estimator=best_estimator,
            cv_roc_auc=cv_score,
            valid_roc_auc=float(metrics["roc_auc"]),
            valid_average_precision=float(metrics["average_precision"]),
            valid_precision=float(metrics["precision"]),
            valid_recall=float(metrics["recall"]),
            valid_brier=float(metrics["brier_score"]),
        )
        results.append(result)

    best_result = sorted(
        results,
        key=lambda item: (item.valid_roc_auc, item.valid_average_precision),
        reverse=True,
    )[0]
    return results, best_result


def maybe_calibrate_best_model(
    best_result: TrainingResult,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> TrainingResult:
    """Calibrate the best model if it is tree-based and calibration improves Brier score."""
    if best_result.name not in {"decision_tree", "random_forest", "xgboost"}:
        return best_result

    calibrated = CalibratedClassifierCV(estimator=clone(best_result.estimator), method="sigmoid", cv=3)
    calibrated.fit(x_train, y_train)
    y_prob = calibrated.predict_proba(x_valid)[:, 1]
    calibrated_metrics = evaluate_predictions(y_valid, y_prob)

    if calibrated_metrics["brier_score"] <= best_result.valid_brier:
        return TrainingResult(
            name=f"{best_result.name}_calibrated",
            estimator=calibrated,
            cv_roc_auc=best_result.cv_roc_auc,
            valid_roc_auc=float(calibrated_metrics["roc_auc"]),
            valid_average_precision=float(calibrated_metrics["average_precision"]),
            valid_precision=float(calibrated_metrics["precision"]),
            valid_recall=float(calibrated_metrics["recall"]),
            valid_brier=float(calibrated_metrics["brier_score"]),
        )
    return best_result


def save_model_comparison(results: List[TrainingResult], output_path: Path) -> pd.DataFrame:
    """Persist the comparison table to CSV."""
    comparison = pd.DataFrame(
        [
            {
                "model": result.name,
                "cv_roc_auc": result.cv_roc_auc,
                "valid_roc_auc": result.valid_roc_auc,
                "valid_average_precision": result.valid_average_precision,
                "valid_precision": result.valid_precision,
                "valid_recall": result.valid_recall,
                "valid_brier": result.valid_brier,
            }
            for result in results
        ]
    ).sort_values(by="valid_roc_auc", ascending=False)
    comparison.to_csv(output_path, index=False)
    return comparison


def plot_roc_pr_curves(
    estimator: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> Dict[str, float]:
    """Generate ROC and PR curve plots."""
    y_prob = estimator.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {metrics['average_precision']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()

    return metrics


def plot_confusion_matrix_and_report(
    estimator: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Save a confusion matrix image and a classification report."""
    y_prob = estimator.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def run_error_analysis(
    estimator: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Inspect false positives and false negatives to understand failure modes."""
    analysis = x_test.copy()
    analysis["actual"] = y_test.values
    analysis["predicted_probability"] = estimator.predict_proba(x_test)[:, 1]
    analysis["predicted"] = (analysis["predicted_probability"] >= 0.5).astype(int)

    conditions = [
        (analysis["actual"] == 0) & (analysis["predicted"] == 1),
        (analysis["actual"] == 1) & (analysis["predicted"] == 0),
    ]
    labels = ["false_positive", "false_negative"]
    analysis["error_type"] = np.select(conditions, labels, default="correct")

    summary_frames = []
    for error_label in ["false_positive", "false_negative"]:
        subset = analysis[analysis["error_type"] == error_label]
        if subset.empty:
            continue

        numeric_summary = subset.select_dtypes(include=["number"]).mean().to_dict()
        record = {"error_type": error_label, "count": int(len(subset))}
        record.update({f"mean_{key}": value for key, value in numeric_summary.items()})

        if "Location" in subset.columns:
            top_locations = subset["Location"].value_counts().head(3).to_dict()
            record["top_locations"] = json.dumps(top_locations, ensure_ascii=False)

        summary_frames.append(record)

    summary = pd.DataFrame(summary_frames)
    summary.to_csv(output_path, index=False)
    return analysis


def plot_permutation_importance(
    estimator: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Compute permutation importance on the raw input columns."""
    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="roc_auc",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    importance.to_csv(output_path.with_suffix(".csv"), index=False)

    top_features = importance.head(12).sort_values(by="importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"])
    plt.xlabel("Permutation Importance (ROC-AUC drop)")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return importance


def plot_partial_dependence(
    estimator: object,
    x_test: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create partial dependence plots for the most interpretable numeric features."""
    preferred_features = [
        feature
        for feature in ["Humidity3pm", "Pressure3pm", "Sunshine", "Rainfall"]
        if feature in x_test.columns
    ]
    if len(preferred_features) < 2:
        return

    PartialDependenceDisplay.from_estimator(
        estimator,
        x_test,
        features=preferred_features[:4],
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_best_model(estimator: object, output_path: Path) -> None:
    """Serialize the best model to disk."""
    joblib.dump(estimator, output_path)


def save_training_summary(best_result: TrainingResult, test_metrics: Dict[str, float], output_path: Path) -> None:
    """Write a compact JSON summary for the Streamlit app and quick inspection."""
    summary = {
        "best_model": best_result.name,
        "validation_roc_auc": best_result.valid_roc_auc,
        "validation_average_precision": best_result.valid_average_precision,
        "validation_precision": best_result.valid_precision,
        "validation_recall": best_result.valid_recall,
        "validation_brier_score": best_result.valid_brier,
        "test_metrics": test_metrics,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

