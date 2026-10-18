"""Write metrics, plots, predictions, and the saved model pipelines."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import LLLL, STUDENT_NAME, TEST_SIZE, VERIFICATION_CODE


def _builtin(value):
    """Convert NumPy values so the JSON files stay stable and readable."""
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 6)
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_json(path: Path, payload: dict) -> None:
    """Write one metrics file with the verification code preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_builtin(payload), indent=2) + "\n", encoding="utf-8")


def _identity(fingerprint: str, train_rows: int, test_rows: int) -> dict[str, object]:
    return {
        "verification_code": VERIFICATION_CODE,
        "student_name": STUDENT_NAME,
        "seed": LLLL,
        "dataset_sha256": fingerprint,
        "test_size": TEST_SIZE,
        "random_state": LLLL,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }


def write_metrics(results: dict, fingerprint: str, artifact_dir: Path) -> None:
    """Write the three required metrics files."""
    shared = _identity(fingerprint, results["train_rows"], results["test_rows"])
    regression = results["regression"]
    classification = results["classification"]
    clustering = results["clustering"]
    write_json(
        artifact_dir / "regression_metrics.json",
        {
            **shared,
            "target": "yield_tons",
            "baseline": regression["baseline"],
            "candidates": regression["candidates"],
            "selected_model": regression["selected_model"],
            "selection_reason": regression["selection_reason"],
            "notable_test_error": regression["notable_test_error"],
        },
    )
    write_json(
        artifact_dir / "classification_metrics.json",
        {
            **shared,
            "target": "high_risk",
            "positive_class": 1,
            "excluded_inputs": ["yield_tons", "yield_per_ha", "high_risk"],
            "baseline": classification["baseline"],
            "candidates": classification["candidates"],
            "selected_model": classification["selected_model"],
            "selection_reason": classification["selection_reason"],
            "notable_test_error": classification["notable_test_error"],
        },
    )
    write_json(
        artifact_dir / "clustering_report.json",
        {
            **shared,
            "features": "growing conditions only; yield and high_risk were excluded",
            "k_scores": clustering["k_scores"],
            "selected_k": clustering["selected_k"],
            "selection_reason": clustering["selection_reason"],
            "fit_on": clustering["fit_on"],
            "scaled_centers": clustering["scaled_centers"],
            "profiles": clustering["profiles"],
            "plot_note": (
                "clusters.png shows a two-component PCA of the scaled training features "
                "for display. Membership comes from KMeans on the full scaled feature set."
            ),
        },
    )


def write_predictions(predictions, artifact_dir: Path) -> None:
    """Save the held-out predictions sorted by row id."""
    path = artifact_dir / "test_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")


def _save_plot(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def write_plots(results: dict, plot_dir: Path) -> None:
    """Save residual, confusion-matrix, and cluster plots."""
    predictions = results["predictions"]
    code = VERIFICATION_CODE

    residual_figure, residual_axis = plt.subplots()
    residual_axis.scatter(
        predictions["predicted_yield_tons"],
        predictions["yield_error"],
        alpha=0.85,
        color="#1b4f72",
    )
    residual_axis.axhline(0.0, color="#922b21", linewidth=1)
    residual_axis.set_xlabel("Predicted yield (tons)")
    residual_axis.set_ylabel("Actual minus predicted (tons)")
    residual_axis.set_title(f"Held-out yield residuals ({code})")
    _save_plot(residual_figure, plot_dir / "residuals.png")

    matrix = results["classification"]["candidates"]
    selected_name = results["classification"]["selected_model"]
    selected = next(row for row in matrix if row["name"] == selected_name)
    confusion_figure, confusion_axis = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=np.array(selected["confusion_matrix"]),
        display_labels=["low risk", "high risk"],
    ).plot(ax=confusion_axis, colorbar=False, cmap="Blues")
    confusion_axis.set_title(f"{selected_name} on held-out farms ({code})")
    _save_plot(confusion_figure, plot_dir / "confusion_matrix.png")

    clustering = results["clustering"]
    transformed = clustering["transformed_train"]
    display = PCA(n_components=2, random_state=LLLL).fit_transform(transformed)
    cluster_figure, cluster_axis = plt.subplots()
    labels = np.asarray(clustering["train_labels"])
    for cluster_id in range(int(clustering["selected_k"])):
        mask = labels == cluster_id
        cluster_axis.scatter(display[mask, 0], display[mask, 1], label=f"Cluster {cluster_id}", alpha=0.85)
    cluster_axis.set_xlabel("Display component 1")
    cluster_axis.set_ylabel("Display component 2")
    cluster_axis.set_title(f"Training farms, K={clustering['selected_k']} ({code})")
    cluster_axis.legend(frameon=False)
    _save_plot(cluster_figure, plot_dir / "clusters.png")


def save_models(results: dict, model_dir: Path, fingerprint: str) -> None:
    """Save the three selected pipelines and a short manifest."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["regression"]["pipeline"], model_dir / "regression_pipeline.joblib")
    joblib.dump(results["classification"]["pipeline"], model_dir / "classification_pipeline.joblib")
    joblib.dump(results["clustering"]["pipeline"], model_dir / "clustering_pipeline.joblib")
    write_json(
        model_dir / "manifest.json",
        {
            "verification_code": VERIFICATION_CODE,
            "dataset_sha256": fingerprint,
            "regression_model": "models/regression_pipeline.joblib",
            "classification_model": "models/classification_pipeline.joblib",
            "clustering_model": "models/clustering_pipeline.joblib",
            "selected_regression": results["regression"]["selected_model"],
            "selected_classification": results["classification"]["selected_model"],
            "selected_k": results["clustering"]["selected_k"],
            "feature_columns": [
                "rainfall_mm",
                "avg_temp_c",
                "soil_moisture_pct",
                "fertilizer_kg",
                "field_size_ha",
                "season",
                "cooperative_support",
            ],
        },
    )


def write_training_artifacts(results: dict, fingerprint: str, project_root: Path) -> list[Path]:
    """Regenerate every modeling artifact and return the written paths."""
    artifact_dir = project_root / "artifacts"
    write_metrics(results, fingerprint, artifact_dir)
    write_predictions(results["predictions"], artifact_dir)
    write_plots(results, artifact_dir / "plots")
    save_models(results, project_root / "models", fingerprint)
    return [
        artifact_dir / "regression_metrics.json",
        artifact_dir / "classification_metrics.json",
        artifact_dir / "clustering_report.json",
        artifact_dir / "test_predictions.csv",
        artifact_dir / "plots" / "residuals.png",
        artifact_dir / "plots" / "confusion_matrix.png",
        artifact_dir / "plots" / "clusters.png",
        project_root / "models" / "regression_pipeline.joblib",
        project_root / "models" / "classification_pipeline.joblib",
        project_root / "models" / "clustering_pipeline.joblib",
        project_root / "models" / "manifest.json",
    ]
