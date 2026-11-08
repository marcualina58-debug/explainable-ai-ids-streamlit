"""Read-only support functions for the Streamlit demonstration artefact."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src import TARGET_CLASSES
from src.config import PROJECT_ROOT, load_config

RF_INTERNAL_ORDER = ("BENIGN", "Brute Force", "DDoS", "Port Scan")
RF_CANONICAL_MAPPING = (0, 2, 1, 3)
MODEL_FEATURE_COUNT = 65
DEPLOYMENT_MODES = ("LOCAL", "CLOUD_DEMO")
RF_NATIVE_TOP10_OVERLAP = 6
FINAL_COMPARISON_CLAIM = (
    "These results apply only to CICIDS2017, the frozen train/test split, and the final model "
    "configurations used in this experiment."
)
MODEL_DIRECTORIES = {
    "Random Forest": PROJECT_ROOT / "results/explainability/random_forest/local",
    "MLP": PROJECT_ROOT / "results/explainability/mlp/local",
}


class DemoDataError(ValueError):
    """Raised when a read-only demonstration input or artifact is invalid."""


@dataclass(frozen=True)
class PredictionResult:
    model: str
    predicted_class: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class CuratedCase:
    model: str
    case: str
    row_index: int
    true_class: str
    predicted_class: str
    probabilities: dict[str, float]
    contributions: pd.DataFrame
    waterfall_path: Path | None

    @property
    def correct(self) -> bool:
        return self.true_class == self.predicted_class


def read_config() -> dict[str, Any]:
    return load_config()


def resolve_deployment_mode(config: dict[str, Any], environ: dict[str, str] | None = None) -> str:
    """Resolve LOCAL/CLOUD_DEMO without changing the persisted configuration."""
    environment = os.environ if environ is None else environ
    configured = config.get("deployment", {}).get("mode", "LOCAL")
    mode = str(environment.get("STREAMLIT_DEPLOYMENT_MODE", configured)).strip().upper()
    if mode not in DEPLOYMENT_MODES:
        raise DemoDataError(f"Unsupported deployment mode: {mode}")
    return mode


def resolve_demo_dataset(
    config: dict[str, Any],
    mode: str,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Return the full local dataset or isolated cloud demonstration subset."""
    root = Path(project_root).resolve()
    if mode == "LOCAL":
        return root / config["data"]["processed_dir"] / config["data"]["processed_filename"]
    if mode != "CLOUD_DEMO":
        raise DemoDataError(f"Unsupported deployment mode: {mode}")
    relative = Path(config["deployment"]["cloud_demo"]["sample_path"])
    if relative.is_absolute():
        raise DemoDataError("Cloud demonstration sample path must be project-relative")
    resolved = (root / relative).resolve()
    assets_root = (root / "deployment_assets").resolve()
    if not resolved.is_relative_to(assets_root):
        raise DemoDataError("Cloud demonstration sample must be inside deployment_assets")
    return resolved


def validate_cloud_demo_sample(
    path: str | Path,
    feature_names: Sequence[str],
    target_column: str = "target",
) -> pd.DataFrame:
    """Validate the small deployment subset without consulting the full dataset."""
    sample = pd.read_csv(_require_file(path, "Cloud demonstration sample"))
    expected = [*feature_names, target_column]
    if sample.columns.tolist() != expected:
        raise DemoDataError("Cloud demonstration sample schema or feature order is invalid")
    if set(sample[target_column].astype(str)) != set(TARGET_CLASSES):
        raise DemoDataError("Cloud demonstration sample must contain all four canonical classes")
    numeric = sample.loc[:, list(feature_names)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DemoDataError("Cloud demonstration sample contains NaN or infinity")
    return sample


def prepare_probability_chart_data(probabilities: dict[str, float]) -> pd.DataFrame:
    """Pair probabilities with classes in the canonical presentation order."""
    if set(probabilities) != set(TARGET_CLASSES):
        raise DemoDataError("Probability mapping must contain exactly the four canonical classes")
    return pd.DataFrame({
        "Class": list(TARGET_CLASSES),
        "Probability": [float(probabilities[label]) for label in TARGET_CLASSES],
    }).set_index("Class")


def load_global_explainability_summary(
    rf_metadata_path: str | Path | None = None,
    comparison_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load already-computed global explainability statistics from frozen evidence."""
    rf_path = _require_file(
        rf_metadata_path or PROJECT_ROOT / "results/explainability/random_forest/rf_shap_metadata.json",
        "RF SHAP metadata",
    )
    comparison_path = _require_file(
        comparison_summary_path
        or PROJECT_ROOT / "results/explainability/comparison/explainability_comparison_summary.txt",
        "explainability comparison summary",
    )
    rf_metadata = json.loads(rf_path.read_text(encoding="utf-8"))
    rf_stats = rf_metadata["native_vs_shap_spearman"]
    comparison_text = comparison_path.read_text(encoding="utf-8")

    def extract(pattern: str, cast: type) -> Any:
        match = re.search(pattern, comparison_text)
        if not match:
            raise DemoDataError("Frozen explainability comparison summary is missing a required statistic")
        return cast(match.group(1))

    shared_match = re.search(r"Top-10 overlap: \[(.+)\]", comparison_text)
    if not shared_match:
        raise DemoDataError("Frozen explainability comparison summary is missing shared top-10 features")
    shared_features = re.findall(r"'([^']+)'", shared_match.group(1))
    return {
        "rf_rho": float(rf_stats["spearman_rho"]),
        "rf_p_value": float(rf_stats["p_value"]),
        "rf_top10_overlap": RF_NATIVE_TOP10_OVERLAP,
        "cross_model_rho": extract(r"Global feature-rank Spearman rho: ([^\s]+)", float),
        "cross_model_p_value": extract(r"Global feature-rank p-value: ([^\s]+)", float),
        "cross_model_top10_overlap": extract(r"Top-10 overlap count: (\d+)", int),
        "shared_top10_features": shared_features,
    }


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_rf_bundle(path: str | Path | None = None) -> dict[str, Any]:
    bundle = joblib.load(_require_file(path or PROJECT_ROOT / "models/random_forest.pkl", "RF model"))
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
        raise DemoDataError("RF model bundle is missing required fields")
    features = list(bundle["feature_names"])
    if len(features) != MODEL_FEATURE_COUNT or len(set(features)) != MODEL_FEATURE_COUNT:
        raise DemoDataError("RF bundle must contain 65 unique feature names")
    classes = tuple(str(value) for value in bundle["model"].classes_)
    if classes != RF_INTERNAL_ORDER:
        raise DemoDataError(f"Unexpected RF internal class order: {classes}")
    return bundle


def load_scaler_bundle(path: str | Path | None = None) -> dict[str, Any]:
    bundle = joblib.load(_require_file(path or PROJECT_ROOT / "models/mlp_scaler.joblib", "MLP scaler"))
    if not isinstance(bundle, dict) or "scaler" not in bundle or "feature_names" not in bundle:
        raise DemoDataError("Scaler bundle is missing required fields")
    features = list(bundle["feature_names"])
    if len(features) != MODEL_FEATURE_COUNT or len(set(features)) != MODEL_FEATURE_COUNT:
        raise DemoDataError("Scaler bundle must contain 65 unique feature names")
    return bundle


def load_mlp_model(path: str | Path | None = None) -> Any:
    model_path = _require_file(path or PROJECT_ROOT / "models/mlp_model.keras", "MLP model")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    if int(model.input_shape[-1]) != MODEL_FEATURE_COUNT or int(model.output_shape[-1]) != 4:
        raise DemoDataError(f"Unexpected MLP dimensions: {model.input_shape} -> {model.output_shape}")
    return model


def validate_feature_agreement(rf_features: Sequence[str], scaler_features: Sequence[str]) -> list[str]:
    rf = list(rf_features)
    scaler = list(scaler_features)
    if rf != scaler or len(rf) != MODEL_FEATURE_COUNT or len(set(rf)) != MODEL_FEATURE_COUNT:
        raise DemoDataError("RF and scaler feature order must match exactly across 65 unique features")
    return rf


def validate_probability_vector(values: Any) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if probabilities.shape != (4,):
        raise DemoDataError(f"Expected four probabilities, found shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise DemoDataError("Probabilities contain NaN or infinity")
    if ((probabilities < -1e-8) | (probabilities > 1 + 1e-8)).any():
        raise DemoDataError("Probability lies outside [0, 1]")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise DemoDataError(f"Probabilities sum to {probabilities.sum()}, not one")
    return probabilities


def remap_rf_probabilities(internal: Any) -> np.ndarray:
    probabilities = validate_probability_vector(internal)
    return validate_probability_vector(probabilities[list(RF_CANONICAL_MAPPING)])


def prepare_feature_row(values: pd.Series | pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    frame = values.to_frame().T if isinstance(values, pd.Series) else values.copy()
    expected = list(feature_names)
    if len(expected) != MODEL_FEATURE_COUNT or len(set(expected)) != MODEL_FEATURE_COUNT:
        raise DemoDataError("Expected feature specification must contain 65 unique names")
    if frame.shape[0] != 1:
        raise DemoDataError("Exactly one record is required")
    if frame.columns.duplicated().any():
        raise DemoDataError("Record contains duplicated feature columns")
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise DemoDataError(f"Record is missing expected features: {missing}")
    ordered = frame.loc[:, expected].apply(pd.to_numeric, errors="raise")
    if ordered.shape != (1, MODEL_FEATURE_COUNT):
        raise DemoDataError(f"Expected one row with 65 features, found {ordered.shape}")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise DemoDataError("Record contains NaN or infinity")
    return ordered


def predict_random_forest(model: Any, raw_row: pd.DataFrame, feature_names: Sequence[str]) -> PredictionResult:
    ordered = prepare_feature_row(raw_row, feature_names)
    if tuple(str(value) for value in model.classes_) != RF_INTERNAL_ORDER:
        raise DemoDataError("RF internal class order does not match the verified order")
    internal = np.asarray(model.predict_proba(ordered), dtype=float)
    if internal.shape != (1, 4):
        raise DemoDataError(f"Unexpected RF probability shape: {internal.shape}")
    canonical = remap_rf_probabilities(internal[0])
    mapping = {label: float(canonical[index]) for index, label in enumerate(TARGET_CLASSES)}
    return PredictionResult("Random Forest", TARGET_CLASSES[int(np.argmax(canonical))], mapping)


def predict_mlp(model: Any, scaler: Any, raw_row: pd.DataFrame, feature_names: Sequence[str]) -> PredictionResult:
    ordered = prepare_feature_row(raw_row, feature_names)
    scaled = np.asarray(scaler.transform(ordered), dtype=np.float32)
    if scaled.shape != (1, MODEL_FEATURE_COUNT) or not np.isfinite(scaled).all():
        raise DemoDataError("Saved scaler returned invalid MLP input")
    probabilities = np.asarray(model.predict(scaled, verbose=0), dtype=float)
    if probabilities.shape != (1, 4):
        raise DemoDataError(f"Unexpected MLP probability shape: {probabilities.shape}")
    canonical = validate_probability_vector(probabilities[0])
    mapping = {label: float(canonical[index]) for index, label in enumerate(TARGET_CLASSES)}
    return PredictionResult("MLP", TARGET_CLASSES[int(np.argmax(canonical))], mapping)


def compare_predictions(
    rf_model: Any,
    mlp_model: Any,
    scaler: Any,
    raw_row: pd.DataFrame,
    feature_names: Sequence[str],
) -> dict[str, PredictionResult]:
    return {
        "Random Forest": predict_random_forest(rf_model, raw_row, feature_names),
        "MLP": predict_mlp(mlp_model, scaler, raw_row, feature_names),
    }


def discover_curated_cases() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, directory in MODEL_DIRECTORIES.items():
        selected_path = _require_file(directory / "selected_records.csv", f"{model} curated index")
        frame = pd.read_csv(selected_path)
        required = {"case", "true_class", "predicted_class", "row_index", "available"}
        if not required.issubset(frame.columns):
            raise DemoDataError(f"{selected_path} has an invalid schema")
        frame.insert(0, "model", model)
        frames.append(frame)
    inventory = pd.concat(frames, ignore_index=True)
    if len(inventory) != 11:
        raise DemoDataError(f"Expected 11 curated model/case combinations, found {len(inventory)}")
    return inventory


def _validated_final_local_directory(model: str) -> Path:
    if model not in MODEL_DIRECTORIES:
        raise DemoDataError(f"Unsupported model: {model}")
    directory = MODEL_DIRECTORIES[model].resolve()
    if "smoke_test" in directory.parts or not directory.is_relative_to(
        (PROJECT_ROOT / "results/explainability").resolve()
    ):
        raise DemoDataError("Curated artifact path is not in the final explainability tree")
    return directory


def load_curated_case(model: str, case: str, expected_features: Sequence[str]) -> CuratedCase:
    directory = _validated_final_local_directory(model)
    inventory = discover_curated_cases()
    selected = inventory[(inventory.model == model) & (inventory.case == case)]
    if len(selected) != 1:
        raise DemoDataError(f"Curated case not found: {model} / {case}")
    record = selected.iloc[0]
    if not bool(record.available):
        raise DemoDataError(f"Curated case is recorded as unavailable: {model} / {case}")
    json_path = _require_file(directory / f"{case}.json", "Curated JSON")
    csv_path = _require_file(directory / f"{case}_contributions.csv", "Curated contributions")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    contributions = pd.read_csv(csv_path)
    if payload.get("case") != case or int(payload.get("row_index")) != int(record.row_index):
        raise DemoDataError("Curated JSON does not match its selected-record entry")
    true_class, predicted_class = payload.get("true_class"), payload.get("predicted_class")
    if true_class not in TARGET_CLASSES or predicted_class not in TARGET_CLASSES:
        raise DemoDataError("Curated JSON contains an unexpected class")
    raw_probabilities = payload.get("class_probabilities", {})
    if set(raw_probabilities) != set(TARGET_CLASSES):
        raise DemoDataError("Curated JSON must contain exactly four canonical probabilities")
    vector = validate_probability_vector([raw_probabilities[label] for label in TARGET_CLASSES])
    if "feature" not in contributions or contributions.feature.duplicated().any():
        raise DemoDataError("Curated contribution table has invalid feature labels")
    expected = list(expected_features)
    if len(contributions) != MODEL_FEATURE_COUNT or set(contributions.feature) != set(expected):
        raise DemoDataError("Curated contribution table does not contain the expected 65 features")
    waterfall = directory / f"{case}_waterfall.png"
    return CuratedCase(
        model=model,
        case=case,
        row_index=int(record.row_index),
        true_class=true_class,
        predicted_class=predicted_class,
        probabilities={label: float(vector[index]) for index, label in enumerate(TARGET_CLASSES)},
        contributions=contributions.set_index("feature").loc[expected].reset_index(),
        waterfall_path=waterfall if waterfall.is_file() else None,
    )


def lookup_dataset_record(
    dataset_path: str | Path,
    positional_index: int,
    feature_names: Sequence[str],
    target_column: str = "target",
    chunk_size: int = 50_000,
) -> tuple[str, pd.DataFrame]:
    if isinstance(positional_index, bool) or not isinstance(positional_index, (int, np.integer)):
        raise DemoDataError("Row index must be an integer")
    if positional_index < 0:
        raise DemoDataError("Row index must be non-negative")
    dataset = _require_file(dataset_path, "Processed dataset")
    offset = 0
    for chunk in pd.read_csv(dataset, chunksize=chunk_size):
        if chunk.columns.duplicated().any():
            raise DemoDataError("Processed dataset contains duplicated columns")
        if positional_index < offset + len(chunk):
            row = chunk.iloc[[positional_index - offset]]
            if target_column not in row.columns:
                raise DemoDataError(f"Target column not found: {target_column}")
            true_class = str(row.iloc[0][target_column])
            if true_class not in TARGET_CLASSES:
                raise DemoDataError(f"Unexpected true class: {true_class}")
            return true_class, prepare_feature_row(row.drop(columns=[target_column]), feature_names)
        offset += len(chunk)
    raise IndexError(f"Row index {positional_index} is outside the processed dataset")


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    artifact = _require_file(path, "Frozen JSON artifact")
    return json.loads(artifact.read_text(encoding="utf-8"))


def load_external_validation_results(
    metrics_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load and validate the frozen external RF evaluation outputs."""
    directory = PROJECT_ROOT / "results/external_validation/cse_cic_ids2018"
    metrics = load_json_artifact(metrics_path or directory / "external_rf_metrics.json")
    matrix_file = _require_file(
        matrix_path or directory / "confusion_matrix_3x4.csv",
        "External validation confusion matrix",
    )
    matrix = pd.read_csv(matrix_file, index_col=0)
    expected_true = ["BENIGN", "DDoS", "Brute Force"]
    expected_predicted = ["BENIGN", "DDoS", "Brute Force", "Port Scan"]
    required_metrics = {
        "evaluated_records", "accuracy", "macro_f1_supported_three_classes",
        "per_class", "incorrect_port_scan_predictions", "true_classes",
        "possible_prediction_classes", "port_scan_ground_truth_available",
    }
    missing = sorted(required_metrics - set(metrics))
    if missing:
        raise DemoDataError(f"External metrics are missing required fields: {missing}")
    if metrics["true_classes"] != expected_true:
        raise DemoDataError("External metrics have an unexpected true-class order")
    if metrics["possible_prediction_classes"] != expected_predicted:
        raise DemoDataError("External metrics have an unexpected prediction-class order")
    if bool(metrics["port_scan_ground_truth_available"]):
        raise DemoDataError("External evidence unexpectedly claims Port Scan ground truth")
    if matrix.index.tolist() != expected_true or matrix.columns.tolist() != expected_predicted:
        raise DemoDataError("External confusion matrix must be 3 true classes by 4 predictions")
    numeric_matrix = matrix.apply(pd.to_numeric, errors="raise")
    if (numeric_matrix.to_numpy() < 0).any():
        raise DemoDataError("External confusion matrix contains a negative count")
    if int(numeric_matrix.to_numpy().sum()) != int(metrics["evaluated_records"]):
        raise DemoDataError("External confusion matrix does not match the evaluated record count")
    if set(metrics["per_class"]) != set(expected_true):
        raise DemoDataError("External per-class metrics must contain only supported true classes")
    return metrics, numeric_matrix
