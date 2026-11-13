"""Offline Streamlit demonstration for the frozen intrusion-detection models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit launches this file from ``app/`` on Community Cloud. Add the
# repository root so sibling packages such as ``src`` resolve consistently.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.demo_support import (
    DemoDataError, FINAL_COMPARISON_CLAIM, PredictionResult, discover_curated_cases, load_curated_case,
    load_external_validation_results,
    load_global_explainability_summary,
    load_json_artifact, load_mlp_model, load_rf_bundle, load_scaler_bundle,
    lookup_dataset_record, predict_mlp, predict_random_forest, prepare_probability_chart_data, read_config,
    resolve_demo_dataset, resolve_deployment_mode, validate_cloud_demo_sample,
    validate_feature_agreement,
)
from src import TARGET_CLASSES
from src.config import PROJECT_ROOT

st.set_page_config(page_title="Explainable AI for Cloud Networks", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner=False)
def cached_config():
    return read_config()


@st.cache_resource(show_spinner="Loading final Random Forest…")
def cached_rf_bundle():
    return load_rf_bundle()


@st.cache_resource(show_spinner="Loading final MLP…")
def cached_mlp_model():
    return load_mlp_model()


@st.cache_resource(show_spinner="Loading saved StandardScaler…")
def cached_scaler_bundle():
    return load_scaler_bundle()


@st.cache_data(show_spinner=False)
def cached_curated_inventory():
    return discover_curated_cases()


@st.cache_data(show_spinner=False)
def cached_curated_case(model: str, case: str, features: tuple[str, ...]):
    return load_curated_case(model, case, features)


@st.cache_data(show_spinner="Locating the selected record…")
def cached_record(path: str, index: int, features: tuple[str, ...], target: str):
    return lookup_dataset_record(path, index, features, target)


@st.cache_data(show_spinner=False)
def cached_json(path: str):
    return load_json_artifact(path)


@st.cache_data(show_spinner=False)
def cached_global_explainability_summary(deployment_mode: str):
    metadata = None
    if deployment_mode == "CLOUD_DEMO":
        metadata = PROJECT_ROOT / "deployment_assets/rf_explainability_summary.json"
    return load_global_explainability_summary(rf_metadata_path=metadata)


@st.cache_data(show_spinner=False)
def cached_external_validation_results():
    return load_external_validation_results()


def show_prediction(result: PredictionResult, true_class: str, row_index: int) -> None:
    correct = result.predicted_class == true_class
    columns = st.columns(5)
    for column, label, value in zip(columns, ("Model", "Row index", "True class", "Predicted class", "Status"), (
        result.model, f"{row_index:,}", true_class, result.predicted_class,
        "Correct" if correct else "Incorrect",
    )):
        column.metric(label, value)
    if not correct:
        st.error("The saved model prediction is incorrect for this record.")
    probabilities = prepare_probability_chart_data(result.probabilities)
    st.bar_chart(probabilities, horizontal=True, sort=False)
    st.dataframe(probabilities.style.format("{:.6f}"), width="stretch")


st.title("Explainable AI for Multi-Attack Detection in Cloud Networks")
st.caption("Academic Proof-of-Concept")
st.info(
    "This offline demonstration uses CICIDS2017 and does not capture live network traffic. "
    "It is not a production IDS, and predictions are not operational security alerts. "
    "SHAP explains model behaviour and feature contributions; it does not establish causality."
)

try:
    config = cached_config()
    rf_bundle = cached_rf_bundle()
    scaler_bundle = cached_scaler_bundle()
    features = validate_feature_agreement(rf_bundle["feature_names"], scaler_bundle["feature_names"])
    deployment_mode = resolve_deployment_mode(config)
    demo_dataset = resolve_demo_dataset(config, deployment_mode)
    cloud_sample_rows = None
    if deployment_mode == "CLOUD_DEMO":
        cloud_sample_rows = len(
            validate_cloud_demo_sample(demo_dataset, features, config["data"]["target_column"])
        )
except (FileNotFoundError, DemoDataError, OSError) as error:
    st.error(f"The demonstration cannot start: {error}")
    with st.expander("Technical details"):
        st.code(repr(error))
    st.stop()

if deployment_mode == "CLOUD_DEMO":
    st.info(
        "The online demonstration uses a small representative subset of CICIDS2017. "
        "The original evaluation and external validation results are unchanged."
    )

with st.sidebar:
    st.header("Demonstration controls")
    mode = st.radio("Mode", ("Curated Example", "Custom Dataset Record"))
    model_choice = st.selectbox("Model", ("Random Forest", "MLP", "Compare Both"))
    selected_case = None
    row_index = None
    if mode == "Curated Example":
        try:
            inventory = cached_curated_inventory()
            available = inventory[inventory.available.astype(bool)]
            if model_choice != "Compare Both":
                available = available[available.model == model_choice]
                labels = {
                    f"{row.true_class} → {row.predicted_class} (row {int(row.row_index):,})": row.case
                    for row in available.itertuples()
                }
            else:
                shared = ("correct_BENIGN", "correct_DDoS", "correct_Brute Force", "correct_Port Scan")
                labels = {case.replace("correct_", "Both models: "): case for case in shared}
            selected_case = labels[st.selectbox("Curated example", tuple(labels))]
        except (FileNotFoundError, DemoDataError, OSError) as error:
            st.error(str(error))
    else:
        if deployment_mode == "CLOUD_DEMO":
            st.caption(
                "Enter a zero-based row index in the class-balanced demonstration subset. "
                "This subset is not an independent evaluation dataset."
            )
            row_index = int(st.number_input(
                "Demonstration subset row index", min_value=0, max_value=cloud_sample_rows - 1,
                step=1, value=0,
            ))
        else:
            st.caption("Enter a zero-based positional row index in the processed CICIDS2017 CSV.")
            row_index = int(st.number_input("Positional row index", min_value=0, step=1, value=0))

tabs = st.tabs((
    "Prediction", "Explanation", "Feature Values", "Final Model Comparison",
    "Global Explainability", "External Validation", "Limitations",
))
cases, predictions, true_class, raw_row = [], {}, None, None

try:
    if mode == "Curated Example" and selected_case:
        models = ("Random Forest", "MLP") if model_choice == "Compare Both" else (model_choice,)
        cases = [cached_curated_case(model, selected_case, tuple(features)) for model in models]
    elif mode == "Custom Dataset Record" and row_index is not None:
        true_class, raw_row = cached_record(
            str(demo_dataset), row_index, tuple(features), config["data"]["target_column"]
        )
        if model_choice in ("Random Forest", "Compare Both"):
            predictions["Random Forest"] = predict_random_forest(rf_bundle["model"], raw_row, features)
        if model_choice in ("MLP", "Compare Both"):
            predictions["MLP"] = predict_mlp(cached_mlp_model(), scaler_bundle["scaler"], raw_row, features)
except (FileNotFoundError, DemoDataError, IndexError, OSError, ValueError) as error:
    st.error(f"Unable to prepare the selected demonstration: {error}")
    with st.expander("Technical details"):
        st.code(repr(error))

with tabs[0]:
    st.subheader("Prediction")
    if cases:
        for case in cases:
            show_prediction(PredictionResult(case.model, case.predicted_class, case.probabilities), case.true_class, case.row_index)
    elif predictions:
        for result in predictions.values():
            show_prediction(result, true_class, int(row_index))
    else:
        st.warning("No valid record is currently selected.")

with tabs[1]:
    st.subheader("Explanation")
    st.warning("SHAP explains model behaviour and feature contributions; it does not establish causality.")
    if cases:
        for case in cases:
            st.markdown(f"### {case.model}: {case.true_class} → {case.predicted_class}")
            if case.waterfall_path:
                st.image(str(case.waterfall_path), caption="Precomputed final SHAP waterfall")
            else:
                st.warning("The optional frozen waterfall image is unavailable.")
            positive = case.contributions.nlargest(10, "shap_predicted_class")
            opposing = case.contributions.nsmallest(10, "shap_predicted_class")
            left, right = st.columns(2)
            left.markdown("**Features contributing toward the model prediction**")
            left.dataframe(positive[["feature", "original_value", "shap_predicted_class"]], hide_index=True)
            right.markdown("**Features opposing the model prediction**")
            right.dataframe(opposing[["feature", "original_value", "shap_predicted_class"]], hide_index=True)
            if not case.correct:
                st.error("Misclassified curated example")
                columns = ["feature", "shap_predicted_class", "shap_true_class", "predicted_minus_true_shap"]
                st.dataframe(case.contributions.nlargest(15, "predicted_minus_true_shap")[columns], hide_index=True)
                st.caption(
                    f"True-class probability: {case.probabilities[case.true_class]:.6f}; "
                    f"predicted-class probability: {case.probabilities[case.predicted_class]:.6f}."
                )
    elif predictions:
        st.info(
            "Live prediction generated from the saved final model. A precomputed final SHAP explanation "
            "is available only for curated examples. This keeps demonstration-time predictions separate "
            "from the frozen experimental explainability results."
        )

with tabs[2]:
    st.subheader("Feature Values")
    if cases:
        feature_frame = cases[0].contributions[["feature", "original_value"]].set_index("feature")
    elif raw_row is not None:
        feature_frame = raw_row.T.rename(columns={raw_row.index[0]: "original_value"})
        feature_frame.index.name = "feature"
    else:
        feature_frame = None
    if feature_frame is not None:
        useful = ("Destination Port", "PSH Flag Count", "ACK Flag Count", "Init_Win_bytes_forward", "Min Packet Length", "Fwd Packet Length Max", "Flow Duration")
        st.dataframe(feature_frame.loc[[name for name in useful if name in feature_frame.index]])
        with st.expander("All 65 raw model features"):
            st.dataframe(feature_frame, width="stretch")
        if model_choice in ("MLP", "Compare Both"):
            st.caption("The saved StandardScaler is applied before MLP inference.")

with tabs[3]:
    st.subheader("Final Model Comparison")
    try:
        rf_metrics = cached_json(str(PROJECT_ROOT / "results/metrics/random_forest_metrics.json"))
        mlp_metrics = cached_json(str(PROJECT_ROOT / "results/metrics/mlp_metrics.json"))
        names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
        comparison = pd.DataFrame({"Metric": names, "Random Forest": [rf_metrics[n] for n in names], "MLP": [mlp_metrics[n] for n in names]}).set_index("Metric")
        st.dataframe(comparison.style.format("{:.6f}"), width="stretch")
        per_class = pd.DataFrame({"Class": TARGET_CLASSES, "Random Forest F1": [rf_metrics["per_class"][c]["f1"] for c in TARGET_CLASSES], "MLP F1": [mlp_metrics["per_class"][c]["f1"] for c in TARGET_CLASSES]}).set_index("Class")
        st.dataframe(per_class.style.format("{:.6f}"), width="stretch")
        st.image(str(PROJECT_ROOT / "results/comparison/rf_vs_mlp_macro_metrics.png"))
        st.image(str(PROJECT_ROOT / "results/comparison/rf_vs_mlp_per_class_f1.png"))
        st.markdown(
            "For this frozen CICIDS2017 experiment, RF achieved stronger macro F1, while MLP achieved "
            "slightly higher macro recall. MLP's Brute Force recall was very high, but its precision was "
            "substantially lower because many BENIGN records were classified as Brute Force. This does "
            f"not establish universal model superiority. {FINAL_COMPARISON_CLAIM}"
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        st.error(f"Frozen comparison evidence is unavailable: {error}")

with tabs[4]:
    st.subheader("Global Explainability")
    global_summary = cached_global_explainability_summary(deployment_mode)
    left, right = st.columns(2)
    left.markdown("### Random Forest")
    left.image(str(PROJECT_ROOT / "results/explainability/random_forest/rf_global_importance.png"))
    left.image(str(PROJECT_ROOT / "results/explainability/random_forest/rf_native_vs_shap.png"))
    right.markdown("### MLP")
    right.image(str(PROJECT_ROOT / "results/explainability/mlp/mlp_global_importance.png"))
    st.markdown("### Cross-model rank and normalized-importance comparison")
    st.image(str(PROJECT_ROOT / "results/explainability/comparison/rf_vs_mlp_shap_importance.png"))
    st.markdown(
        f"**RF native importance vs RF global SHAP:** Spearman rho ≈ {global_summary['rf_rho']:.6f}; "
        f"p-value ≈ {global_summary['rf_p_value']:.2e}; top-10 overlap = {global_summary['rf_top10_overlap']}.\n\n"
        f"**RF global SHAP vs MLP global SHAP:** Spearman rho ≈ {global_summary['cross_model_rho']:.6f}; "
        f"p-value ≈ {global_summary['cross_model_p_value']:.6f}; "
        f"top-10 overlap = {global_summary['cross_model_top10_overlap']}.\n\n"
        f"**Shared global top-10 features:** {', '.join(global_summary['shared_top10_features'])}.\n\n"
        "Destination Port ranked 1st for RF native importance and RF global SHAP, but 18th for MLP global "
        "SHAP. This cross-model comparison uses feature ranks and normalized importance; raw RF and MLP "
        "SHAP magnitudes are not treated as equivalent."
    )

with tabs[5]:
    st.subheader("External Validation")
    st.caption("Frozen Random Forest evaluation on CSE-CIC-IDS2018; no external CSV is loaded here.")
    try:
        external_metrics, external_matrix = cached_external_validation_results()
        original_rf = cached_json(str(PROJECT_ROOT / "results/metrics/random_forest_metrics.json"))
        comparison = pd.DataFrame([
            {
                "Evaluation dataset": "CICIDS2017",
                "Accuracy": original_rf["accuracy"],
                "Macro F1": original_rf["macro_f1"],
                "Ground-truth classes": len(original_rf["per_class"]),
            },
            {
                "Evaluation dataset": "CSE-CIC-IDS2018",
                "Accuracy": external_metrics["accuracy"],
                "Macro F1": external_metrics["macro_f1_supported_three_classes"],
                "Ground-truth classes": len(external_metrics["true_classes"]),
            },
        ]).set_index("Evaluation dataset")
        st.markdown("### Original and external results")
        st.dataframe(
            comparison.style.format({"Accuracy": "{:.6f}", "Macro F1": "{:.6f}"}),
            width="stretch",
        )
        st.warning(
            "The macro F1 values are not directly comparable. CICIDS2017 contains four true classes, "
            "whereas this external evaluation contains BENIGN, DDoS and Brute Force but no Port Scan "
            "ground truth. All four possible Random Forest prediction outputs were still retained."
        )

        st.markdown("### External class-level results")
        per_class = pd.DataFrame.from_dict(external_metrics["per_class"], orient="index")
        per_class.index.name = "True class"
        per_class = per_class.rename(columns={"f1": "F1", "support": "Support"})
        per_class = per_class.rename(columns={"precision": "Precision", "recall": "Recall"})
        st.dataframe(
            per_class[["Precision", "Recall", "F1", "Support"]].style.format({
                "Precision": "{:.6f}", "Recall": "{:.6f}", "F1": "{:.6f}", "Support": "{:,.0f}",
            }),
            width="stretch",
        )

        st.markdown("### External confusion matrix")
        st.dataframe(external_matrix.style.format("{:,.0f}"), width="stretch")
        matrix_image = PROJECT_ROOT / "results/external_validation/cse_cic_ids2018/confusion_matrix_3x4.png"
        if matrix_image.is_file():
            st.image(str(matrix_image), caption="Saved 3 × 4 external-validation confusion matrix")

        st.error(
            "Key finding: the Random Forest achieved high accuracy on CICIDS2017 but performed poorly "
            "on CSE-CIC-IDS2018. Most external attack records were predicted as BENIGN."
        )
        st.markdown("### Why did performance decrease?")
        st.markdown(
            "The audited datasets showed substantial differences in **Flow Duration**, **Flow Bytes/s**, "
            "**Flow Packets/s** and the initial **TCP window features**. The network environments, "
            "services, attack tools and traffic patterns also differed. These are credible contributing "
            "factors, but the evaluation does not prove that any single difference caused the decrease."
        )
        st.markdown("### Scientific limitations")
        st.markdown("""
- CSE-CIC-IDS2018 provides no Port Scan ground truth in the selected external files.
- The original and external evaluations cover different numbers of true classes.
- Both datasets use the CICFlowMeter methodology, so they are not fully independent.
- Similarly named features may still differ because of extraction settings or flow construction.
- Attack tools, network services and traffic distributions differ between datasets.
- This is offline cross-dataset validation, not real-time traffic monitoring or production validation.
""")
        st.success(
            "Conclusion: high performance on the original dataset does not guarantee strong "
            "generalisation to a different network environment."
        )
    except (FileNotFoundError, DemoDataError, KeyError, ValueError, OSError) as error:
        st.error(f"External validation evidence is unavailable: {error}")

with tabs[6]:
    st.subheader("Limitations and Claim Boundaries")
    st.markdown("""
- CICIDS2017 is an older benchmark and may not represent current cloud traffic.
- The dataset is substantially class imbalanced and uses a random row-level stratified split.
- Related capture-period traffic may occur across partitions.
- Destination Port may encode dataset-specific patterns; correlated features can redistribute SHAP attribution.
- MLP explanations operate on standardized inputs, and GradientExplainer is approximate.
- Balanced class-conditional samples are not population-representative; local explanations describe individual records only.
- SHAP is non-causal, and RF/MLP raw SHAP magnitudes are not automatically comparable.
- This is an offline academic proof-of-concept, not a production IDS.
""")
