import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


TARGET = "MedHouseVal"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CaliforniaHousingModel")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

OUT_DIR = Path("reports/shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def load_production_model():
    # If you use a custom tracking uri, set MLFLOW_TRACKING_URI in env
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri


def unwrap_if_pipeline(model):
    """
    SHAP needs the actual estimator + the transformed X.
    - If model is a Pipeline with a preprocessor (scaler), we keep the pipeline for predict,
      but for SHAP we prefer explaining the final estimator on preprocessed data when possible.
    This function returns:
      (predict_fn, estimator_for_shap, transform_fn_or_None)
    """
    # sklearn Pipeline has attribute named_steps
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
        # Try to find a preprocessing step (often called 'scaler' or 'prep')
        # We'll build a transform function that applies all steps except the final estimator.
        def transform_fn(X: pd.DataFrame):
            Xt = X
            for name, step in model.named_steps.items():
                if name == "model":
                    break
                Xt = step.transform(Xt)
            return Xt

        def predict_fn(X: pd.DataFrame):
            return model.predict(X)

        return predict_fn, estimator, transform_fn

    # Not a pipeline
    def predict_fn(X: pd.DataFrame):
        return model.predict(X)

    return predict_fn, model, None


def pick_background_and_examples(X_train, X_test, background_size=200, explain_size=500, seed=42):
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=min(background_size, len(X_train)), replace=False)
    ex_idx = rng.choice(len(X_test), size=min(explain_size, len(X_test)), replace=False)

    X_bg = X_train.iloc[bg_idx].copy()
    X_explain = X_test.iloc[ex_idx].copy()
    return X_bg, X_explain


def build_explainer(estimator, X_background, feature_names):
    """
    Choose SHAP explainer based on model type.
    - Tree models -> TreeExplainer
    - Linear models -> LinearExplainer
    - Fallback -> KernelExplainer (slower)
    """
    est_name = estimator.__class__.__name__.lower()

    # Tree-based
    if "forest" in est_name or "gb" in est_name or "xgb" in est_name or "lgbm" in est_name or "tree" in est_name:
        return shap.TreeExplainer(estimator)

    # Linear
    if "linear" in est_name or "ridge" in est_name or "lasso" in est_name or "elasticnet" in est_name:
        # For linear models, provide background for expected value stability
        return shap.LinearExplainer(estimator, X_background, feature_perturbation="interventional")

    # Fallback (slow)
    # KernelExplainer requires a predict function; we’ll build it later if needed.
    return None


def save_global_plots(shap_values, X_explain, prefix="global"):
    # Bar plot: mean(|SHAP|)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_bar.png", dpi=180)
    plt.close()

    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_beeswarm.png", dpi=180)
    plt.close()


def save_local_plots(shap_values, idx=0, prefix="local"):
    # Waterfall plot for one example
    plt.figure()
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_waterfall_idx{idx}.png", dpi=180)
    plt.close()

    # Force plot as HTML (works great for reports)
    force = shap.plots.force(shap_values[idx], matplotlib=False)
    shap.save_html(str(OUT_DIR / f"{prefix}_force_idx{idx}.html"), force)


def main():
    X_train, X_test, y_train, y_test = load_data()
    model, model_uri = load_production_model()

    predict_fn, estimator_for_shap, transform_fn = unwrap_if_pipeline(model)

    X_bg_raw, X_explain_raw = pick_background_and_examples(X_train, X_test)

    # If pipeline: explain on transformed features (scaler output), feature names become generic
    # For this project, your Production model is most likely RandomForest (no scaler), so names stay clean.
    if transform_fn is not None:
        X_bg = pd.DataFrame(transform_fn(X_bg_raw))
        X_explain = pd.DataFrame(transform_fn(X_explain_raw))
        feature_names = [f"f{i}" for i in range(X_bg.shape[1])]
        X_bg.columns = feature_names
        X_explain.columns = feature_names
    else:
        X_bg = X_bg_raw
        X_explain = X_explain_raw
        feature_names = list(X_bg.columns)

    explainer = build_explainer(estimator_for_shap, X_bg, feature_names)

    if explainer is None:
        # Fallback KernelExplainer (slow) — we limit to a very small sample
        X_bg_small = X_bg.iloc[:50].copy()
        X_explain_small = X_explain.iloc[:50].copy()

        def pred_np(X_np):
            X_df = pd.DataFrame(X_np, columns=feature_names)
            # if pipeline, we would need raw features; but here we already transformed, so estimator predict:
            return estimator_for_shap.predict(X_df)

        explainer = shap.KernelExplainer(pred_np, X_bg_small)
        shap_vals = explainer.shap_values(X_explain_small, nsamples=200)
        shap_values = shap.Explanation(
            values=shap_vals,
            base_values=explainer.expected_value,
            data=X_explain_small.values,
            feature_names=feature_names,
        )
        X_explain_used = X_explain_small
    else:
        # SHAP Explanation object (new API)
        shap_values = explainer(X_explain)
        X_explain_used = X_explain

    # Save plots
    save_global_plots(shap_values, X_explain_used, prefix="global")
    save_local_plots(shap_values, idx=0, prefix="local")

    # Save a small text summary for your report
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:8]
    top_features = [(feature_names[i], float(mean_abs[i])) for i in top_idx]

    summary_path = OUT_DIR / "summary_top_features.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model URI: {model_uri}\n")
        f.write("Top features by mean(|SHAP|):\n")
        for name, score in top_features:
            f.write(f"- {name}: {score:.6f}\n")

    print("SHAP report generated in:", OUT_DIR.resolve())
    print("Top features:", top_features)


if __name__ == "__main__":
    main()
