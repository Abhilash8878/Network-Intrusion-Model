from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "final_ids_models"
# The loader searches the project folder and its subfolders for model files or ZIP archives.
MODEL_ZIP = ROOT / "final_ids_models.zip"
MAX_FILE_SIZE_MB = 50
MAX_DISPLAY_ROWS = 500

st.set_page_config(page_title="Network IDS | Security Analytics", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 14px; background: linear-gradient(120deg,#0b1d33,#173d64); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
.hero p {margin: 0; color: #d6e7f7;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); padding: .85rem; border-radius: 12px;}
</style>
<div class="hero"><h1>🛡️ Network Intrusion Detection System</h1>
<p>Machine-learning traffic analysis using Random Forest, Isolation Forest, and hybrid detection.</p></div>
""", unsafe_allow_html=True)

st.sidebar.title("Sentinel IDS")
st.sidebar.caption("Local traffic analysis prototype")
st.sidebar.markdown("---")
st.sidebar.write("**Pipeline**")
st.sidebar.write("1. Upload a compatible feature CSV")
st.sidebar.write("2. Validate the feature columns")
st.sidebar.write("3. Run the saved models")
st.sidebar.write("4. Review and export results")
st.sidebar.info("Uploaded CSV analysis only; no live packet capture.")


@st.cache_resource(show_spinner="Loading saved IDS models...")
def load_artifacts():
    required = [
        "random_forest_zero_day.pkl",
        "isolation_forest.pkl",
        "imputer.pkl",
        "scaler.pkl",
        "feature_names.pkl",
    ]

    # Load models directly from the final_ids_models folder
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_DIR}"
        )

    missing = [
        name for name in required
        if not (MODEL_DIR / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

    rf = joblib.load(MODEL_DIR / "random_forest_zero_day.pkl")
    isolation = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
        names = [str(x) for x in list(pickle.load(f))]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names found in feature_names.pkl")

    return rf, isolation, imputer, scaler, names

    


def process_csv(uploaded_file, feature_names, imputer, scaler,
                rf_model, isolation_model):
    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")

    # Remove target and identifier columns
    forbidden = {
        "label", "attack_cat", "isfraud", "is_fraud", "id"
    }
    drop_cols = [
        c for c in df.columns
        if str(c).strip().lower() in forbidden
    ]
    X_raw = df.drop(columns=drop_cols, errors="ignore")

    # Handle both raw CSV files and already-encoded CSV files
    if all(col in X_raw.columns for col in feature_names):
        X_df = X_raw.loc[:, feature_names].copy()
    else:
        # Convert categorical columns using the training-style encoding
        X_df = pd.get_dummies(X_raw, drop_first=True)

        # Match the exact training feature names and order
        X_df = X_df.reindex(
            columns=feature_names,
            fill_value=0
        )

    # Convert invalid values to missing values
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # Apply saved preprocessing
    X = imputer.transform(X_df)
    X = scaler.transform(X)

    # Predictions
    rf_pred = np.asarray(rf_model.predict(X)).astype(int)

    isolation_raw = np.asarray(isolation_model.predict(X))
    isolation_pred = np.where(isolation_raw == -1, 1, 0).astype(int)

    hybrid_pred = np.where(
        (rf_pred == 1) | (isolation_pred == 1),
        1,
        0
    ).astype(int)

    results = pd.DataFrame({
        "row_number": np.arange(1, len(df) + 1),
        "random_forest": np.where(
            rf_pred == 1, "Attack", "Normal"
        ),
        "isolation_forest": np.where(
            isolation_pred == 1, "Anomaly", "Normal"
        ),
        "hybrid_result": np.where(
            hybrid_pred == 1,
            "Suspicious / Attack",
            "Normal"
        ),
    })

    summary = {
        "total_records": len(df),
        "random_forest_attacks": int(np.sum(rf_pred == 1)),
        "isolation_forest_anomalies": int(
            np.sum(isolation_pred == 1)
        ),
        "hybrid_attacks": int(np.sum(hybrid_pred == 1)),
        "hybrid_normal": int(np.sum(hybrid_pred == 0))
    }

    extras = [
        c for c in df.columns
        if c not in feature_names
    ]

    return results, summary, extras


try:
    rf_model, isolation_model, imputer, scaler, feature_names = load_artifacts()
    models_ready = True
    st.sidebar.success("Saved models loaded")
    st.sidebar.caption(f"Expected features: {len(feature_names)}")
except Exception as exc:
    models_ready = False
    st.sidebar.error("Models not ready")
    st.error(f"**Model setup required:** {exc}")
    st.info(
        "The app searches this project folder and its subfolders for the five model files, "
        "or any ZIP archive containing them. Check the error details and ensure all five "
        "artifacts are present, then restart Streamlit."
    )
st.subheader("Traffic analysis")
left, right = st.columns([1.4, 1])
with left:
    uploaded = st.file_uploader("Upload a network traffic feature CSV", type=["csv"])
with right:
    st.markdown("#### Before uploading")
    st.write("• Use the encoded feature columns from training.")
    st.write("• Target labels and `id` are not used.")
    st.write("• Maximum upload size: 50 MB.")

if uploaded is not None:
    if uploaded.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File exceeds the {MAX_FILE_SIZE_MB} MB upload limit.")
    else:
        st.caption(f"Selected: **{uploaded.name}** · {uploaded.size / 1024:.1f} KB")

run = st.button("🔎 Run intrusion detection", type="primary",
                disabled=(not models_ready or uploaded is None))
if run and uploaded is not None and models_ready:
    if uploaded.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File exceeds the {MAX_FILE_SIZE_MB} MB upload limit.")
    else:
        try:
            with st.spinner("Validating features and running models..."):
                results, summary, extras = process_csv(
                    uploaded, feature_names, imputer, scaler, rf_model, isolation_model
                )
            st.session_state["ids_results"] = results
            st.session_state["ids_summary"] = summary
            st.session_state["ids_filename"] = uploaded.name
            if extras:
                st.warning("Ignored extra CSV columns: " + ", ".join(map(str, extras[:20])))
            st.success("Analysis completed.")
        except Exception as exc:
            st.session_state.pop("ids_results", None)
            st.session_state.pop("ids_summary", None)
            st.error(str(exc))

if "ids_results" in st.session_state:
    results = st.session_state["ids_results"]
    summary = st.session_state["ids_summary"]
    st.markdown("---")
    st.subheader("Detection overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analyzed", f"{summary['total_records']:,}")
    c2.metric("Hybrid detections", f"{summary['hybrid_attacks']:,}")
    c3.metric("Normal records", f"{summary['hybrid_normal']:,}")
    c4.metric("Isolation anomalies", f"{summary['isolation_forest_anomalies']:,}")

    chart_df = pd.DataFrame({
        "Outcome": ["Suspicious / Attack", "Normal"],
        "Records": [summary["hybrid_attacks"], summary["hybrid_normal"]],
    }).set_index("Outcome")
    chart_col, counts_col = st.columns(2)
    with chart_col:
        st.markdown("#### Hybrid result distribution")
        st.bar_chart(chart_df)
    with counts_col:
        st.markdown("#### Model counts")
        st.dataframe(pd.DataFrame([
            {"Model": "Random Forest", "Flagged records": summary["random_forest_attacks"]},
            {"Model": "Isolation Forest", "Flagged records": summary["isolation_forest_anomalies"]},
            {"Model": "Hybrid rule", "Flagged records": summary["hybrid_attacks"]},
        ]), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Prediction results")
    f1, f2 = st.columns([1, 2])
    with f1:
        selected_filter = st.selectbox("Filter results", ["All", "Suspicious / Attack", "Normal"])
    with f2:
        search = st.text_input("Search by row number or result", placeholder="e.g. 12 or Normal")

    shown = results.copy()
    if selected_filter == "Suspicious / Attack":
        shown = shown[shown["hybrid_result"] != "Normal"]
    elif selected_filter == "Normal":
        shown = shown[shown["hybrid_result"] == "Normal"]
    if search.strip():
        mask = shown.astype(str).apply(
            lambda col: col.str.contains(search.strip(), case=False, regex=False)
        ).any(axis=1)
        shown = shown[mask]

    st.caption(f"Showing {min(len(shown), MAX_DISPLAY_ROWS):,} of {len(shown):,} matching rows.")
    st.dataframe(shown.head(MAX_DISPLAY_ROWS), hide_index=True, use_container_width=True, height=420)
    if len(shown) > MAX_DISPLAY_ROWS:
        st.info(f"Table shows the first {MAX_DISPLAY_ROWS} rows; download includes all results.")
    st.download_button(
        "⬇️ Download all predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="network_ids_predictions.csv",
        mime="text/csv",
    )
    st.caption(
        f"Analyzed file: {st.session_state.get('ids_filename', 'uploaded CSV')}. "
        "Predictions are model outputs, not proof of malicious activity."
    )

st.markdown("---")
st.caption("Academic prototype · Uploaded CSV analysis only · No live packet capture")
