"""
MRI QC Dashboard
=================
A Streamlit dashboard that scores batches of MRI-derived QC features,
classifies likely artifacts with an XGBoost model, and produces a
PASS / REVIEW / FAIL scorecard with drill-down and reporting.

Run with:
    streamlit run app.py
"""

import hashlib
import io
import os
import tempfile
import zipfile

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from feature_extraction import (
    find_hdr_img_pairs,
    extract_scan_features,
    load_volume,
    normalize_volume,
    ALL_FEATURE_COLS,
    BIAS_FEATURE_COLS,
)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MRI QC Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_FEATURES = [
    "SNR",
    "Entropy",
    "LaplacianVariance",
    "GLCMContrast",
    "GLCMEnergy",
    "GLCMHomogeneity",
]

# The active feature set for this session: base 6 plus any bias-detection
# features (BiasQuadrantRange, BiasGradientMagnitude) that turn out to be
# present in the loaded reference data. Finalized once the reference CSV
# is loaded, below.
FEATURES = list(BASE_FEATURES)

DEFAULT_CLASS_SCORES = {
    "original": 95,
    "blur": 55,
    "bias": 60,
    "motion": 10,
    "noise": 15,
}

STATUS_COLORS = {"PASS": "#2ecc71", "REVIEW": "#f39c12", "FAIL": "#e74c3c"}

SAMPLE_PATH = "final_artifact_features.csv"
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_artifact_classifier.joblib")


# --------------------------------------------------------------------------
# Data / model helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    return df


def validate_columns(df: pd.DataFrame):
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    return missing


def _fit_model(train_df: pd.DataFrame):
    X = train_df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(train_df["Artifact"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # refit on the full reference set for best deployed predictions
    model.fit(X, y)

    return model, le, acc, cm, list(le.classes_)


@st.cache_resource(show_spinner="Training XGBoost artifact classifier...")
def train_model(train_df_json: str):
    """Train the XGBoost classifier on the reference labeled dataset, or
    load a previously-trained one from disk (via joblib) if the reference
    data hasn't changed. `st.cache_resource` avoids repeat work within a
    running session; the joblib file on disk avoids repeat work across
    separate `streamlit run` restarts entirely.
    """
    train_df = pd.read_json(io.StringIO(train_df_json), orient="split")
    data_hash = hashlib.md5(train_df_json.encode("utf-8")).hexdigest()

    if os.path.exists(MODEL_PATH):
        try:
            saved = joblib.load(MODEL_PATH)
            if saved.get("data_hash") == data_hash:
                return (
                    saved["model"], saved["le"], saved["acc"],
                    saved["cm"], saved["classes"], True,  # loaded_from_disk
                )
        except Exception:
            pass  # corrupt/incompatible cache file — fall through and retrain

    model, le, acc, cm, classes = _fit_model(train_df)
    _save_model(model, le, acc, cm, classes, data_hash)
    return model, le, acc, cm, classes, False  # loaded_from_disk


def _save_model(model, le, acc, cm, classes, data_hash):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"model": model, "le": le, "acc": acc, "cm": cm,
         "classes": classes, "data_hash": data_hash},
        MODEL_PATH,
    )


def predict_batch(model, le, df: pd.DataFrame):
    X = df[FEATURES]
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    pred_class = le.inverse_transform(pred_idx)
    confidence = proba.max(axis=1)
    return pred_class, confidence, proba


def compute_qc_score(proba: np.ndarray, classes, class_scores: dict) -> np.ndarray:
    weights = np.array([class_scores.get(c, 50) for c in classes])
    return proba @ weights


def score_to_status(score: float, pass_th: float, review_th: float) -> str:
    if score >= pass_th:
        return "PASS"
    elif score >= review_th:
        return "REVIEW"
    else:
        return "FAIL"


def normalize_features(df: pd.DataFrame, ref_df: pd.DataFrame, cols):
    mins = ref_df[cols].min()
    maxs = ref_df[cols].max()
    rng = (maxs - mins).replace(0, 1)
    return (df[cols] - mins) / rng


def _extract_dir(dir_path: str, max_slices: int):
    """Core extraction routine shared by ZIP / local-folder / Kaggle inputs."""
    pairs, warnings = find_hdr_img_pairs(dir_path)
    rows, path_map, errors = [], {}, []
    for p in pairs:
        try:
            feats = extract_scan_features(p["hdr"], max_slices=max_slices)
            meta = {k: v for k, v in feats.items() if k.startswith("_")}
            clean = {k: v for k, v in feats.items() if not k.startswith("_")}
            clean["Scan_ID"] = p["name"]
            clean.update(meta)
            rows.append(clean)
            path_map[p["name"]] = p["hdr"]
        except Exception as e:
            errors.append(f"{p['name']}: {e}")
    return pd.DataFrame(rows), path_map, warnings, errors


@st.cache_data(show_spinner="Unzipping and extracting features from uploaded scans...")
def extract_batch_from_zip(zip_bytes: bytes, max_slices: int):
    """Unzip a batch of .hdr/.img (or .nii/.nii.gz) volumes, pair them up,
    and run the notebook's slice-wise feature pipeline on each scan.
    Cached on (zip contents, max_slices) so re-running the app doesn't
    re-extract every rerun.
    """
    h = hashlib.md5(zip_bytes).hexdigest()[:16]
    extract_dir = os.path.join(tempfile.gettempdir(), f"mriqc_{h}")
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        zpath = os.path.join(extract_dir, "_upload.zip")
        with open(zpath, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extract_dir)
    return _extract_dir(extract_dir, max_slices)


@st.cache_data(show_spinner="Scanning folder and extracting features...")
def extract_batch_from_folder(dir_path: str, max_slices: int, _bust: int = 0):
    """Read .hdr/.img (or .nii/.nii.gz) volumes straight from a local
    folder — no upload involved, so no size limit. `_bust` lets the
    sidebar force a rescan (e.g. after new files were added).
    """
    return _extract_dir(dir_path, max_slices)


@st.cache_data(show_spinner="Downloading dataset from Kaggle...")
def download_kaggle_dataset(slug: str) -> str:
    """Download a public Kaggle dataset via kagglehub and return its
    local cache path. Requires Kaggle API credentials (KAGGLE_USERNAME /
    KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json).
    """
    import kagglehub
    return kagglehub.dataset_download(slug)


@st.cache_data(show_spinner=False)
def load_volume_cached(hdr_path: str):
    volume, _ = load_volume(hdr_path)
    return normalize_volume(volume)


# --------------------------------------------------------------------------
# Sidebar — data loading & configuration
# --------------------------------------------------------------------------
st.sidebar.title("🧠 MRI QC Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📋 Batch Scorecard", "📊 Analytics", "🔍 Scan Drill-down", "🛣️ Roadmap"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Reference (labeled) data")
st.sidebar.caption("Used only to train the XGBoost classifier & set normalization ranges.")
uploaded = st.sidebar.file_uploader("Labeled QC features CSV", type=["csv"], key="ref_csv")

try:
    if uploaded is not None:
        raw_df = load_csv(uploaded)
    else:
        raw_df = load_csv(SAMPLE_PATH)
except FileNotFoundError:
    st.sidebar.error(
        "No sample CSV found and no file uploaded. Please upload a labeled QC features CSV."
    )
    st.stop()

missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(
        f"The reference CSV is missing required feature columns: {missing_cols}. "
        f"Required columns: {BASE_FEATURES}"
    )
    st.stop()

if "Artifact" not in raw_df.columns:
    st.error(
        "The reference/training CSV must include an 'Artifact' label column "
        "for the XGBoost model to train on."
    )
    st.stop()

active_bias_cols = [c for c in BIAS_FEATURE_COLS if c in raw_df.columns]
FEATURES = BASE_FEATURES + active_bias_cols
if active_bias_cols:
    st.sidebar.caption(
        f"✅ Using {len(active_bias_cols)} bias-detection feature(s) "
        f"({', '.join(active_bias_cols)}) from the reference data."
    )
else:
    st.sidebar.caption(
        "ℹ️ Reference data doesn't include the newer bias-detection features "
        "(BiasQuadrantRange / BiasGradientMagnitude) — training on the base "
        "6 features only. Re-extract your reference set from raw volumes "
        "with the updated pipeline to pick these up."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("2. Scan batch to score")
st.sidebar.caption("Contains paired .hdr/.img (or .nii/.nii.gz) volumes — one pair per scan.")

input_method = st.sidebar.radio(
    "Input method",
    ["Upload ZIP", "Local folder path", "Download from Kaggle"],
    help="For datasets over a few hundred MB, prefer 'Local folder path' or "
         "'Download from Kaggle' — browser ZIP upload is slow at that size "
         "and this app runs locally anyway, so it can just read the disk directly.",
)

max_slices = st.sidebar.slider(
    "Max axial slices sampled per scan", 8, 256, 48, step=8,
    help="Feature extraction runs per-slice GLCM/gradient computations; "
         "lower this for faster extraction on large batches.",
)

scan_path_map = {}
extraction_warnings, extraction_errors = [], []
extracted_df = None
data_source_label = None

if input_method == "Upload ZIP":
    zip_upload = st.sidebar.file_uploader("Scans ZIP", type=["zip"], key="scan_zip")
    if zip_upload is not None:
        zip_bytes = zip_upload.getvalue()
        extracted_df, scan_path_map, extraction_warnings, extraction_errors = extract_batch_from_zip(
            zip_bytes, max_slices
        )
        data_source_label = f"{zip_upload.name} → {len(extracted_df)} scans extracted"

elif input_method == "Local folder path":
    folder_path = st.sidebar.text_input(
        "Folder path (on the machine running this app)",
        placeholder="/path/to/dataset  or  C:\\path\\to\\dataset",
    )
    rescan = st.sidebar.button("🔄 (Re)scan folder")
    if "folder_bust" not in st.session_state:
        st.session_state.folder_bust = 0
    if rescan:
        st.session_state.folder_bust += 1

    if folder_path:
        if not os.path.isdir(folder_path):
            st.sidebar.error(f"'{folder_path}' is not a folder this app can see.")
        else:
            extracted_df, scan_path_map, extraction_warnings, extraction_errors = extract_batch_from_folder(
                folder_path, max_slices, st.session_state.folder_bust
            )
            data_source_label = f"{folder_path} → {len(extracted_df)} scans extracted"

else:  # Download from Kaggle
    st.sidebar.caption(
        "Requires Kaggle API credentials on this machine — either a "
        "`~/.kaggle/kaggle.json` file, or `KAGGLE_USERNAME`/`KAGGLE_KEY` "
        "environment variables. Get a key from kaggle.com → Account → "
        "Create New Token."
    )
    kaggle_slug = st.sidebar.text_input(
        "Kaggle dataset", placeholder="owner/dataset-slug (from the dataset's URL)"
    )
    fetch = st.sidebar.button("⬇️ Download dataset")
    if fetch and kaggle_slug:
        try:
            local_path = download_kaggle_dataset(kaggle_slug)
            st.session_state["kaggle_local_path"] = local_path
        except ImportError:
            st.sidebar.error("`kagglehub` isn't installed — run `pip install kagglehub`.")
        except Exception as e:
            st.sidebar.error(f"Kaggle download failed: {e}")

    local_path = st.session_state.get("kaggle_local_path")
    if local_path:
        st.sidebar.caption(f"Using cached download: `{local_path}`")
        extracted_df, scan_path_map, extraction_warnings, extraction_errors = extract_batch_from_folder(
            local_path, max_slices
        )
        data_source_label = f"kaggle:{kaggle_slug or ''} → {len(extracted_df)} scans extracted"

if extracted_df is not None and not extracted_df.empty:
    df = extracted_df.copy()
else:
    if extracted_df is not None and extracted_df.empty:
        st.error(
            "No usable .hdr/.img (or .nii/.nii.gz) pairs were found or successfully "
            "processed in that batch."
        )
        if extraction_warnings:
            st.write(extraction_warnings)
        if extraction_errors:
            st.write(extraction_errors)
        st.stop()
    # Demo fallback: nothing supplied yet — score the reference set itself
    df = raw_df.copy()
    if "MRI_ID" in df.columns:
        df["Scan_ID"] = (
            df["MRI_ID"].astype(str)
            + " | " + df.get("Artifact", "").astype(str)
            + " | " + df.get("Level", "").astype(str)
        )
    else:
        df["Scan_ID"] = df.index.astype(str)
    data_source_label = "no batch supplied yet — scoring the reference CSV as a demo"

st.sidebar.caption(f"Batch: **{data_source_label}**")
if extraction_warnings:
    with st.sidebar.expander(f"⚠️ {len(extraction_warnings)} unmatched file(s)"):
        for w in extraction_warnings:
            st.write(w)
if extraction_errors:
    with st.sidebar.expander(f"❌ {len(extraction_errors)} scan(s) failed to process"):
        for e in extraction_errors:
            st.write(e)

st.sidebar.markdown("---")
st.sidebar.subheader("QC Scoring Rules")
with st.sidebar.expander("Class quality weights (0-100)", expanded=False):
    class_scores = {}
    for cls, default in DEFAULT_CLASS_SCORES.items():
        class_scores[cls] = st.slider(f"{cls}", 0, 100, default, key=f"w_{cls}")

pass_th = st.sidebar.slider("PASS threshold (≥)", 0, 100, 75)
review_th = st.sidebar.slider("REVIEW threshold (≥)", 0, 100, 45)
if review_th >= pass_th:
    st.sidebar.warning("REVIEW threshold should be lower than PASS threshold.")

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
force_retrain = st.sidebar.button(
    "🔁 Force retrain (ignore saved model)",
    help=f"The trained model is cached to disk at `{MODEL_PATH}` and reused "
         "automatically as long as the reference CSV doesn't change. Use "
         "this to retrain anyway.",
)

# --------------------------------------------------------------------------
# Train model + run predictions (cached)
# --------------------------------------------------------------------------
train_json = raw_df.to_json(orient="split")

if force_retrain:
    train_df_now = pd.read_json(io.StringIO(train_json), orient="split")
    data_hash = hashlib.md5(train_json.encode("utf-8")).hexdigest()
    model, le, holdout_acc, cm, classes = _fit_model(train_df_now)
    _save_model(model, le, holdout_acc, cm, classes, data_hash)
    train_model.clear()  # drop stale in-memory cache so future reruns pick up the new file
    loaded_from_disk = False
else:
    model, le, holdout_acc, cm, classes, loaded_from_disk = train_model(train_json)

st.sidebar.caption(
    f"Model: {'📦 loaded from `saved_models/`' if loaded_from_disk else '🆕 freshly trained'} "
    f"— hold-out accuracy {holdout_acc:.1%}"
)

pred_class, confidence, proba = predict_batch(model, le, df)
qc_score = compute_qc_score(proba, classes, class_scores)
qc_status = np.array(
    [score_to_status(s, pass_th, review_th) for s in qc_score]
)

df["Predicted_Artifact"] = pred_class
df["Prediction_Confidence"] = confidence
df["QC_Score"] = qc_score.round(1)
df["QC_Status"] = qc_status


# ==========================================================================
# PAGE 1 — Batch Scorecard
# ==========================================================================
if page == "📋 Batch Scorecard":
    st.title("📋 Batch QC Scorecard")
    st.caption(
        "XGBoost artifact classifier trained on the loaded reference dataset "
        f"— hold-out accuracy **{holdout_acc:.1%}** on classes: {', '.join(classes)}."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Scans", len(df))
    c2.metric("✅ PASS", int((df.QC_Status == "PASS").sum()))
    c3.metric("⚠️ REVIEW", int((df.QC_Status == "REVIEW").sum()))
    c4.metric("❌ FAIL", int((df.QC_Status == "FAIL").sum()))
    c5.metric("Avg QC Score", f"{df.QC_Score.mean():.1f}")

    st.markdown("---")

    def highlight_status(val):
        color = STATUS_COLORS.get(val, "")
        return f"background-color: {color}; color: white; font-weight: 600;"

    display_cols = ["Scan_ID", "Predicted_Artifact", "Prediction_Confidence",
                     "QC_Score", "QC_Status"] + FEATURES
    styled = (
        df[display_cols]
        .style.map(highlight_status, subset=["QC_Status"])
        .format({"Prediction_Confidence": "{:.1%}", "QC_Score": "{:.1f}"})
    )
    st.dataframe(styled, width='stretch', height=450)

    st.markdown("---")
    st.subheader("Export QC Report")
    report_df = df[["Scan_ID"] + (["MRI_ID"] if "MRI_ID" in df.columns else []) +
                    (["Artifact"] if "Artifact" in df.columns else []) +
                    (["Level"] if "Level" in df.columns else []) +
                    FEATURES +
                    ["Predicted_Artifact", "Prediction_Confidence", "QC_Score", "QC_Status"]]
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download QC Report (CSV)",
        data=csv_bytes,
        file_name="mri_qc_report.csv",
        mime="text/csv",
    )

    with st.expander("Model performance details (hold-out test split)"):
        cm_fig = px.imshow(
            cm, x=classes, y=classes, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        cm_fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(cm_fig, width='stretch')


# ==========================================================================
# PAGE 2 — Analytics
# ==========================================================================
elif page == "📊 Analytics":
    st.title("📊 Batch Analytics")

    tab1, tab2, tab3 = st.tabs(
        ["Artifact Distribution", "Feature Breakdown", "Radar Comparison"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            dist = df["Predicted_Artifact"].value_counts().reset_index()
            dist.columns = ["Artifact", "Count"]
            fig = px.bar(dist, x="Artifact", y="Count", color="Artifact",
                         title="Predicted Artifact Distribution")
            st.plotly_chart(fig, width='stretch')
        with col2:
            status_dist = df["QC_Status"].value_counts().reset_index()
            status_dist.columns = ["Status", "Count"]
            fig2 = px.pie(
                status_dist, names="Status", values="Count", hole=0.45,
                color="Status", color_discrete_map=STATUS_COLORS,
                title="PASS / REVIEW / FAIL Split",
            )
            st.plotly_chart(fig2, width='stretch')

    with tab2:
        feat_choice = st.selectbox("Feature", FEATURES, index=0)
        fig3 = px.box(
            df, x="Predicted_Artifact", y=feat_choice, color="Predicted_Artifact",
            points="outliers", title=f"{feat_choice} by Predicted Artifact Class",
        )
        st.plotly_chart(fig3, width='stretch')

        avg_by_class = df.groupby("Predicted_Artifact")[FEATURES].mean().reset_index()
        fig4 = px.bar(
            avg_by_class.melt(id_vars="Predicted_Artifact", var_name="Feature", value_name="Value"),
            x="Feature", y="Value", color="Predicted_Artifact", barmode="group",
            title="Average Feature Profile by Predicted Class",
        )
        st.plotly_chart(fig4, width='stretch')

    with tab3:
        st.caption("Feature values min-max normalized (0-1) against the reference dataset.")
        norm_all = normalize_features(df, raw_df, FEATURES)
        norm_all["Predicted_Artifact"] = df["Predicted_Artifact"].values
        radar_avg = norm_all.groupby("Predicted_Artifact")[FEATURES].mean()

        fig5 = go.Figure()
        for cls in radar_avg.index:
            vals = radar_avg.loc[cls, FEATURES].tolist()
            fig5.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=FEATURES + [FEATURES[0]],
                fill="toself", name=cls,
            ))
        fig5.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Feature Radar — Average per Predicted Class",
            showlegend=True,
        )
        st.plotly_chart(fig5, width='stretch')


# ==========================================================================
# PAGE 3 — Scan Drill-down
# ==========================================================================
elif page == "🔍 Scan Drill-down":
    st.title("🔍 Scan Drill-down")

    scan_id = st.selectbox("Select scan", df["Scan_ID"].tolist())
    row = df[df["Scan_ID"] == scan_id].iloc[0]

    status = row["QC_Status"]
    color = STATUS_COLORS[status]
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:10px;background-color:{color};color:white;">
        <h3 style="margin:0;">Status: {status}</h3>
        <p style="margin:0;">Predicted artifact: <b>{row['Predicted_Artifact']}</b>
        &nbsp;|&nbsp; Confidence: <b>{row['Prediction_Confidence']:.1%}</b>
        &nbsp;|&nbsp; QC Score: <b>{row['QC_Score']:.1f}</b> / 100</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("###")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Probabilities")
        prob_row = proba[df["Scan_ID"].tolist().index(scan_id)]
        prob_df = pd.DataFrame({"Class": classes, "Probability": prob_row})
        fig6 = px.bar(prob_df, x="Class", y="Probability", color="Class", range_y=[0, 1])
        st.plotly_chart(fig6, width='stretch')

    with col2:
        st.subheader("Feature Profile vs. Dataset Average")
        scan_norm = normalize_features(row.to_frame().T, raw_df, FEATURES).iloc[0]
        dataset_norm_mean = normalize_features(raw_df, raw_df, FEATURES).mean()
        fig7 = go.Figure()
        fig7.add_trace(go.Scatterpolar(
            r=scan_norm.tolist() + [scan_norm.tolist()[0]],
            theta=FEATURES + [FEATURES[0]], fill="toself", name="This scan",
        ))
        fig7.add_trace(go.Scatterpolar(
            r=dataset_norm_mean.tolist() + [dataset_norm_mean.tolist()[0]],
            theta=FEATURES + [FEATURES[0]], fill="toself", name="Dataset average",
            opacity=0.5,
        ))
        fig7.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig7, width='stretch')

    st.subheader("Raw Feature Values")
    st.dataframe(row[ALL_FEATURE_COLS if all(c in df.columns for c in ALL_FEATURE_COLS) else FEATURES]
                 .to_frame().T, width='stretch')

    st.markdown("---")
    st.subheader("Slice Viewer")
    if scan_id in scan_path_map:
        volume = load_volume_cached(scan_path_map[scan_id])
        n_slices = volume.shape[2]
        z = st.slider("Axial slice", 0, n_slices - 1, n_slices // 2)
        fig8 = px.imshow(
            volume[:, :, z].T, color_continuous_scale="gray", origin="lower",
            labels=dict(color="Intensity"),
        )
        fig8.update_layout(title=f"Axial slice {z + 1} / {n_slices}", coloraxis_showscale=False)
        fig8.update_xaxes(visible=False)
        fig8.update_yaxes(visible=False)
        st.plotly_chart(fig8, width='stretch')
    else:
        st.caption(
            "No raw volume available for this scan (it came from a precomputed "
            "feature CSV). Upload a ZIP of .hdr/.img or .nii(.gz) volumes to enable "
            "the slice viewer for this scan."
        )


# ==========================================================================
# PAGE 4 — Roadmap
# ==========================================================================
else:
    st.title("🛣️ Roadmap — Future Integrations")

    st.success(
        "**✅ NIfTI / Analyze Viewer — live.** Upload a ZIP of .hdr/.img or "
        ".nii(.gz) volumes and open **Scan Drill-down** to page through axial "
        "slices for any scan."
    )
    st.caption("Still planned:")

    r1, r2 = st.columns(2)
    with r1:
        st.info(
            "**🧠 Skull-strip Overlay**\n\nRun a brain-extraction model (e.g. "
            "HD-BET, FSL BET) on each uploaded volume and overlay the resulting "
            "mask on the slice viewer, so under/over-stripping is visible "
            "alongside the QC score."
        )
    with r2:
        st.info(
            "**🌐 ABIDE Site Analysis**\n\nWhen filenames or metadata encode "
            "acquisition site/scanner (as in ABIDE), group the scorecard by "
            "site to surface site-level bias — e.g. a `Site` column parsed "
            "from the `Scan_ID` or a sidecar CSV joined on scan name."
        )

    st.markdown("---")
    st.write(
        "Implementation notes: the slice viewer reuses `feature_extraction.load_volume` "
        "(a thin `nibabel` wrapper). A skull-strip overlay would add a second array "
        "(the mask) plotted with `opacity` over the same `plotly` `imshow` figure. "
        "Site analysis just needs a `groupby` on top of the existing `QC_Score` / "
        "`QC_Status` columns — no new extraction work required."
    )
