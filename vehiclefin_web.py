# ------------------------------------------------------------
# Vehicle Loan Default – Streamlit Analysis & Modeling App
# ------------------------------------------------------------
# Features:
# - Upload train/test CSVs
# - Data preview + schema infer
# - Toggleable steps: Missing-value imputation, Outliers (IQR/IsolationForest),
#   Feature engineering (OHE + Scaling), VarianceThreshold, SMOTE, PCA
# - Models: LogisticRegression, DecisionTree, RandomForest, XGBClassifier, AdaBoostClassifier
# - Metrics: Accuracy, ROC-AUC, Confusion Matrix, Classification Report
# - Single-record scoring with dynamic form widgets
# - Download trained models and preprocessing pipeline
# ------------------------------------------------------------

import io
import sys
import json
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # not used by default, kept for extendability
from sklearn.svm import SVC  # not used by default, kept for extendability

from sklearn.ensemble import IsolationForest
from scipy.stats import iqr

try:
    from imblearn.over_sampling import SMOTE
    IMB_OK = True
except Exception:
    IMB_OK = False

# XGBoost optional
XGB_OK = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_OK = False

# -----------------------------
# UI – Sidebar
# -----------------------------
st.set_page_config(page_title="Vehicle Loan Default – ML Workbench", layout="wide")
st.title("Vehicle Loan Default – Streamlit Analysis & Modeling")

st.sidebar.header("1) Data Upload")
train_file = st.sidebar.file_uploader("Upload TRAIN CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload TEST CSV (optional)", type=["csv"])

st.sidebar.header("2) Target & Split")
default_target = "LOAN_DEFAULT"
target_col = st.sidebar.text_input("Target column", value=default_target)
test_size = st.sidebar.slider("Test size (hold-out %)", min_value=10, max_value=40, value=20, step=5) / 100.0
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("3) Preprocessing Steps")
do_impute = st.sidebar.checkbox("Missing-value imputation", value=True)
impute_num_strategy = st.sidebar.selectbox("Numeric imputer", ["median", "mean"], index=0)
impute_cat_strategy = st.sidebar.selectbox("Categorical imputer", ["most_frequent", "constant"], index=0)

do_outliers = st.sidebar.checkbox("Outlier handling", value=True)
outlier_method = st.sidebar.selectbox("Method", ["IQR capping", "IsolationForest"], index=0)
iqr_k = st.sidebar.slider("IQR cap multiplier (k)", 1.0, 3.0, 1.5, 0.1)
if_params = {
    "n_estimators": st.sidebar.slider("IF n_estimators", 50, 400, 100, 25),
    "contamination": st.sidebar.slider("IF contamination", 0.01, 0.20, 0.05, 0.01),
}

do_variance = st.sidebar.checkbox("VarianceThreshold", value=True)
vt_threshold = st.sidebar.slider("Variance threshold", 0.0, 0.20, 0.05, 0.01)

do_scale = st.sidebar.checkbox("StandardScaler for numeric", value=True)
do_ohe = st.sidebar.checkbox("One-Hot Encode categoricals", value=True)

do_smote = st.sidebar.checkbox("SMOTE (class imbalance)", value=True and IMB_OK, disabled=not IMB_OK)
if not IMB_OK and do_smote:
    st.sidebar.warning("imblearn not installed; SMOTE disabled.")

do_pca = st.sidebar.checkbox("PCA (after encoding & scaling)", value=False)
pca_components = st.sidebar.slider("PCA components (0 = auto)", 0, 50, 0, 1)

st.sidebar.header("4) Models")
model_choices = []
model_map = {
    "Logistic Regression": LogisticRegression(max_iter=200, n_jobs=None if sys.platform == "darwin" else -1),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
}
if XGB_OK:
    model_map["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, random_state=random_state,
        eval_metric="logloss", tree_method="hist"
    )

default_models = ["Logistic Regression", "Decision Tree", "Random Forest"] + (["XGBoost"] if XGB_OK else [])
chosen_models = st.sidebar.multiselect("Select models to train", list(model_map.keys()), default=default_models)

cv_folds = st.sidebar.slider("Cross-validation folds", 3, 10, 5, 1)

st.sidebar.header("5) Run")
run_button = st.sidebar.button("Run Training & Evaluation", type="primary")

# -----------------------------
# Helper utilities
# -----------------------------

def iqr_capper(df, num_cols, k=1.5):
    """Winsorize numeric columns using IQR capping."""
    capped = df.copy()
    for c in num_cols:
        s = capped[c].astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        i = q3 - q1
        low = q1 - k * i
        high = q3 + k * i
        capped[c] = np.clip(s, low, high)
    return capped

def isolation_forest_filter(df, num_cols, n_estimators=100, contamination=0.05, rs=42):
    """Remove outliers based on IsolationForest; keep inliers only."""
    if len(num_cols) == 0:
        return df, pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=rs,
        n_jobs=-1
    )
    preds = iso.fit_predict(df[num_cols])
    mask = pd.Series(preds == 1, index=df.index)
    return df.loc[mask].copy(), mask

def build_preprocessor(X, do_impute=True, do_scale=True, do_ohe=True,
                       do_variance=True, vt_threshold=0.05, do_pca=False, pca_components=0):
    """Create a ColumnTransformer + optional PCA."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numeric pipeline
    num_steps = []
    if do_impute:
        num_steps.append(("imputer", SimpleImputer(strategy=impute_num_strategy)))
    if do_scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps) if num_steps else "passthrough"

    # Categorical pipeline
    cat_steps = []
    if do_impute:
        fill_val = "missing" if impute_cat_strategy == "constant" else None
        cat_steps.append(("imputer", SimpleImputer(strategy=impute_cat_strategy, fill_value=fill_val)))
    if do_ohe:
        cat_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    cat_pipe = Pipeline(steps=cat_steps) if cat_steps else "passthrough"

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    steps = [("pre", pre)]
    if do_variance:
        steps.append(("varth", VarianceThreshold(threshold=vt_threshold)))
    if do_pca:
        n_comp = None if pca_components == 0 else pca_components
        steps.append(("pca", PCA(n_components=n_comp, random_state=random_state)))

    preproc = Pipeline(steps=steps)
    return preproc

def df_profile(df, target):
    info = {}
    info["rows"], info["cols"] = df.shape
    info["missing_pct"] = df.isna().mean().mean() * 100
    info["target_distribution"] = None
    if target in df.columns:
        info["target_distribution"] = df[target].value_counts(dropna=False, normalize=True) * 100
    return info

def to_download_link(obj, filename):
    if isinstance(obj, (dict, list)):
        payload = json.dumps(obj, default=str).encode()
    else:
        payload = obj
    b64 = base64.b64encode(payload).decode()
    href = f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}">Download {filename}</a>'
    return href

# -----------------------------
# Data ingestion
# -----------------------------
@st.cache_data
def read_csv_uploaded(file):
    return pd.read_csv(file)

train_df, test_df = None, None
if train_file:
    train_df = read_csv_uploaded(train_file)
    st.subheader("Train Data Preview")
    st.dataframe(train_df.head(20), use_container_width=True)

    profile = df_profile(train_df, target_col)
    st.markdown(f"**Rows × Cols:** {profile['rows']} × {profile['cols']}  |  **Avg Missing %:** {profile['missing_pct']:.2f}%")
    if profile["target_distribution"] is not None:
        st.write("**Target distribution (% of rows):**")
        st.dataframe(profile["target_distribution"].rename("Percent").to_frame(), use_container_width=True)

if test_file:
    test_df = read_csv_uploaded(test_file)
    st.subheader("Test Data Preview (Optional)")
    st.dataframe(test_df.head(20), use_container_width=True)

if not train_df is None and target_col not in train_df.columns:
    st.error(f"Target column '{target_col}' not found in training data.")
    st.stop()

# -----------------------------
# Run Training
# -----------------------------
results = {}
trained_artifacts = {}

if run_button and (train_df is not None):
    df = train_df.copy()

    # Split features/target early
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Outlier handling (on numeric columns only)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if do_outliers:
        if outlier_method == "IQR capping":
            X = iqr_capper(X, num_cols, k=iqr_k)
            st.info("Applied IQR capping to numeric features.")
        else:
            # IsolationForest – remove outliers before split
            joined = X.copy()
            joined[target_col] = y.values
            filtered, mask = isolation_forest_filter(joined, num_cols, **if_params, rs=random_state)
            X, y = filtered.drop(columns=[target_col]), filtered[target_col]
            st.info(f"IsolationForest retained {len(filtered)} inliers out of {len(df)} rows.")

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build preprocessing
    preproc = build_preprocessor(
        X_train,
        do_impute=do_impute,
        do_scale=do_scale,
        do_ohe=do_ohe,
        do_variance=do_variance,
        vt_threshold=vt_threshold,
        do_pca=do_pca,
        pca_components=pca_components
    )

    # Optional SMOTE
    use_smote = do_smote and IMB_OK
    if use_smote:
        st.info("SMOTE will be applied **inside** the training fold only.")

    # Train each model
    for name in chosen_models:
        model = model_map[name]
        clf = Pipeline(steps=[("pre", preproc), ("clf", model)])

        # If SMOTE is enabled, perform it safely within CV loop for metrics.
        # For holdout, we apply SMOTE only on X_train transformed space.
        # To keep pipeline simple in Streamlit, we’ll do:
        # - For CV: report standard CV on original data (approximate view)
        # - For holdout: apply SMOTE after preproc.fit_transform(X_train)
        # For production, imblearn Pipeline is preferred; here we keep clarity.

        # Cross-validation (approximate – without SMOTE to avoid leakage in this quick view)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        try:
            cv_acc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        except Exception:
            cv_acc = np.array([])

        # Fit on training (now handle SMOTE on transformed features if enabled)
        if use_smote:
            # Fit preprocessor, transform train
            Z_train = preproc.fit_transform(X_train, y_train)
            sm = SMOTE(random_state=random_state)
            Z_res, y_res = sm.fit_resample(Z_train, y_train)
            # Fit classifier on resampled transformed space
            model.fit(Z_res, y_res)
            # Build a 2-step predict function that applies preproc then model
            class Wrapped(object):
                def __init__(self, preproc, model):
                    self.pre = preproc
                    self.model = model
                def predict(self, X):
                    Z = self.pre.transform(X)
                    return self.model.predict(Z)
                def predict_proba(self, X):
                    Z = self.pre.transform(X)
                    if hasattr(self.model, "predict_proba"):
                        return self.model.predict_proba(Z)
                    # fallback via decision_function if exists
                    if hasattr(self.model, "decision_function"):
                        s = self.model.decision_function(Z)
                        # convert to 2-col proba via logistic (rough)
                        from sklearn.preprocessing import MinMaxScaler
                        s = MinMaxScaler().fit_transform(s.reshape(-1,1)).ravel()
                        return np.vstack([1-s, s]).T
                    # default
                    preds = self.model.predict(Z)
                    return np.vstack([1-preds, preds]).T
            fitted_model = Wrapped(preproc, model)
        else:
            # Fit entire pipeline
            fitted_model = clf.fit(X_train, y_train)

        # Hold-out evaluation
        y_pred = fitted_model.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        acc_pct = 100.0 * acc

        row = {
            "Model": name,
            "Accuracy % (hold-out)": round(acc_pct, 2),
        }

        # ROC-AUC if possible
        roc_auc = None
        try:
            if hasattr(fitted_model, "predict_proba"):
                proba = fitted_model.predict_proba(X_valid)[:, 1]
                roc_auc = roc_auc_score(y_valid, proba)
            row["ROC-AUC"] = round(roc_auc, 4) if roc_auc is not None else None
        except Exception:
            row["ROC-AUC"] = None

        # CV
        if cv_acc.size > 0:
            row["CV Acc Mean"] = round(100.0 * cv_acc.mean(), 2)
            row["CV Acc Std"] = round(100.0 * cv_acc.std(), 2)

        results[name] = {
            "row": row,
            "confusion_matrix": confusion_matrix(y_valid, y_pred),
            "report": classification_report(y_valid, y_pred, output_dict=True)
        }
        trained_artifacts[name] = fitted_model

    # Display metrics table
    if results:
        st.subheader("Results – Accuracy & Metrics")
        tbl = pd.DataFrame([v["row"] for v in results.values()]).set_index("Model")
        st.dataframe(tbl, use_container_width=True)

        # Detailed per-model views
        for name in results:
            st.markdown(f"### {name}")
            cm = results[name]["confusion_matrix"]
            rep = pd.DataFrame(results[name]["report"]).T
            st.write("**Confusion Matrix**")
            st.dataframe(pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["True 0","True 1"]))

            st.write("**Classification Report**")
            st.dataframe(rep.style.format(precision=3), use_container_width=True)

    # Save artifacts for download (pickle via joblib)
    import joblib
    zip_buf = io.BytesIO()
    with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as _:
        pass  # ensure xlsxwriter present; optional check

    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        # Save a standalone preprocessor (fit on full training for inference)
        preproc_full = build_preprocessor(
            X, do_impute, do_scale, do_ohe, do_variance, vt_threshold, do_pca, pca_components
        ).fit(X, y)
        buf = io.BytesIO()
        joblib.dump(preproc_full, buf)
        zf.writestr("preprocessor.joblib", buf.getvalue())

        for name, mdl in trained_artifacts.items():
            b = io.BytesIO()
            joblib.dump(mdl, b)
            zf.writestr(f"{name.replace(' ','_').lower()}_model.joblib", b.getvalue())

    st.markdown("#### Download Trained Artifacts")
    st.markdown(to_download_link(zip_buf.getvalue(), "vehicle_loan_models.zip"), unsafe_allow_html=True)

# -----------------------------
# Single-record Scoring
# -----------------------------
st.markdown("---")
st.header("Single-Record Scoring (Interactive Inputs)")

if train_df is not None and results:
    # Build dynamic form based on training feature schema
    X_schema = train_df.drop(columns=[target_col])
    num_cols = X_schema.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_schema.select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown("Provide feature values below and select a trained model to score.")
    with st.form("score_form"):
        inputs = {}
        st.subheader("Numeric Features")
        for c in num_cols:
            # sensible defaults using median
            val = float(train_df[c].median() if pd.api.types.is_numeric_dtype(train_df[c]) else 0.0)
            inputs[c] = st.number_input(c, value=val)

        st.subheader("Categorical Features")
        for c in cat_cols:
            # use top category as default
            top_cat = str(train_df[c].mode().iloc[0]) if not train_df[c].dropna().empty else "unknown"
            # choose from observed categories (limited to top 50 to avoid huge lists)
            cats = list(map(str, train_df[c].dropna().astype(str).value_counts().index[:50]))
            if top_cat not in cats:
                cats = [top_cat] + cats
            inputs[c] = st.selectbox(c, options=cats, index=0)

        pick_model = st.selectbox("Model to use", list(trained_artifacts.keys()))
        submitted = st.form_submit_button("Score")
    if submitted:
        x_new = pd.DataFrame([inputs])
        mdl = trained_artifacts[pick_model]
        proba = None
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(x_new)[:, 1][0]
        pred = int(mdl.predict(x_new)[0])
        st.success(f"**Prediction:** {pred}  |  **Probability of Default (if available):** {None if proba is None else round(float(proba), 4)}")

else:
    st.info("Upload data, run training, then return here to use single-record scoring.")

# -----------------------------
# EDA & Step Visuals (Optional)
# -----------------------------
st.markdown("---")
st.header("Exploratory Diagnostics & Step Outputs")
if train_df is not None:
    with st.expander("Missing-Value Heatmap (simple %)"):
        miss = train_df.isna().mean().sort_values(ascending=False) * 100
        st.dataframe(miss.rename("Missing %").to_frame(), use_container_width=True)

    with st.expander("Numeric Summary"):
        st.dataframe(train_df.describe().T, use_container_width=True)

    with st.expander("Categorical Cardinality (Top 30)"):
        cats = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
        card = {c: train_df[c].nunique(dropna=False) for c in cats}
        st.dataframe(pd.Series(card, name="nunique").sort_values(ascending=False).head(30).to_frame())

    with st.expander("Outlier Diagnostics (IQR bounds preview)"):
        ncols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        prev = []
        for c in ncols[:30]:
            s = train_df[c].dropna().astype(float)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            i = q3 - q1
            low, high = q1 - 1.5 * i, q3 + 1.5 * i
            prev.append([c, s.min(), s.max(), low, high])
        if prev:
            st.dataframe(pd.DataFrame(prev, columns=["col","min","max","IQR_low","IQR_high"]))
