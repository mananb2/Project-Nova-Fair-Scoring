# project_nova_final_enhanced_visuals.py

import os
# Set environment variables for stability on different systems
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import shap
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import LogisticRegression

from xgboost import XGBRegressor
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import joblib

# ========= Config =========
DATA_PATH = "data/partners_synthetic_upgraded.csv"
TARGET = "nova_score"
SENSITIVE_ATTRS = ["gender", "age_group", "city_tier"]
PRIMARY_SENSITIVE = "city_tier"
OUTDIR = Path("reports")
OUTDIR.mkdir(exist_ok=True)
np.random.seed(42)

# --- Enhanced Visuals Config ---
plt.style.use('seaborn-v0_8-whitegrid')
BASELINE_COLOR = '#007ACC' # A professional blue
MITIGATED_COLOR = '#FFB900' # A vibrant orange/gold

# ========= Enhanced Plotting Functions =========

def plot_bar_comparison(series_a, series_b, title, ylabel, labels=("Baseline", "Mitigated"), fname="plot.png"):
    """Creates an eye-catching bar chart comparing two series with data labels."""
    idx = series_a.index.astype(str)
    a_vals = series_a.values
    b_vals = series_b.reindex(series_a.index).values
    
    x = np.arange(len(idx))
    w = 0.38
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_a = ax.bar(x - w/2, a_vals, width=w, label=labels[0], color=BASELINE_COLOR, zorder=3)
    bar_b = ax.bar(x + w/2, b_vals, width=w, label=labels[1], color=MITIGATED_COLOR, zorder=3)
    
    # Add precise data labels
    ax.bar_label(bar_a, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(bar_b, padding=3, fmt='%.3f', fontsize=9)
    
    # Improve aesthetics
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(x, idx, fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=150)
    plt.close()

def plot_shap_summary(shap_values, X, feature_names, fname="shap_summary.png"):
    """Saves a styled SHAP summary plot."""
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type='bar')
    plt.title("Feature Importance (SHAP)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=150)
    plt.close()
    
# ========= Helpers =========
def make_preprocessor(df_raw):
    cat_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df_raw.select_dtypes(exclude=["object"]).columns.tolist()
    
    if sklearn.__version__ >= "1.2":
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')
    else:
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown='ignore')
        
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", ohe, cat_cols)
    ], remainder="passthrough")

def save_summary(lines, filename="Fairness_Summary.txt"):
    (OUTDIR / filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved summary to: {OUTDIR.resolve()}/{filename}")

# ========= Load and Prepare Data =========
df = pd.read_csv(DATA_PATH)
X_raw = df.drop(columns=[TARGET, "partner_id"])
y = df[TARGET]
sens_all = df[SENSITIVE_ATTRS]

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X_raw, y, sens_all, test_size=0.2, random_state=42, stratify=sens_all[PRIMARY_SENSITIVE]
)

# ========= 1. Baseline XGB Regressor for Nova Score =========
print("Training the baseline Nova Score regression model...")
preprocessor = make_preprocessor(X_train)
reg_pipe = Pipeline([
    ("prep", preprocessor),
    ("reg", XGBRegressor(random_state=42, n_jobs=1, tree_method="hist", verbosity=0))
])
reg_pipe.fit(X_train, y_train)
y_pred_score = reg_pipe.predict(X_test)
mae_base = mean_absolute_error(y_test, y_pred_score)
print(f"Baseline Nova Score MAE: {mae_base:.2f}")

# ========= 2. Explainability with SHAP =========
print("Generating SHAP explainability plots...")
X_train_transformed = reg_pipe.named_steps['prep'].transform(X_train)
explainer = shap.TreeExplainer(reg_pipe.named_steps['reg'])
shap_values = explainer.shap_values(X_train_transformed)

try:
    feature_names = reg_pipe.named_steps['prep'].get_feature_names_out()
except AttributeError:
    feature_names = reg_pipe.named_steps['prep'].get_feature_names()

plot_shap_summary(shap_values, X_train_transformed, feature_names, fname="shap_summary_plot.png")
print("Saved SHAP explainability plots.")

# ========= 3. Fair Loan Approval Classification =========
loan_approval_threshold = np.median(y_train)
y_train_approve = (y_train > loan_approval_threshold).astype(int)
y_test_approve = (y_test > loan_approval_threshold).astype(int)

# --- Baseline Approval ---
y_pred_approve_base = (y_pred_score > loan_approval_threshold).astype(int)
base_acc = accuracy_score(y_test_approve, y_pred_approve_base)
base_dp_diff = demographic_parity_difference(y_test_approve, y_pred_approve_base, sensitive_features=sens_test[PRIMARY_SENSITIVE])

# --- Mitigated Approval with ExponentiatedGradient ---
print("\nTraining fairness-aware classifier...")
X_train_prep = reg_pipe.named_steps['prep'].transform(X_train)
X_test_prep = reg_pipe.named_steps['prep'].transform(X_test)

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(solver='liblinear', random_state=42),
    constraints=DemographicParity(),
    eps=0.01
)
mitigator.fit(X_train_prep, y_train_approve, sensitive_features=sens_train[PRIMARY_SENSITIVE])
y_pred_approve_mitigated = mitigator.predict(X_test_prep)
mitigated_acc = accuracy_score(y_test_approve, y_pred_approve_mitigated)
mitigated_dp_diff = demographic_parity_difference(y_test_approve, y_pred_approve_mitigated, sensitive_features=sens_test[PRIMARY_SENSITIVE])

print("Fairness analysis complete.")

# ========= 4. Business Impact Analysis =========
print("Generating business impact analysis...")
mf_base = MetricFrame(metrics=selection_rate, y_true=y_test_approve, y_pred=y_pred_approve_base, sensitive_features=sens_test[PRIMARY_SENSITIVE])
mf_mitigated = MetricFrame(metrics=selection_rate, y_true=y_test_approve, y_pred=y_pred_approve_mitigated, sensitive_features=sens_test[PRIMARY_SENSITIVE])

# Generate the enhanced plot
plot_bar_comparison(
    series_a=mf_base.by_group,
    series_b=mf_mitigated.by_group,
    title="Loan Approval Rates by City Tier",
    ylabel="Selection Rate (Approval Rate)",
    fname="business_impact_approvals.png"
)
print("Saved enhanced Business Impact plot.")

# ========= 5. Save Final Summary Report =========
impact_df = pd.DataFrame({
    "Baseline Selection Rate": mf_base.by_group,
    "Mitigated Selection Rate": mf_mitigated.by_group
})
impact_df.index.name = "City Tier"

summary = [
    "======= Project Nova: Fairness & Impact Summary =======",
    f"\nBaseline Nova Score Model MAE: {mae_base:.3f}",
    "\n--- Loan Approval Classification ---",
    f"Approval Threshold (based on median Nova Score): {loan_approval_threshold:.2f}",
    "\n**Baseline Approval Model:**",
    f"  - Accuracy: {base_acc:.3f}",
    f"  - Demographic Parity Difference ({PRIMARY_SENSITIVE}): {base_dp_diff:.4f}",
    "\n**Fairness-Mitigated Approval Model (ExponentiatedGradient):**",
    f"  - Accuracy: {mitigated_acc:.3f}",
    f"  - Demographic Parity Difference ({PRIMARY_SENSITIVE}): {mitigated_dp_diff:.4f}",
    "\n--- Business Impact: Selection Rates by City Tier ---",
    impact_df.to_string()
]
save_summary(summary)


# Save the trained models for the Streamlit app
print("\nSaving models for Streamlit app...")
joblib.dump(reg_pipe, 'nova_regression_pipeline.joblib')
joblib.dump(mitigator, 'fairness_mitigator.joblib')
joblib.dump(loan_approval_threshold, 'loan_approval_threshold.joblib')
print("Models saved successfully.")
# Add this to the very end of the file
joblib.dump(X_train.columns, 'model_columns.joblib')