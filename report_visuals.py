# report_visuals.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from packaging import version
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# ---------------------
# Config
# ---------------------
DATA = r"D:\project-nova\data\partners_synthetic_upgraded.csv"
TARGET = "nova_score"
SENSITIVE_ATTRS = ["gender", "age_group", "city_tier"]
PRIMARY_SENSITIVE = "city_tier"  # biggest disparity
OUTDIR = Path("reports")
OUTDIR.mkdir(exist_ok=True)

# ---------------------
# Helpers
# ---------------------
def make_ohe():
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        return OneHotEncoder(drop="first", sparse_output=False)
    return OneHotEncoder(drop="first", sparse=False)

def make_preprocessor(cat_cols):
    return ColumnTransformer([("cat", make_ohe(), cat_cols)], remainder="passthrough")

def xgb_reg_default():
    return XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=1, tree_method="hist", verbosity=0
    )

def compute_soft_weights(series, alpha=0.35, clip_q=0.90):
    counts = series.value_counts()
    inv = 1.0 / counts
    raw = series.map(inv).astype(float) ** alpha
    clip_val = np.quantile(raw, clip_q)
    raw = np.minimum(raw, clip_val)
    return raw / raw.mean()

def mae_by_group(y_true, y_pred, sensitive):
    mf = MetricFrame(metrics={"MAE": mean_absolute_error}, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    return float(mf.overall["MAE"]), mf.by_group["MAE"]

def binarize_on_train_median(y_train, y):
    thr = np.median(y_train)
    return (y > thr).astype(int), thr

def dp_eo(y_true_bin, y_pred_bin, sensitive):
    dp = demographic_parity_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive)
    eo = equalized_odds_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive)
    return float(dp), float(eo)

def bar_two_series(series_a, series_b, title, ylabel, labels=("Baseline","Mitigated"), fname="plot.png"):
    idx = series_a.index.tolist()
    a_vals = series_a.values
    b_vals = series_b.reindex(idx).values
    x = np.arange(len(idx))
    w = 0.38

    plt.figure(figsize=(8,5))
    plt.bar(x - w/2, a_vals, width=w, label=labels[0])
    plt.bar(x + w/2, b_vals, width=w, label=labels[1])
    plt.xticks(x, idx, rotation=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=150)
    plt.close()

def bar_one_series(series, title, ylabel, fname="plot.png"):
    plt.figure(figsize=(7,5))
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=150)
    plt.close()

def residual_boxplot(residuals, groups, title, fname):
    # residuals: array-like; groups: categorical series aligned with residuals
    df = pd.DataFrame({"resid": residuals, "group": groups})
    order = df.groupby("group")["resid"].median().sort_values().index.tolist()
    data = [df.loc[df["group"] == g, "resid"].values for g in order]

    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=[str(g) for g in order], showmeans=True)
    plt.title(title)
    plt.ylabel("Residual (y_true - y_pred)")
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=150)
    plt.close()

# ---------------------
# Load data & split
# ---------------------
df = pd.read_csv(DATA)

# keep sensitive columns for audit; drop them from model features
drop_cols = [TARGET]
if "partner_id" in df.columns:
    drop_cols.append("partner_id")
for a in SENSITIVE_ATTRS:
    if a in df.columns:
        drop_cols.append(a)

X_raw = df.drop(columns=drop_cols)
y = df[TARGET]

sens_all = {a: df[a].copy() for a in SENSITIVE_ATTRS if a in df.columns}
cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()

stratify_series = sens_all.get(PRIMARY_SENSITIVE, None)
X_train, X_test, y_train, y_test, sens_train_df, sens_test_df = train_test_split(
    X_raw, y, pd.DataFrame(sens_all), test_size=0.2, random_state=42,
    stratify=stratify_series if stratify_series is not None else None
)
sens_train = {k: sens_train_df[k] for k in sens_all}
sens_test  = {k: sens_test_df[k] for k in sens_all}

# ---------------------
# Baseline regressor
# ---------------------
prep = make_preprocessor(cat_cols)
reg = xgb_reg_default()
pipe = Pipeline([("prep", prep), ("reg", reg)])
pipe.fit(X_train, y_train)
y_pred_base = pipe.predict(X_test)
mae_overall_base = mean_absolute_error(y_test, y_pred_base)

# ---------------------
# Mitigated regressor (soft reweighting on city_tier)
# ---------------------
if PRIMARY_SENSITIVE in sens_train:
    weights = compute_soft_weights(sens_train[PRIMARY_SENSITIVE], alpha=0.35, clip_q=0.90)
    reg2 = xgb_reg_default()
    pipe_m = Pipeline([("prep", prep), ("reg", reg2)])
    pipe_m.fit(X_train, y_train, reg__sample_weight=weights.values)
    y_pred_mitig = pipe_m.predict(X_test)
    mae_overall_mitig = mean_absolute_error(y_test, y_pred_mitig)
else:
    y_pred_mitig = y_pred_base.copy()
    mae_overall_mitig = mae_overall_base

# ---------------------
# Regression fairness plots
# ---------------------
summary_lines = []
summary_lines.append(f"Overall MAE - Baseline:  {mae_overall_base:.2f}")
summary_lines.append(f"Overall MAE - Mitigated: {mae_overall_mitig:.2f}")

for attr in ["gender", "age_group", "city_tier"]:
    if attr not in sens_test:
        continue
    _, grp_mae_base = mae_by_group(y_test, y_pred_base, sens_test[attr])
    _, grp_mae_mitig = mae_by_group(y_test, y_pred_mitig, sens_test[attr])
    bar_two_series(
        grp_mae_base, grp_mae_mitig,
        title=f"MAE by {attr} (Baseline vs Mitigated)",
        ylabel="MAE",
        fname=f"mae_by_{attr}.png"
    )

# Mean predicted score by city_tier (to see directionality)
if "city_tier" in sens_test:
    by_tier_base = pd.Series(y_pred_base).groupby(sens_test["city_tier"].values).mean()
    by_tier_mitig = pd.Series(y_pred_mitig).groupby(sens_test["city_tier"].values).mean()
    bar_two_series(
        by_tier_base, by_tier_mitig,
        title="Mean Predicted Nova Score by city_tier",
        ylabel="Predicted score",
        fname="pred_mean_by_city_tier.png"
    )
    # Residuals boxplot by city_tier
    residuals_base = (y_test.values - y_pred_base)
    residuals_mitig = (y_test.values - y_pred_mitig)
    residual_boxplot(residuals_base, sens_test["city_tier"], "Residuals by city_tier (Baseline)", "residuals_city_tier_baseline.png")
    residual_boxplot(residuals_mitig, sens_test["city_tier"], "Residuals by city_tier (Mitigated)", "residuals_city_tier_mitigated.png")

# ---------------------
# Approval classifier fairness (DP on city_tier)
# ---------------------
approve_train, thr = binarize_on_train_median(y_train, y_train)
approve_test, _ = binarize_on_train_median(y_train, y_test)

# Baseline approval = threshold on predicted nova_score from baseline regressor
approve_pred_base = (y_pred_base > thr).astype(int)
acc_base = accuracy_score(approve_test, approve_pred_base)
dp_base, eo_base = dp_eo(approve_test, approve_pred_base, sens_test["city_tier"])

mf_sel_base = MetricFrame(
    metrics={"sel_rate": selection_rate, "accuracy": accuracy_score},
    y_true=approve_test, y_pred=approve_pred_base, sensitive_features=sens_test["city_tier"]
)
sel_base = mf_sel_base.by_group["sel_rate"]
bar_one_series(sel_base, "Selection Rate by city_tier (Baseline Approval)", "Selection rate", "sel_rate_city_tier_baseline.png")

# Fairness-constrained classifier via ExponentiatedGradient (reusing features)
prep_cls = make_preprocessor(cat_cols)
prep_cls.fit(X_train)
Xtr = prep_cls.transform(X_train)
Xte = prep_cls.transform(X_test)

eg = ExponentiatedGradient(
    estimator=None,  # LogisticRegression created internally if None? safer to pass explicitly
    constraints=DemographicParity(),
    eps=0.05  # tune this: 0.02 stricter, 0.08 looser
)
from sklearn.linear_model import LogisticRegression
eg = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=200),
    constraints=DemographicParity(),
    eps=0.05
)
eg.fit(Xtr, approve_train, sensitive_features=sens_train["city_tier"])
approve_pred_fair = eg.predict(Xte)
acc_fair = accuracy_score(approve_test, approve_pred_fair)
dp_fair, eo_fair = dp_eo(approve_test, approve_pred_fair, sens_test["city_tier"])

mf_sel_fair = MetricFrame(
    metrics={"sel_rate": selection_rate, "accuracy": accuracy_score},
    y_true=approve_test, y_pred=approve_pred_fair, sensitive_features=sens_test["city_tier"]
)
sel_fair = mf_sel_fair.by_group["sel_rate"]
bar_one_series(sel_fair, "Selection Rate by city_tier (Fair Approval)", "Selection rate", "sel_rate_city_tier_fair.png")

# Side-by-side selection rates (baseline vs fair)
sel_compare = pd.DataFrame({"Baseline": sel_base, "Fair": sel_fair})
ax = sel_compare.plot(kind="bar", figsize=(7,5))
ax.set_title("Selection Rate by city_tier (Baseline vs Fair)")
ax.set_ylabel("Selection rate")
plt.tight_layout()
plt.savefig(OUTDIR / "sel_rate_city_tier_compare.png", dpi=150)
plt.close()

# ---------------------
# Save summary
# ---------------------
summary_lines.append("")
summary_lines.append("=== Approval (city_tier) ===")
summary_lines.append(f"Baseline Accuracy: {acc_base:.3f} | DP: {dp_base:.4f} | EO: {eo_base:.4f}")
summary_lines.append(f"Fair     Accuracy: {acc_fair:.3f} | DP: {dp_fair:.4f} | EO: {eo_fair:.4f}")
summary_lines.append("")
summary_lines.append("Selection rate by city_tier (baseline):")
summary_lines.append(sel_base.to_string())
summary_lines.append("")
summary_lines.append("Selection rate by city_tier (fair):")
summary_lines.append(sel_fair.to_string())

(OUTDIR / "Fairness_Summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print(f"\nSaved charts and summary to: {OUTDIR.resolve()}")
