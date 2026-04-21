"""
quantile_analysis.py
--------------------
Applies Quantile Regression to ECLS-K style data to examine how predictor
variables (SES, vocabulary, phonological awareness, special education status)
relate to children's reading and math outcomes ACROSS the full distribution —
not just the conditional mean.

This directly mirrors the analytical approach used in Dr. Jessica Logan's
research, where heterogeneous effects across the outcome distribution are
central to understanding individual differences in child development.

Key insight: OLS tells you how predictors affect the *average* child.
Quantile Regression tells you how they affect children at the bottom,
middle, and top of the distribution — which is crucial for identifying
children at risk for learning disabilities.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg

warnings.filterwarnings("ignore")

# ── Plot styling ─────────────────────────────────────────────────────────────
PALETTE = {
    "q10":  "#d62728",
    "q25":  "#ff7f0e",
    "q50":  "#2ca02c",
    "q75":  "#1f77b4",
    "q90":  "#9467bd",
    "ols":  "#333333",
}
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_LABELS = {0.10: "Q10 (at-risk)", 0.25: "Q25", 0.50: "Q50 (median)",
                   0.75: "Q75", 0.90: "Q90 (high achiever)"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fit quantile regression models
# ─────────────────────────────────────────────────────────────────────────────

def fit_quantile_models(df: pd.DataFrame, outcome: str = "reading_w5") -> dict:
    """
    Fit quantile regression models for each quantile in QUANTILES,
    plus an OLS model for comparison.

    Formula includes SES, vocabulary, phonological awareness, working memory,
    parent education, special ed flag, and school type as predictors.

    Returns
    -------
    dict with keys 'ols' and tau values (0.10, 0.25, ...) → fitted models
    """
    formula = (
        f"{outcome} ~ ses_quintile + vocab_baseline + phonological + "
        "working_memory + parent_edu + special_ed_flag + school_type + "
        "teacher_experience_yrs"
    )

    models = {}

    # OLS baseline
    models["ols"] = smf.ols(formula, data=df).fit()

    # Quantile models
    for tau in QUANTILES:
        models[tau] = smf.quantreg(formula, data=df).fit(q=tau)
        print(f"  Fitted QR τ={tau:.2f} | Pseudo-R²={models[tau].prsquared:.3f}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract coefficient table across quantiles
# ─────────────────────────────────────────────────────────────────────────────

def build_coef_table(models: dict, predictor: str = "ses_quintile") -> pd.DataFrame:
    """
    Extract the coefficient, CI lower, and CI upper for a given predictor
    across all quantile models and OLS.
    """
    rows = []
    for key, model in models.items():
        if predictor not in model.params.index:
            continue
        coef = model.params[predictor]
        ci = model.conf_int().loc[predictor]
        label = "OLS" if key == "ols" else f"Q{int(key*100)}"
        tau_val = None if key == "ols" else key
        rows.append({
            "model": label,
            "tau": tau_val,
            "coef": coef,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisation: coefficient plot across quantiles
# ─────────────────────────────────────────────────────────────────────────────

def plot_coef_across_quantiles(
    models: dict,
    predictors: list,
    outcome_label: str = "Grade 4 Reading Score",
    save_path: Path = None,
):
    """
    For each predictor, plot how the coefficient changes across quantiles
    vs. the OLS estimate — revealing heterogeneous effects.
    """
    n_preds = len(predictors)
    fig, axes = plt.subplots(1, n_preds, figsize=(5 * n_preds, 5), sharey=False)
    if n_preds == 1:
        axes = [axes]

    fig.suptitle(
        f"Quantile Regression Coefficients\nOutcome: {outcome_label}",
        fontsize=14, fontweight="bold", y=1.02
    )

    tau_vals = [t for t in QUANTILES]
    colors = ["#d62728","#ff7f0e","#2ca02c","#1f77b4","#9467bd"]

    for ax, pred in zip(axes, predictors):
        coef_df = build_coef_table(models, predictor=pred)
        qr_df = coef_df[coef_df["tau"].notna()].sort_values("tau")
        ols_row = coef_df[coef_df["model"] == "OLS"].iloc[0]

        # Shaded CI band for QR
        ax.fill_between(
            qr_df["tau"], qr_df["ci_low"], qr_df["ci_high"],
            alpha=0.15, color="#1f77b4", label="95% CI (QR)"
        )
        # QR coefficient line
        ax.plot(
            qr_df["tau"], qr_df["coef"],
            marker="o", color="#1f77b4", linewidth=2, markersize=7,
            label="Quantile Reg"
        )
        # OLS horizontal line
        ax.axhline(ols_row["coef"], color="#333333", linewidth=2,
                   linestyle="--", label=f"OLS ({ols_row['coef']:.2f})")
        ax.axhline(ols_row["ci_low"], color="#333333", linewidth=1,
                   linestyle=":", alpha=0.5)
        ax.axhline(ols_row["ci_high"], color="#333333", linewidth=1,
                   linestyle=":", alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.4)

        ax.set_title(pred.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Quantile (τ)", fontsize=10)
        ax.set_ylabel("Coefficient Estimate", fontsize=10)
        ax.set_xticks(QUANTILES)
        ax.set_xticklabels([f"τ={t}" for t in QUANTILES], rotation=30, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [plot] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualisation: distribution of outcomes by SES & special ed
# ─────────────────────────────────────────────────────────────────────────────

def plot_outcome_distributions(df: pd.DataFrame, outcome: str = "reading_w5",
                                save_path: Path = None):
    """
    Violin plots showing outcome distribution stratified by SES quintile
    and special ed flag — motivating why quantile regression is needed
    over OLS (distributions are not uniform across groups).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SES quintile
    sns.violinplot(
        data=df, x="ses_quintile", y=outcome,
        palette="coolwarm", inner="quartile", ax=axes[0]
    )
    axes[0].set_title("Reading Score Distribution by SES Quintile\n"
                       "(showing why mean-only models miss the story)",
                       fontsize=11, fontweight="bold")
    axes[0].set_xlabel("SES Quintile (1=lowest, 5=highest)")
    axes[0].set_ylabel("Grade 4 Reading IRT Score")
    axes[0].grid(True, alpha=0.3)

    # Special ed flag
    df_plot = df.copy()
    df_plot["Special Ed"] = df_plot["special_ed_flag"].map({0: "No IEP", 1: "Has IEP"})
    sns.violinplot(
        data=df_plot, x="Special Ed", y=outcome,
        palette={"No IEP": "#2ca02c", "Has IEP": "#d62728"},
        inner="quartile", ax=axes[1]
    )
    axes[1].set_title("Reading Score Distribution by Special Education Status\n"
                       "(heterogeneous effects visible across distribution)",
                       fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Special Education Status")
    axes[1].set_ylabel("Grade 4 Reading IRT Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [plot] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(models: dict, predictors: list) -> pd.DataFrame:
    """
    Build a clean summary table of coefficients across quantile models
    for the key predictors of interest.
    """
    rows = []
    for pred in predictors:
        row = {"predictor": pred}
        for key in ["ols"] + QUANTILES:
            model = models[key]
            if pred not in model.params.index:
                continue
            coef = model.params[pred]
            pval = model.pvalues[pred]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            label = "OLS" if key == "ols" else f"Q{int(key*100)}"
            row[label] = f"{coef:.3f}{stars}"
        rows.append(row)

    return pd.DataFrame(rows).set_index("predictor")
