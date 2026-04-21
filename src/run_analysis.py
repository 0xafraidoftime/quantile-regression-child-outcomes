"""
run_analysis.py
---------------
End-to-end runner for the quantile regression analysis of child academic
outcomes. Generates synthetic ECLS-K data, fits OLS + quantile regression
models, and saves all outputs (plots + tables) to the outputs/ directory.

Usage:
    python src/run_analysis.py
"""

from pathlib import Path
import pandas as pd
from generate_data import generate_ecls_synthetic
from quantile_analysis import (
    fit_quantile_models,
    build_coef_table,
    build_summary_table,
    plot_coef_across_quantiles,
    plot_outcome_distributions,
)

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PREDICTORS_OF_INTEREST = [
    "ses_quintile",
    "vocab_baseline",
    "phonological",
    "special_ed_flag",
]


def main():
    print("=" * 60)
    print("Quantile Regression: Child Academic Outcomes (ECLS-K style)")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/4] Generating synthetic ECLS-K dataset...")
    df = generate_ecls_synthetic(n=2000)
    print(f"      Dataset: {df.shape[0]} children × {df.shape[1]} variables")

    # 2. Fit models for reading
    print("\n[2/4] Fitting QR + OLS models for Grade 4 Reading Score...")
    reading_models = fit_quantile_models(df, outcome="reading_w5")

    # 3. Fit models for math
    print("\n[3/4] Fitting QR + OLS models for Grade 4 Math Score...")
    math_models = fit_quantile_models(df, outcome="math_w5")

    # 4. Save outputs
    print("\n[4/4] Generating outputs...")

    # Distribution plots
    plot_outcome_distributions(
        df, outcome="reading_w5",
        save_path=OUT / "reading_outcome_distributions.png"
    )

    # Coefficient plots — reading
    plot_coef_across_quantiles(
        reading_models,
        predictors=PREDICTORS_OF_INTEREST,
        outcome_label="Grade 4 Reading Score",
        save_path=OUT / "reading_coef_across_quantiles.png",
    )

    # Coefficient plots — math
    plot_coef_across_quantiles(
        math_models,
        predictors=PREDICTORS_OF_INTEREST,
        outcome_label="Grade 4 Math Score",
        save_path=OUT / "math_coef_across_quantiles.png",
    )

    # Summary tables
    reading_table = build_summary_table(reading_models, PREDICTORS_OF_INTEREST)
    math_table = build_summary_table(math_models, PREDICTORS_OF_INTEREST)

    reading_table.to_csv(OUT / "reading_coef_table.csv")
    math_table.to_csv(OUT / "math_coef_table.csv")

    print("\nReading Coefficient Summary:")
    print(reading_table.to_string())
    print("\nMath Coefficient Summary:")
    print(math_table.to_string())

    print(f"\n✓ All outputs saved to {OUT}/")
    print("  Significance codes: *** p<0.001 | ** p<0.01 | * p<0.05")


if __name__ == "__main__":
    main()
