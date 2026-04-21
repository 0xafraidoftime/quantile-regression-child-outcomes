# quantile-regression-child-outcomes

> Applying Quantile Regression to understand **how socioeconomic status, vocabulary, phonological awareness, and special education status** relate to children's reading and math achievement — not just at the average, but across the *entire* outcome distribution.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-orange)](https://www.statsmodels.org/)

---

## Motivation

Ordinary Least Squares (OLS) regression tells you how predictors affect the **average child**. But in child development research — especially when studying children at risk for learning disabilities — the average can be deeply misleading.

A child at the **10th percentile** of reading achievement (at-risk) may respond very differently to socioeconomic advantage than a child at the **90th percentile** (high achiever). **Quantile Regression** makes this visible.

This project is directly inspired by the methodological work of Dr. Jessica Logan (Vanderbilt University / Peabody College), whose research applies advanced statistical models — including Quantile Regression — to answer complex questions about how children's academic skills grow and change.

---

## What This Repo Does

1. **Generates** a synthetic dataset modeled on the ECLS-K (Early Childhood Longitudinal Study — Kindergarten Cohort) with 2,000 children tracked from kindergarten through Grade 4
2. **Fits** Quantile Regression models at τ = 0.10, 0.25, 0.50, 0.75, 0.90 alongside OLS for reading and math outcomes
3. **Visualizes** how predictor effects differ across the outcome distribution — revealing patterns OLS hides
4. **Exports** clean coefficient tables and publication-ready plots

---

## Key Research Question

> *Does the effect of SES, vocabulary, or special education status on Grade 4 reading/math scores differ for children at the bottom vs. top of the achievement distribution?*

---

## Project Structure

```
quantile-regression-child-outcomes/
│
├── data/
│   └── synthetic_ecls.csv          # Auto-generated synthetic ECLS-K style data
│
├── src/
│   ├── generate_data.py            # Synthetic data generation
│   ├── quantile_analysis.py        # QR model fitting + visualisation functions
│   └── run_analysis.py             # End-to-end runner
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Step-by-step walkthrough (coming soon)
│
├── outputs/
│   ├── reading_coef_across_quantiles.png
│   ├── math_coef_across_quantiles.png
│   ├── reading_outcome_distributions.png
│   ├── reading_coef_table.csv
│   └── math_coef_table.csv
│
├── tests/
│   └── test_quantile_analysis.py
│
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone & install dependencies

```bash
git clone https://github.com/0xafraidoftime/quantile-regression-child-outcomes.git
cd quantile-regression-child-outcomes
pip install -r requirements.txt
```

### 2. Run the full analysis

```bash
cd src
python run_analysis.py
```

This will:
- Generate `data/synthetic_ecls.csv`
- Fit OLS + QR models for both reading and math
- Save all plots and tables to `outputs/`

---

## Key Variables

| Variable | Description |
|---|---|
| `ses_quintile` | Socioeconomic status composite (1=lowest, 5=highest) |
| `vocab_baseline` | Vocabulary composite at kindergarten entry (0–100) |
| `phonological` | Phonological awareness score (0–50) |
| `working_memory` | Working memory composite (0–100) |
| `parent_edu` | Highest parental education (1=<HS … 4=BA+) |
| `special_ed_flag` | Indicator for IEP / special education services |
| `reading_w1–w5` | Reading IRT scores: Kindergarten → Grade 4 |
| `math_w1–w5` | Math IRT scores: Kindergarten → Grade 4 |

---

## Methodological Notes

### Why Quantile Regression?

- OLS minimises the sum of **squared** residuals → estimates the **conditional mean**
- Quantile Regression minimises the **asymmetrically weighted sum of absolute residuals** → estimates any **conditional quantile**
- For at-risk children research: knowing that SES has a *larger* effect at τ=0.10 than at τ=0.90 is clinically and policy-relevant information that OLS completely suppresses

### Pseudo-R² (Koenker-Machado)

The goodness-of-fit statistic for quantile regression, analogous to R² but computed as:

```
ρ(τ) = 1 - (sum of weighted residuals, fitted) / (sum of weighted residuals, intercept-only)
```

### Data Source Note

All data in this repository is **entirely synthetic**, generated to mirror the structure, variable distributions, and inter-correlations of the publicly available ECLS-K dataset. No real child records are used.

Public ECLS-K data can be accessed via: https://nces.ed.gov/ecls/

---

## References

- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica, 46*(1), 33–50.
- Logan, J. A. R., & Pentimonti, J. M. (methodological work on child development research)
- National Center for Education Statistics. (2010). *Early Childhood Longitudinal Study, Kindergarten Class of 1998–99 (ECLS-K)*. U.S. Department of Education.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

---

## Author

**0xafraidoftime** — [GitHub](https://github.com/0xafraidoftime)

Motivated by Dr. Jessica Logan's research on applying advanced statistical models to child learning and development at Vanderbilt University's Peabody College of Education.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
