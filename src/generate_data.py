"""
generate_data.py
----------------
Generates a synthetic dataset modeled after the Early Childhood Longitudinal
Study - Kindergarten Cohort (ECLS-K), capturing key variables relevant to
child academic skill development from preschool through 4th grade.

Variables are based on published ECLS-K codebooks and research literature.
All data is entirely synthetic — no real child records are used.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)


def generate_ecls_synthetic(n: int = 2000, save: bool = True) -> pd.DataFrame:
    """
    Generate a synthetic ECLS-K style longitudinal dataset.

    Parameters
    ----------
    n : int
        Number of children to simulate.
    save : bool
        If True, saves the CSV to data/synthetic_ecls.csv.

    Returns
    -------
    pd.DataFrame
    """

    # ── Demographic & SES variables ──────────────────────────────────────────
    sex = rng.choice([0, 1], size=n, p=[0.49, 0.51])            # 0=Male, 1=Female
    race = rng.choice(
        [1, 2, 3, 4, 5], size=n,
        p=[0.52, 0.15, 0.23, 0.04, 0.06]                        # White, Black, Hispanic, Asian, Other
    )
    ses_quintile = rng.choice([1, 2, 3, 4, 5], size=n,          # SES composite quintile
                              p=[0.20, 0.20, 0.20, 0.20, 0.20])
    parent_edu = rng.choice(                                      # Highest parent education
        [1, 2, 3, 4], size=n,
        p=[0.15, 0.30, 0.35, 0.20]                              # <HS, HS/GED, Some college, BA+
    )
    special_ed_flag = rng.choice([0, 1], size=n, p=[0.87, 0.13]) # IEP / special ed services

    # ── Baseline cognitive measures (preschool / kindergarten entry) ─────────
    # Vocabulary composite (0–100 scale, ~N(50,15))
    vocab_baseline = np.clip(
        rng.normal(50, 15, n) - (3 - ses_quintile) * 2.5, 5, 100
    )

    # Phonological awareness (0–50 scale)
    phonological = np.clip(
        rng.normal(25, 8, n) + (ses_quintile - 3) * 1.5
        + (parent_edu - 2) * 2, 0, 50
    )

    # Working memory (0–100 scale)
    working_memory = np.clip(rng.normal(50, 12, n), 10, 100)

    # ── Teacher & classroom variables ────────────────────────────────────────
    teacher_experience_yrs = np.clip(rng.poisson(8, n), 0, 35)
    class_size = rng.integers(12, 30, n)
    school_type = rng.choice([0, 1], n, p=[0.75, 0.25])         # 0=Public, 1=Private

    # ── Outcome variables across time ────────────────────────────────────────
    # Reading IRT score — modeled with SES, vocab, phonological awareness as drivers
    def reading_score(wave_boost, noise_sd=8):
        base = (
            40
            + wave_boost
            + 0.25 * vocab_baseline
            + 0.40 * phonological
            + 0.10 * working_memory
            + (ses_quintile - 3) * 3.5
            + (parent_edu - 2) * 2.0
            - special_ed_flag * 8
            + sex * 2                                            # girls slightly higher reading
        )
        return np.clip(rng.normal(base, noise_sd), 0, 200)

    # Math IRT score — modeled similarly with different coefficients
    def math_score(wave_boost, noise_sd=9):
        base = (
            38
            + wave_boost
            + 0.15 * vocab_baseline
            + 0.20 * phonological
            + 0.30 * working_memory
            + (ses_quintile - 3) * 4.0
            + (parent_edu - 2) * 2.5
            - special_ed_flag * 7
            - sex * 1                                            # boys slightly higher math
        )
        return np.clip(rng.normal(base, noise_sd), 0, 200)

    df = pd.DataFrame({
        "child_id":              np.arange(1, n + 1),
        "sex":                   sex,
        "race":                  race,
        "ses_quintile":          ses_quintile,
        "parent_edu":            parent_edu,
        "special_ed_flag":       special_ed_flag,
        "vocab_baseline":        vocab_baseline.round(2),
        "phonological":          phonological.round(2),
        "working_memory":        working_memory.round(2),
        "teacher_experience_yrs": teacher_experience_yrs,
        "class_size":            class_size,
        "school_type":           school_type,
        # Reading scores: Kindergarten (W1) → Grade 4 (W5)
        "reading_w1":            reading_score(0),
        "reading_w2":            reading_score(15),
        "reading_w3":            reading_score(28),
        "reading_w4":            reading_score(39),
        "reading_w5":            reading_score(49),
        # Math scores
        "math_w1":               math_score(0),
        "math_w2":               math_score(14),
        "math_w3":               math_score(26),
        "math_w4":               math_score(37),
        "math_w5":               math_score(47),
    })

    # Round numeric columns
    for col in ["reading_w1","reading_w2","reading_w3","reading_w4","reading_w5",
                "math_w1","math_w2","math_w3","math_w4","math_w5"]:
        df[col] = df[col].round(1)

    if save:
        out = Path(__file__).parent.parent / "data" / "synthetic_ecls.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[generate_data] Saved {n} records → {out}")

    return df


if __name__ == "__main__":
    df = generate_ecls_synthetic(n=2000)
    print(df.describe())
