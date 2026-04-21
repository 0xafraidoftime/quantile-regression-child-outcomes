"""
test_quantile_analysis.py
--------------------------
Unit tests for the quantile regression pipeline.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_data import generate_ecls_synthetic
from quantile_analysis import fit_quantile_models, build_coef_table, build_summary_table, QUANTILES


@pytest.fixture(scope="module")
def sample_df():
    return generate_ecls_synthetic(n=300, save=False)


def test_data_shape(sample_df):
    assert sample_df.shape[0] == 300
    assert "reading_w5" in sample_df.columns
    assert "math_w5" in sample_df.columns


def test_data_ranges(sample_df):
    assert sample_df["ses_quintile"].between(1, 5).all()
    assert sample_df["special_ed_flag"].isin([0, 1]).all()
    assert sample_df["vocab_baseline"].between(0, 100).all()


def test_quantile_models_fit(sample_df):
    models = fit_quantile_models(sample_df, outcome="reading_w5")
    assert "ols" in models
    for tau in QUANTILES:
        assert tau in models


def test_coef_table_shape(sample_df):
    models = fit_quantile_models(sample_df, outcome="reading_w5")
    df = build_coef_table(models, predictor="ses_quintile")
    assert len(df) == len(QUANTILES) + 1  # QR models + OLS
    assert "coef" in df.columns


def test_summary_table(sample_df):
    models = fit_quantile_models(sample_df, outcome="reading_w5")
    table = build_summary_table(models, ["ses_quintile", "vocab_baseline"])
    assert "ses_quintile" in table.index
    assert "OLS" in table.columns


def test_no_null_outcomes(sample_df):
    for col in ["reading_w1", "reading_w5", "math_w1", "math_w5"]:
        assert sample_df[col].isna().sum() == 0
