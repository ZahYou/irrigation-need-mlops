"""Tests for src/data/features.py — add assertions once you paste feature code."""

import pandas as pd

from src.data.features import build_features


def test_build_features_returns_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)
