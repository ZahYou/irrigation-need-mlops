"""Tests for src/data/preprocess.py — expanded in Week 1 Day 2-3 with real data."""

import pandas as pd
import pytest

from src.data.preprocess import clean


def test_clean_drops_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    result = clean(df)
    assert len(result) == 2


def test_clean_resets_index() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[5, 10, 15])
    result = clean(df)
    assert list(result.index) == [0, 1, 2]
