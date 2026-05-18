"""Feature engineering — takes a clean DataFrame, returns a feature-ready DataFrame."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return the augmented DataFrame.

    Paste your Kaggle feature engineering logic here during Week 1, Day 2-3.
    Keep each logical group in its own private helper (_encode_categoricals,
    _add_interactions, etc.) and call them from this function.
    """
    logger.info("Building features for %d rows", len(df))
    # --- placeholder: add your transformations below ---
    return df
