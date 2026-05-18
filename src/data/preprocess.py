"""Raw data loading and cleaning — outputs a clean DataFrame for feature engineering."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV from *path* and return a DataFrame."""
    path = Path(path)
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning: drop duplicates, reset index.

    Extend this function with dataset-specific logic once you paste your
    Kaggle preprocessing code (Week 1, Day 2-3).
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)
    return df
