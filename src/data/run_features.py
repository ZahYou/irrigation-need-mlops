"""
Features stage entry-point for `dvc repro`.

Reads raw train/test CSVs, runs build_features() from feature_engineering.py,
and writes the processed artifacts to data/processed/ for the train stage.
"""

import json
import logging
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import build_feature_artifacts, build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    with open("configs/train_config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_train = cfg["data"]["raw_train"]
    raw_test  = cfg["data"]["raw_test"]
    out_dir   = Path(cfg["data"]["processed_dir"])
    target    = cfg["data"]["target_column"]
    label_map = cfg["data"]["label_map"]
    n_splits  = cfg["training"]["n_splits"]
    seed      = cfg["training"]["seed"]

    dev_sample      = cfg["data"].get("dev_sample")
    dev_test_sample = cfg["data"].get("dev_test_sample")

    log.info("Loading raw data — train: %s  test: %s", raw_train, raw_test)
    train = pd.read_csv(raw_train)
    test  = pd.read_csv(raw_test)

    if dev_sample is not None and dev_sample < len(train):
        train, _ = train_test_split(
            train, train_size=dev_sample, stratify=train[target], random_state=seed
        )
        train = train.reset_index(drop=True)
        log.info("Dev mode: sampled %d train rows (stratified)", dev_sample)

    if dev_test_sample is not None and dev_test_sample < len(test):
        test = test.sample(n=dev_test_sample, random_state=seed).reset_index(drop=True)
        log.info("Dev mode: sampled %d test rows", dev_test_sample)

    y_train = train[target].map(label_map)

    log.info("Running build_features()  n_splits=%d  seed=%d", n_splits, seed)
    X_train, X_test, cat_cols, num_cols = build_features(
        train,
        test,
        y=y_train,
        n_splits=n_splits,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_pickle(out_dir / "X_train.pkl")
    X_test.to_pickle(out_dir  / "X_test.pkl")
    y_train.to_pickle(out_dir / "y_train.pkl")

    with open(out_dir / "cat_cols.json", "w") as f:
        json.dump(cat_cols, f)
    with open(out_dir / "num_cols.json", "w") as f:
        json.dump(num_cols, f)

    artifacts = build_feature_artifacts(X_train, y_train)
    with open(out_dir / "feature_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    log.info(
        "Saved to %s — X_train %s  X_test %s  cats %d  nums %d",
        out_dir, X_train.shape, X_test.shape, len(cat_cols), len(num_cols),
    )


if __name__ == "__main__":
    main()
