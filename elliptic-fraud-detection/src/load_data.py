"""
load_data.py
------------
Load the Elliptic Bitcoin dataset (real, public blockchain fraud data).

The dataset ships as two CSVs:
  - elliptic_txs_features.csv : NO header. col 0 = txId, col 1 = time step (1-49),
                                cols 2-166 = 165 anonymised features
                                (94 local + 71 aggregated from graph neighbours).
  - elliptic_txs_classes.csv  : header "txId,class". class is
                                "1" = illicit, "2" = licit, "unknown" = unlabelled.

We keep only the labelled transactions and frame it as binary classification:
illicit (1) vs licit (0). About 10% of labelled nodes are illicit -> the class
imbalance that makes this a realistic detection problem.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

N_FEATURES = 165  # 94 local + 71 aggregated


def _feature_columns() -> list[str]:
    """Column names for the header-less features file."""
    cols = ["txId", "time_step"]
    cols += [f"local_{i}" for i in range(1, 94)]       # 93 local (feat 2 is time)
    cols += [f"agg_{i}" for i in range(1, 73)]         # 72 aggregated
    return cols


def load_elliptic(data_dir: str = "data"):
    """
    Returns a single DataFrame of labelled transactions with columns:
    ['txId', 'time_step', <165 features>, 'label'] where label is 1=illicit, 0=licit.
    """
    feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")

    if not (os.path.exists(feat_path) and os.path.exists(class_path)):
        raise FileNotFoundError(
            "Elliptic CSVs not found in '%s'.\n"
            "Download them (free Kaggle account) from:\n"
            "  https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
            "and place elliptic_txs_features.csv and elliptic_txs_classes.csv there."
            % data_dir
        )

    features = pd.read_csv(feat_path, header=None)
    features.columns = _feature_columns()

    classes = pd.read_csv(class_path)                 # has header: txId, class
    df = features.merge(classes, on="txId", how="left")

    # Keep only labelled rows; map to a clean binary target.
    df = df[df["class"] != "unknown"].copy()
    df["label"] = (df["class"] == "1").astype(int)    # 1 = illicit
    df = df.drop(columns=["class"])
    return df


def temporal_split(df: pd.DataFrame, split_step: int = 34):
    """
    Split by time step (the standard Elliptic protocol): train on the earlier
    time steps, test on the later ones. This mimics deployment — you train on the
    past and score the future — and avoids leaking future patterns into training.
    """
    feature_cols = [c for c in df.columns
                    if c not in ("txId", "time_step", "label")]
    train = df[df["time_step"] <= split_step]
    test = df[df["time_step"] > split_step]

    X_train = train[feature_cols].to_numpy()
    y_train = train["label"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["label"].to_numpy()
    return X_train, y_train, X_test, y_test, feature_cols
