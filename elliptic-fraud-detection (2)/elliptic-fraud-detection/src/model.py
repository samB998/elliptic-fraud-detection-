"""
model.py
--------
Supervised illicit-transaction detection with XGBoost, plus the two standard
ways of handling the class imbalance that defines fraud detection.

Fraud is rare, so a model that predicts "always licit" scores high on accuracy
and catches nothing. Two remedies, compared in this project:

  1. Cost-sensitive learning : `scale_pos_weight` tells XGBoost that missing an
     illicit transaction is far more costly than a false alarm. One parameter,
     no data distortion, leak-free.

  2. SMOTE resampling        : synthesise extra illicit examples so the training
     set is less skewed. Fit on TRAINING data only, never the test set.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def _scale_pos_weight(y) -> float:
    """Negatives-to-positives ratio — XGBoost's cost-sensitive knob."""
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    return int((y == 0).sum()) / pos


def train_cost_sensitive(X_train, y_train, seed: int = 42) -> XGBClassifier:
    """XGBoost that up-weights the rare illicit class via scale_pos_weight."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=_scale_pos_weight(y_train),
        eval_metric="aucpr",         # optimise precision-recall area (imbalanced)
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_with_smote(X_train, y_train, sampling_strategy: float = 0.3,
                     seed: int = 42) -> XGBClassifier:
    """
    XGBoost on a SMOTE-rebalanced training set (illicit grown to 30% of licit).
    We deliberately don't balance 50/50 — over-synthesising the minority class
    tends to flood the model with false positives.
    """
    import numpy as np
    y = np.asarray(y_train)
    current_ratio = (y == 1).sum() / max((y == 0).sum(), 1)
    if current_ratio >= sampling_strategy * 0.95:
        # Minority already at/above the target share — nothing to oversample.
        X_res, y_res = X_train, y_train
    else:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="aucpr", n_jobs=-1, random_state=seed,
    )
    model.fit(X_res, y_res)
    return model


def train_baseline(X_train, y_train, seed: int = 42) -> XGBClassifier:
    """Plain XGBoost with NO imbalance handling — the 'what not to do' baseline."""
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="aucpr", n_jobs=-1, random_state=seed,
    )
    model.fit(X_train, y_train)
    return model
