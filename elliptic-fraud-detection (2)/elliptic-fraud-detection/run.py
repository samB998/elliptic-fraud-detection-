"""
run.py
------
End-to-end illicit-transaction detection on the Elliptic Bitcoin dataset:

    load labelled transactions -> temporal train/test split
    -> train XGBoost 3 ways (no handling / cost-sensitive / SMOTE)
    -> compare on imbalance-aware metrics -> save table + figures

Usage:  python run.py
(Place the two Elliptic CSVs in ./data first — see data/README.md.)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load_data import load_elliptic, temporal_split          # noqa: E402
from model import train_baseline, train_cost_sensitive, train_with_smote  # noqa: E402
import evaluate as ev                                          # noqa: E402

REPORTS = "reports"


def main():
    os.makedirs(REPORTS, exist_ok=True)

    print("Loading Elliptic dataset ...")
    df = load_elliptic("data")
    illicit_rate = df["label"].mean()
    print(f"Labelled transactions: {len(df):,}  |  illicit: {illicit_rate:.1%}")

    X_train, y_train, X_test, y_test, feature_cols = temporal_split(df)
    print(f"Train: {len(y_train):,} rows  |  Test: {len(y_test):,} rows "
          f"(temporal split on time step)\n")

    print("Training models ...")
    models = {
        "XGBoost (no handling)":   train_baseline(X_train, y_train),
        "XGBoost (cost-sensitive)": train_cost_sensitive(X_train, y_train),
        "XGBoost (SMOTE)":          train_with_smote(X_train, y_train),
    }

    scores = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}
    metrics = {name: ev.evaluate(y_test, s) for name, s in scores.items()}

    print("\n===============  RESULTS (test set)  ===============")
    ev.print_table(metrics)

    ev.save_metrics(metrics, f"{REPORTS}/metrics.json")
    ev.plot_pr_curves(scores, y_test, f"{REPORTS}/pr_curves.png")
    ev.plot_confusion(y_test, scores["XGBoost (cost-sensitive)"],
                      f"{REPORTS}/confusion_cost_sensitive.png",
                      "XGBoost (cost-sensitive) @ 0.5")
    ev.plot_feature_importance(models["XGBoost (cost-sensitive)"], feature_cols,
                               f"{REPORTS}/feature_importance.png")

    print(f"\nSaved metrics -> {REPORTS}/metrics.json and figures -> {REPORTS}/")
    print("Done.")


if __name__ == "__main__":
    main()
