# Illicit Bitcoin Transaction Detection (Elliptic Dataset)

Supervised detection of **illicit cryptocurrency transactions** on the
[Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) —
a real, public graph of 203,769 Bitcoin transactions labelled licit / illicit,
released by the blockchain-analytics firm Elliptic. This is a standard benchmark
for financial-crime / AML detection on-chain.

The project focuses on the part that makes fraud detection *hard in practice*:
**severe class imbalance** (illicit transactions are rare), and it treats that
as the central problem rather than an afterthought.

## What it does

- Loads the real Elliptic transactions and frames a binary task: **illicit (1)**
  vs **licit (0)**, keeping only labelled nodes.
- Splits **by time step** (train on earlier blocks, test on later ones) — the
  standard Elliptic protocol, which mirrors deployment and avoids leaking future
  transactions into training.
- Trains **XGBoost** three ways to show the impact of imbalance handling:
  1. no handling (baseline — what *not* to do),
  2. **cost-sensitive** learning via `scale_pos_weight`,
  3. **SMOTE** oversampling of the minority class (training fold only).
- Evaluates with metrics that are meaningful when positives are rare:
  **PR-AUC**, ROC-AUC, illicit-class precision/recall/F1, and
  **precision @ top-k%** (how clean the top of an analyst's alert queue is).

## Why these choices

- **XGBoost** — gradient-boosted trees are the strong default for tabular,
  imbalanced financial data; fast, robust, and interpretable via feature
  importance. (LightGBM would be near-identical, so it's intentionally omitted.)
- **Imbalance handling is the point.** A model that predicts "always licit"
  is ~90% accurate and catches zero fraud. Comparing *no handling* vs
  *cost-sensitive* vs *SMOTE* is the core result — it shows why accuracy is the
  wrong metric and what actually moves recall on the rare class.
- **Temporal split, not random.** Random splitting lets the model peek at future
  transactions and inflates scores. Splitting by time step is honest.
- **PR-AUC over accuracy/ROC.** With ~10% positives in the labelled set, the
  precision-recall trade-off is what a real detection team optimises.

## Project structure

```
elliptic-fraud-detection/
├── run.py                 # end-to-end pipeline
├── requirements.txt
├── data/                  # download the Elliptic CSVs here (see data/README.md)
├── reports/               # metrics.json + figures (generated)
└── src/
    ├── load_data.py       # load Elliptic, binary labels, temporal split
    ├── model.py           # XGBoost + cost-sensitive + SMOTE
    └── evaluate.py        # PR-AUC, precision@top-k, plots
```

## Run it

```bash
pip install -r requirements.txt
# download the two Elliptic CSVs into data/  (see data/README.md)
python run.py
```

Output: a model-comparison table in the console, plus `reports/metrics.json`,
`reports/pr_curves.png`, a confusion matrix, and a feature-importance chart.

## Results

Running `run.py` prints a comparison table like:

```
model                       pr_auc   roc_auc  precision   recall       f1  precision_at_top_5pct
XGBoost (no handling)        ...       ...       ...        ...        ...        ...
XGBoost (cost-sensitive)     ...       ...       ...        ...        ...        ...
XGBoost (SMOTE)              ...       ...       ...        ...        ...        ...
```

The headline finding is the **gap between the no-handling baseline and the
imbalance-aware models** on illicit-class recall and PR-AUC. For reference,
published tree-model baselines on Elliptic reach an illicit-class F1 in the
high-0.7s with the temporal split (Weber et al., 2019); your exact numbers will
depend on the run.

## Notes / possible extensions

- The Elliptic features are anonymised and **pre-computed** by the dataset
  authors (94 local + 71 aggregated-from-neighbours), so this project does no
  hand feature engineering — the emphasis is supervised modelling under
  imbalance. A natural extension is to add **graph features** from the provided
  edge list (e.g. distance to known-illicit nodes) or a GNN, which is where the
  research frontier on this dataset sits.

## Data

Not included in the repo (download separately — free Kaggle account).
See [`data/README.md`](data/README.md).
