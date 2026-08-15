# Data

This project uses the **Elliptic Bitcoin Dataset** — a real, public graph of
203,769 Bitcoin transactions labelled licit / illicit, released by the
blockchain-analytics company Elliptic. It's a standard benchmark for
illicit-transaction detection in cryptocurrency.

## Download (free Kaggle account, one click)

https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Place these two files directly in this `data/` folder:

    data/elliptic_txs_features.csv
    data/elliptic_txs_classes.csv

(The dataset also ships `elliptic_txs_edgelist.csv` for graph-based methods —
this project uses the tabular features, so that file is optional.)

Then run, from the project root:

    python run.py

## About the labels

- ~2% of all transactions are labelled **illicit** (class 1)
- ~21% are labelled **licit** (class 2)
- the rest are **unknown** (unlabelled) and are dropped for this supervised task

Among the *labelled* transactions, roughly 10% are illicit — the class
imbalance this project is built to handle.
