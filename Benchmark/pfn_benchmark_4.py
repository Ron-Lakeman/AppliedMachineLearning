# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 13:52:23 2025

@author: marco


Simple TabPFN benchmark script for CoverType, HIGGS and HELOC.

- Train files: contain features + label column.
- Test files:
    contain only features
"""

from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import torch
import pandas as pd

def encode_label_column(df, label_col):
    """
    Turn the label column into integers.

    - CoverType: Cover_Type is already numeric.
    - HIGGS: Label is 's' (signal) or 'b' (background) -> map to 1 / 0.
    - HELOC: RiskPerformance is 'Good' / 'Bad' -> map to 1 / 0.
    """
    if label_col == "Label":
        # HIGGS (Kaggle-style): 's' = signal, 'b' = background
        y_str = df[label_col].astype(str)
        y = (y_str == "s").astype(int).values
    elif label_col == "RiskPerformance":
        # HELOC: 'Good' / 'Bad'
        y_str = df[label_col].astype(str)
        y = (y_str == "Good").astype(int).values
    else:
        # Default: assume labels are already numeric
        y = df[label_col].astype(int).values

    return y


def csv_to_xy(path, label_col, replace_with_nan=False):
    """
    Read a CSV file and return (X, y).

    path            : path to .csv file
    label_col       : name of the target (label) column
    replace_with_nan: if True, replace -9 with NaN in feature columns
                      (used for HELOC missing values).
    """
    df = pd.read_csv(path)

    if replace_with_nan:
        # All columns except the label column are features
        feature_cols = [c for c in df.columns if c != label_col]
        df[feature_cols] = df[feature_cols].replace(-9, np.nan)

    # y = encoded label column
    y = encode_label_column(df, label_col)
    # X = all other columns as float32
    X = df.drop(columns=[label_col]).values.astype(np.float32)

    return X, y


def train_model(X_train, y_train):
    """
    Train a TabPFN model on the given training data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)

    return clf


def evaluate_model(clf, X_test, y_test, n_classes):
    """
    Evaluate a trained model on test data using accuracy and ROC AUC.
    """
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    if n_classes == 2:
        # Binary classification: use probability of the positive class
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        # Multiclass classification
        auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="macro",
        )

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f} (n_classes={n_classes})")

    return acc, auc


def run_dataset_from_csv(train_csv, test_csv, label_col, replace_with_nan=False):
    """
    Full pipeline for one dataset:
    - load training data (must have labels)
    - load test data (may or may not have labels)
    - train model
    - evaluate model if test labels are present
    """
    print("=" * 80)
    print(f"Train file: {train_csv}")
    print(f"Test file : {test_csv}")
    print(f"Label col : {label_col}")

    # 1. Load training data (must have label column)
    X_train, y_train = csv_to_xy(
        train_csv,
        label_col=label_col,
        replace_with_nan=replace_with_nan,
    )

    # 2. Load test data
    df_test = pd.read_csv(test_csv)

    # Handle missing-value code for HELOC in the test file if needed
    if replace_with_nan:
        if label_col in df_test.columns:
            feature_cols_test = [c for c in df_test.columns if c != label_col]
        else:
            feature_cols_test = list(df_test.columns)
        df_test[feature_cols_test] = df_test[feature_cols_test].replace(-9, np.nan)

    # Check if the test file has a label column
    if label_col in df_test.columns:
        # Labeled test set (e.g. your own train/test split)
        y_test = df_test[label_col].astype(int).values
        X_test = df_test.drop(columns=[label_col]).values.astype(np.float32)
        test_has_label = True
    else:
        # Unlabeled test set (Kaggle-style)
        y_test = None
        X_test = df_test.values.astype(np.float32)
        test_has_label = False

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    if test_has_label:
        print(f"Test  shape: X={X_test.shape}, y={y_test.shape}")
    else:
        print(f"Test  shape: X={X_test.shape}, y=None (no label column found)")

    # 3. Train model
    n_classes = len(np.unique(y_train))
    clf = train_model(X_train, y_train)

    # 4. Evaluate model if labels are available for the test set
    if test_has_label:
        acc, auc = evaluate_model(clf, X_test, y_test, n_classes)
    else:
        print("Test file has no label column. Skipping accuracy and ROC AUC.")
        # Still run predictions so you can use them for submission if needed
        test_predictions = clf.predict(X_test)
        print("First 10 predictions on test data:", test_predictions[:10])
        acc = float("nan")
        auc = float("nan")

    return clf, acc, auc


if __name__ == "__main__":
    # Configuration for the three datasets
    datasets = [
        {
            "name": "Cover_Type",
            "train_csv": "covtype_train.csv",
            "test_csv": "covtype_test.csv",
            "label_col": "Cover_Type",     # Cover type label
            "replace_with_nan": False,
        },
        {
            "name": "HIGGS",
            "train_csv": "higgs_train.csv",
            "test_csv": "higgs_test.csv",
            "label_col": "Label",          # 1 for signal, 0 for background
            "replace_with_nan": False,
        },
        {
            "name": "HELOC",
            "train_csv": "heloc_train.csv",
            "test_csv": "heloc_test.csv",
            "label_col": "RiskPerformance",  # 'Good' / 'Bad' encoded as integers
            "replace_with_nan": True,        # handle -9 as missing
        },
    ]

    results = {}

    for cfg in datasets:
        print(f"\nRunning dataset: {cfg['name']}")
        clf, acc, auc = run_dataset_from_csv(
            train_csv=cfg["train_csv"],
            test_csv=cfg["test_csv"],
            label_col=cfg["label_col"],
            replace_with_nan=cfg["replace_with_nan"],
        )
        results[cfg["name"]] = {
            "model": clf,
            "accuracy": acc,
            "roc_auc": auc,
        }

    print("\nSummary of results:")
    for name, metrics in results.items():
        print(
            f"{name:10s} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"ROC AUC: {metrics['roc_auc']:.4f}"
        )
