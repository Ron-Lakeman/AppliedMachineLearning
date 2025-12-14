# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:52:23 2025

@author: marco

TabPFN benchmark script for CoverType, HIGGS and HELOC.

- Train files: contain features + label column.
- Test files:
    contain only features (Kaggle test set).
"""
import os
import random
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.impute import SimpleImputer
import numpy as np
import torch
import pandas as pd

SEED = 42

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
set_seed(SEED)

def encode_label_column(df, label_col):
    """
    Turn the label column into integers.

    - CoverType: Cover_Type is already numeric (often 1..7).
    - HIGGS: Label is 's' (signal) or 'b' (background) -> map to 1 / 0.
    - HELOC: RiskPerformance is 'Good' / 'Bad' -> map to 1 / 0.
    """
    if label_col == "Label":
        y_raw = (df[label_col].astype(str) == "s").astype(int).values
    elif label_col == "RiskPerformance":
        y_raw = (df[label_col].astype(str) == "Good").astype(int).values
    else:
        y_raw = df[label_col].astype(int).values
        
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    return y_encoded, le


def clean_features(df, label_col=None, replace_heloc=False):
    """
    Apply the HELOC special-value cleaning to a DataFrame.

    Replacements in feature columns:
      -9 -> NaN
      -8 -> NaN
      -7 -> 0 --> absence of delinquency is good
      
    Higgs:
        drop 'EventID column'
        -999 -> NaN
    """
    
    feature_cols = [c for c in df.columns if c != label_col]
    
    if replace_heloc:
        # If missing rows can't be handled by the submission just deactivate this line of code
        # df = df[~(df[feature_cols] == -9).all(axis=1)]
        
        df[feature_cols] = df[feature_cols].replace({-9: np.nan, -8: np.nan, -7: 0})
    # HIGGS convention: -999 means missing in features
    if label_col == "Label":
        df[feature_cols] = df[feature_cols].replace(-999, np.nan)
        df = df.drop(columns=["EventId", "Weight"], errors="ignore")

    if label_col == "Cover_Type":
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    
    return df


def load_train_xy(path, label_col, replace_heloc=False):
    """
    Read a CSV file and return (X, y).

    path          : path to .csv file
    label_col     : name of the target (label) column
    replace_heloc : if True, apply HELOC special-value cleaning
    """
    df = pd.read_csv(path)
    df = clean_features(df, label_col=label_col, replace_heloc=replace_heloc)

    y, y_le = encode_label_column(df, label_col)
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    return X, y, y_le


def load_test_x(path, label_col, replace_heloc=False):
    df = pd.read_csv(path)
    df = clean_features(df, label_col=label_col, replace_heloc=replace_heloc)
    X = df.to_numpy(dtype=np.float32)
    return X


def subsample_training(X_train, y_train, max_train_samples=None):
    """
    Subsample the training data to at most max_train_samples.

    This is used for large datasets (like HIGGS) to avoid CUDA out-of-memory
    on GPUs with limited memory (e.g. 8 GB).
    """
    if max_train_samples is None:
        return X_train, y_train

    n_samples = y_train.shape[0]

    if n_samples <= max_train_samples:
        return X_train, y_train

    print(
        f"Training set has {n_samples} samples, "
        f"subsampling to {max_train_samples} for TabPFN."
    )

    rng = np.random.RandomState(0)  # reproducible
    indices = rng.choice(n_samples, size=max_train_samples, replace=False)

    X_sub = X_train[indices]
    y_sub = y_train[indices]

    return X_sub, y_sub


def train_model(X_train, y_train, balance_probabilities=False):
    """
    Train a TabPFN model on the given training data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clf = TabPFNClassifier(
        device=device,
        ignore_pretraining_limits=True,
        balance_probabilities=balance_probabilities,
    )
    clf.fit(X_train, y_train)

    return clf

def batched_call(fn, X, batch_size=2500):
    """
    Run clf.predict on X in small batches to avoid out-of-memory.
    """
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        chunks.append(fn(X[start:end]))
    return np.concatenate(chunks, axis=0)


def evaluate_model(clf, X, y_true, n_classes, batch_size=2500):
    """
    Evaluate a trained model on test data using accuracy and ROC AUC.
    """
    
    y_pred = batched_call(clf.predict, X, batch_size=batch_size)
    y_proba = batched_call(clf.predict_proba, X, batch_size=batch_size)
    
    acc = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        # Binary classification: use probability of the positive class
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        # Multiclass classification
        auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
        )

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f} (n_classes={n_classes})")

    return acc, auc

def run_dataset_from_csv(
    train_csv,
    test_csv,
    label_col,
    replace_heloc=False,
    max_train_samples=None,
    balance_probabilities=False,
):
    """
    Full pipeline for one dataset:
    - Load training data (must have labels)
    - Subsample training data (optional)
    - Split into train/validation
    - Train a model and evaluate on validation
    - Load Kaggle-style test data and predict (for submission)
    """
    print("=" * 75)
    print(f"Train file: {train_csv}")
    print(f"Test file : {test_csv}")
    print(f"Label col : {label_col}")

    # 1. Load full training data (must have label column)
    X_train_full, y_train_full, y_le = load_train_xy(
        train_csv,
        label_col=label_col,
        replace_heloc=replace_heloc,
    )

    print(f"Original train shape: X={X_train_full.shape}, y={y_train_full.shape}")

    # 2. Optionally subsample training data (for HIGGS)
    X_train, y_train = subsample_training(
        X_train_full, y_train_full, max_train_samples=max_train_samples
    )

    print(f"Used train shape   : X={X_train.shape}, y={y_train.shape}")

    # 3. Make a train/validation split on the (possibly subsampled) training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train,
    )

    print(f"Train split shape  : X={X_tr.shape}, y={y_tr.shape}")
    print(f"Val   split shape  : X={X_val.shape}, y={y_val.shape}")

    
    n_classes = len(np.unique(y_train))

    # 4. Train on train split and evaluate on validation split
    print("Training model on train split for validation metrics...")
    clf = train_model(
        X_tr,
        y_tr,
        balance_probabilities=balance_probabilities,
    )

    print("Validation performance:")
    val_acc, val_auc = evaluate_model(clf, X_val, y_val, n_classes)
    
    
    acc, auc = val_acc, val_auc
    
    # 5. Load test data (no labels)
    X_test = load_test_x(test_csv, label_col=label_col, replace_heloc=replace_heloc)

    # 6. Predict and decode to original labels
    test_predictions_enc = batched_call(clf.predict, X_test, batch_size=2500)
    test_predictions = y_le.inverse_transform(test_predictions_enc.astype(int))
    
    return clf, acc, auc, test_predictions
    
if __name__ == "__main__":
    # Configuration for the three datasets
    datasets = [
        {
            "name": "Cover_Type",
            "train_csv": "covtype_simple_train.csv",
            "test_csv": "covtype_simple_test.csv",
            "label_col": "Cover_Type",     # Cover type label
            "replace_heloc": False,
            "max_train_samples": 50000,
            "balance_probabilities": True,
        },
        {
            "name": "HELOC",
            "train_csv": "heloc_train.csv",
            "test_csv": "heloc_test.csv",
            "label_col": "RiskPerformance",  # 'Good' / 'Bad'
            "replace_heloc": True,        
            "max_train_samples": None,
            "balance_probabilities": True,
        },
        
        {
            "name": "HIGGS",
            "train_csv": "higgs_train.csv",
            "test_csv": "higgs_test.csv",
            "label_col": "Label",          # 's' / 'b' for HIGGS
            "replace_heloc": False,     # -999 handled by label_col == "Label"
            "max_train_samples": 50000,    # subsample to avoid OOM
            "balance_probabilities": False,
        },
    ]

    results = {}

    for cfg in datasets:
        print(f"\nRunning dataset: {cfg['name']}")
        clf, acc, auc, test_predictions = run_dataset_from_csv(
            train_csv=cfg["train_csv"],
            test_csv=cfg["test_csv"],
            label_col=cfg["label_col"],
            replace_heloc=cfg["replace_heloc"],
            max_train_samples=cfg["max_train_samples"],
            balance_probabilities=cfg["balance_probabilities"],
        )
        results[cfg["name"]] = {
            "model": clf,
            "accuracy": acc,
            "roc_auc": auc,
            "test_predictions": test_predictions,
        }

    print("\nSummary of results (using validation metrics when test has no labels):")
    for name, metrics in results.items():
        print(
            f"{name:10s} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"ROC AUC: {metrics['roc_auc']:.4f}"
        )

    # ---------------------------------------------------------------------
    # Create Kaggle-style combined submission file
    # ---------------------------------------------------------------------

    # Load the template sample submission
    submission_template_path = "combined_test_sample_submission.csv"
    submission_output_path = "combined_submission.csv"

    sub = pd.read_csv(submission_template_path)

    # Change "Prediction" here if your combined_test_sample_submission.csv
    # uses a different column name for the prediction.
    prediction_column = "Prediction"

    # Build one long array of predictions in the order:
    # Cover_Type test, then HELOC test, HIGGS test .
    preds_cov = results["Cover_Type"]["test_predictions"]
    preds_heloc = results["HELOC"]["test_predictions"]
    preds_higgs = results["HIGGS"]["test_predictions"]

    all_predictions = np.concatenate([preds_cov, preds_heloc, preds_higgs])

    if len(all_predictions) != len(sub):
        print(
            "WARNING: Number of predictions does not match number of rows in",
            "combined_test_sample_submission.csv",
        )
        print("Predictions:", len(all_predictions), " rows, template:", len(sub), " rows")
    else:
        # Fill the prediction column in the template with our predictions
        sub[prediction_column] = all_predictions

        # Save to a new CSV file (Kaggle-style)
        sub.to_csv(submission_output_path, index=False)
        print(f"\nSaved combined Kaggle submission to: {submission_output_path}")

#np.savetxt('higgs_pred.csv',test_predictions)
