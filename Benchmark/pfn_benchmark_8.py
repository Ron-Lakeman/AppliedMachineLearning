# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:52:23 2025

@author: marco

TabPFN benchmark script for CoverType, HIGGS and HELOC.

- Train files: contain features + label column.
- Test files:
    contain only features (Kaggle test set).
"""

from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
#from sklearn.impute import SimpleImputer
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

def clean_heloc_df(df, label_col=None):
    """
    Apply the HELOC special-value cleaning to a DataFrame.

    - If label_col is given and present, keep all other columns as features.
    - If drop_all_minus9 is True, drop rows where all features are -9.
    - Replace -9 and -8 with NaN, and -7 with 0 in feature columns.
    """

    feature_cols = [c for c in df.columns if c != label_col]
    # train-only row dropping if requested
    
    # df = df[~(df[feature_cols] == -9).all(axis=1)] # If missing rows can't be handled by the submission just deactivate this line of code

    # same replacements for train and test
    df[feature_cols] = df[feature_cols].replace(-9, np.nan)
    df[feature_cols] = df[feature_cols].replace(-7, 0)
    df[feature_cols] = df[feature_cols].replace(-8, np.nan)

    return df


def csv_to_xy(path, label_col, replace_heloc=False):
    """
    Read a CSV file and return (X, y).

    path            : path to .csv file
    label_col       : name of the target (label) column
    replace_heloc: if True, replace missing numerical values with NaN in feature columns
    """
    df = pd.read_csv(path)

    if replace_heloc:
        df = clean_heloc_df(df, label_col=label_col)
        
        
     # Handle HIGGS missing values: -999 -> NaN (features only)
    if label_col == "Label":
        feature_cols = [c for c in df.columns if c != label_col]
        df[feature_cols] = df[feature_cols].replace(-999, np.nan)
        
    # y = encoded label column
    y = encode_label_column(df, label_col)
    # X = all other columns as float32
    X = df.drop(columns=[label_col]).values.astype(np.float32)

    return X, y


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

def predict_in_batches(clf, X, batch_size=2500):
    """
    Run clf.predict on X in small batches to avoid CUDA out-of-memory.
    """
    preds = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        preds.append(clf.predict(X[start:end]))
    return np.concatenate(preds, axis=0)

def predict_proba_in_batches(clf, X, batch_size=2500):
    """
    Run clf.predict_proba on X in small batches.
    """
    probs = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        probs.append(clf.predict_proba(X[start:end]))
    return np.concatenate(probs, axis=0)


def evaluate_model(clf, X_test, y_test, n_classes, batch_size=2500):
    """
    Evaluate a trained model on test data using accuracy and ROC AUC.
    """
    y_pred  = predict_in_batches(clf, X_test, batch_size=batch_size)
    y_proba = predict_proba_in_batches(clf, X_test, batch_size=batch_size)

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
    - Subsample training data (optional, for large datasets)
    - Split into train/validation
    - Train a model on the train split, evaluate on validation split
    - Train a final model on the full training data
    - Load test data (Kaggle-style, no labels)
    - Predict on test data (for submission)
    """
    print("=" * 80)
    print(f"Train file: {train_csv}")
    print(f"Test file : {test_csv}")
    print(f"Label col : {label_col}")

    # 1. Load full training data (must have label column)
    X_train_full, y_train_full = csv_to_xy(
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
        random_state=0,
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
    
    
    
    df_test = pd.read_csv(test_csv)
    
    if replace_heloc:
        df_test = clean_heloc_df(df_test, label_col=label_col)

    # Handle HIGGS missing-value code: -999 -> NaN (features only)
    if label_col == "Label":
        feature_cols_test = list(df_test.columns)
        df_test[feature_cols_test] = df_test[feature_cols_test].replace(-999, np.nan)

    
    X_test = df_test.values.astype(np.float32)
    
    acc, auc = val_acc, val_auc
    
    # 7. Compute predictions on test data for submission
    test_predictions = predict_in_batches(clf, X_test, batch_size = 2500)
    
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
