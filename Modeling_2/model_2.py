# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:34:34 2025

@author: marco

XGBoost benchmark script for CoverType, HIGGS and HELOC.

- Train files: contain features + label column.
- Test files:  contain only features (Kaggle test set).

GPU usage:
- If a CUDA GPU is available, the script will try to train/predict on GPU.
"""

import numpy as np
import torch
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid


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
    """
    feature_cols = [c for c in df.columns if c != label_col]
    
    if replace_heloc:
        # If missing rows can't be handled by the submission just deactivate this line of code
        # df = df[~(df[feature_cols] == -9).all(axis=1)]
        
        df[feature_cols] = df[feature_cols].replace({-9: np.nan, -8: np.nan, -7: 0})

    # HIGGS convention: -999 means missing in features
    if label_col == "Label" or (label_col is None and -999 in df[feature_cols].values):
        df[feature_cols] = df[feature_cols].replace(-999, np.nan)

    return df


def load_train_xy(path, label_col, replace_heloc=False):
    """
    Read a CSV file and return (X, y).

    path          : path to .csv file
    label_col     : name of the target (label) column
    replace_heloc : if True, apply HELOC special-value cleaning
    """
    df = pd.read_csv(path)
    clean_features(df, label_col=label_col, replace_heloc=replace_heloc)

    y, y_le = encode_label_column(df, label_col)
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    return X, y, y_le


def load_test_x(path, label_col, replace_heloc=False):
    df = pd.read_csv(path)
    df = clean_features(df, label_col=None, replace_heloc=replace_heloc)
    X = df.to_numpy(dtype=np.float32)
    return X

def subsample_training(X_train, y_train, max_train_samples=None):
    """
    Subsample the training data to at most max_train_samples.

    This is used for large datasets (like HIGGS) to reduce training time and
    avoid CUDA out-of-memory on GPUs with limited memory.
    """
    if max_train_samples is None:
        return X_train, y_train

    n_samples = y_train.shape[0]
    if n_samples <= max_train_samples:
        return X_train, y_train

    print(f"Training set has {n_samples} samples, subsampling to {max_train_samples}.")

    rng = np.random.RandomState(0)
    indices = rng.choice(n_samples, size=max_train_samples, replace=False)
    return X_train[indices], y_train[indices]


# possibly eventually replace with sci-kit inbuilt function
def compute_balanced_sample_weight(y):
    """
    Simple class balancing: weight each class inversely to its frequency,
    normalized to mean weight 1.
    """
    classes, counts = np.unique(y, return_counts=True)
    inv = 1.0 / counts.astype(np.float64)
    weight_map = dict(zip(classes, inv))
    w = np.array([weight_map[int(yi)] for yi in y], dtype=np.float64)
    w *= (len(w) / w.sum())
    return w.astype(np.float32)

def xgb_grid():
    return {
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "n_estimators": [300, 600],
    }

def train_model(X_train, y_train, n_classes, balance_probabilities=False, param_overrides=None):
    """
    Train an XGBoost model on the given training data (GPU if available).
    """
    use_gpu = torch.cuda.is_available()
    print(f"Using device: {'cuda' if use_gpu else 'cpu'}")

    sample_weight = compute_balanced_sample_weight(y_train) if balance_probabilities else None

    # Basic hyperparameters (kept simple on purpose)
    common_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0, # what exactly does the lambda regularizer do?
        random_state=0,
        n_jobs=-1,
        missing=np.nan, # default is NaN, but explicit is clearer
        verbosity=1,
        tree_method="hist", 
        device="cuda"
    )
    
    
    # Allow grid search to override parameters
    if param_overrides is not None:
        common_params.update(param_overrides)


    if n_classes == 2:
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            #eval_metric="logloss",
            #eval_metric="error",
            **common_params,
        )
    else:
        clf = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            #eval_metric="merror",
            num_class=n_classes,
            **common_params,
        )

    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf

def tune_with_grid(X_tr, y_tr, X_val, y_val, n_classes, balance_probabilities, param_grid):
    best_params = None
    best_auc = -np.inf
    best_acc = -np.inf

    for params in ParameterGrid(param_grid):
        clf = train_model(
            X_tr, y_tr,
            n_classes=n_classes,
            balance_probabilities=balance_probabilities,
            param_overrides=params,
        )

        acc, auc = evaluate_model(clf, X_val, y_val, n_classes)
        
        # Does it make more sense to use auc or acc as the deciding evaluation parameter?
        if (auc > best_auc) or (auc == best_auc and acc > best_acc):
            best_auc, best_acc = auc, acc
            best_params = params

    return best_params, best_acc, best_auc



def batched_call(fn, X, batch_size=2500):
    """
    Run clf.predict on X in small batches to avoid out-of-memory.
    """
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        chunks.append(fn(X[start:end]))
    return np.concatenate(chunks, axis=0)


def evaluate_model(clf, X_test, y_test, n_classes, batch_size=2500):
    """
    Evaluate a trained model on test data using accuracy and ROC AUC.
    """
    y_pred = batched_call(clf.predict, X_test, batch_size=batch_size)
    y_proba = batched_call(clf.predict_proba, X_test, batch_size=batch_size)

    acc = accuracy_score(y_test, y_pred)

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
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
    - Subsample training data (optional)
    - Split into train/validation
    - Train a model and evaluate on validation
    - Load Kaggle-style test data and predict (for submission)
    """
    print("=" * 75)
    print(f"Train file: {train_csv}")
    print(f"Test file : {test_csv}")
    print(f"Label col : {label_col}")

    # 1. Load full training data
    X_train_full, y_train_full, y_le = load_train_xy(
        train_csv,
        label_col=label_col,
        replace_heloc=replace_heloc,
    )

    print(f"Original train shape: X={X_train_full.shape}, y={y_train_full.shape}")

    # 2. Optional subsampling
    X_train, y_train = subsample_training(
        X_train_full, y_train_full, max_train_samples=max_train_samples
    )

    print(f"Used train shape   : X={X_train.shape}, y={y_train.shape}")

    # 3. Train/validation split
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
    
    # 4. Grid search
    param_grid = xgb_grid()

    best_params, best_acc, best_auc = tune_with_grid(
        X_tr, y_tr, X_val, y_val,
        n_classes=n_classes,
        balance_probabilities=balance_probabilities,
        param_grid=param_grid,
    )

    print("Best params:", best_params)
    print(f"Best val AUC: {best_auc:.4f}, Best val Acc: {best_acc:.4f}")
    
    # 5. Train and validate
    # retrain final model on full labeled data (X_train, y_train) with best params
    clf = train_model(
        X_train, y_train,
        n_classes=n_classes,
        balance_probabilities=balance_probabilities,
        param_overrides=best_params,
    )

    # 5. Load test data (no labels)
    X_test = load_test_x(test_csv, label_col=label_col, replace_heloc=replace_heloc)

    # 6. Predict and decode to original labels
    test_predictions_enc = batched_call(clf.predict, X_test, batch_size=2500)
    test_predictions = y_le.inverse_transform(test_predictions_enc.astype(int))

    return clf, best_acc, best_auc, test_predictions




if __name__ == "__main__":
    # Configuration for the three datasets
    datasets = [
        {
            "name": "Cover_Type",
            "train_csv": "covtype_simple_train.csv",
            "test_csv": "covtype_simple_test.csv",
            "label_col": "Cover_Type",
            "replace_heloc": False,
            "max_train_samples": None,
            "balance_probabilities": True,
        },
        {
            "name": "HELOC",
            "train_csv": "heloc_train.csv",
            "test_csv": "heloc_test.csv",
            "label_col": "RiskPerformance",
            "replace_heloc": True,
            "max_train_samples": None,
            "balance_probabilities": True,
        },
        {
            "name": "HIGGS",
            "train_csv": "higgs_train.csv",
            "test_csv": "higgs_test.csv",
            "label_col": "Label",
            "replace_heloc": False,
            "max_train_samples": 50000,
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

    submission_template_path = "combined_test_sample_submission.csv"
    submission_output_path = "combined_submission.csv"

    sub = pd.read_csv(submission_template_path)

    prediction_column = "Prediction"

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
        sub[prediction_column] = all_predictions
        sub.to_csv(submission_output_path, index=False)
        print(f"\nSaved combined Kaggle submission to: {submission_output_path}")

# np.savetxt('higgs_pred.csv', results["HIGGS"]["test_predictions"])
