# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:17:45 2025

@author: marco

# 1. Clean features
# 2. Load three datasets into one unified dataframe [242514,67] --> fill non-applicable columns with Nan values
# --> covtype.shape [58101, 13] --> 12 features, 1 label
# --> Heloc.shape [9413, 23] --> 23 features, 1 label
# --> Higgs.shape [175000, 32] --> 30 features, 1 label, 1 weight (needs to be excluded --> data leakage)

# 3. Encode labels to 0-10 --> Do I need a separate column that acts as a dataset index (0,1,2)? I don't think I do,
# but I'll keep it in mind --> more robust with dataset index
# 4. Optional subsampling --> Let's see if it works without first
# 5. Train/validation split
# 6. Random Search hyperparameter optimization on f1 as evaluation metric, because heavily imbalanced dataset
# --> We don't want to have false positives when giving out loans, nor when deciding, if we've just found a new
# elementary particle --> Accuracy is not the best eval metric
# 7. Retrain model on best params
# 8. Predict labels
# 9. Create submission file
"""


import os
import random
import numpy as np
import torch
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

SEED = 42

EXCLUDE_COLS = {"Unnamed: 0", "EventId", "Weight"}  # drop 


COV_TRAIN = "covtype_simple_train.csv"
HELOC_TRAIN = "heloc_train.csv"
HIGGS_TRAIN = "higgs_train.csv"

COV_TEST = "covtype_simple_test.csv"
HELOC_TEST = "heloc_test.csv"
HIGGS_TEST = "higgs_test.csv"

DATASET_ONEHOT_COL = {"covtype": "is_covtype", "heloc": "is_heloc", "higgs": "is_higgs"}
DATASET_ONEHOT_COLS = list(DATASET_ONEHOT_COL.values())


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
set_seed(SEED)


def add_dataset_onehot(out: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    # initialize all to 0
    for c in DATASET_ONEHOT_COLS:
        out[c] = 0
    # set the matching dataset flag to 1
    out[DATASET_ONEHOT_COL[dataset_name]] = 1
    return out


def load_dataset(path, dataset_name, label_col):
    """
    Load and clean a labeled dataset.
    """
    df = pd.read_csv(path)
    drop_cols = set([label_col]) | (EXCLUDE_COLS & set(df.columns))
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # clean ONLY the features
    df[feature_cols] = df[feature_cols].replace({-9: np.nan, -8: np.nan, -999: np.nan, -7: 0})

    out = df[feature_cols].copy()
    out["dataset"] = dataset_name
    out = add_dataset_onehot(out, dataset_name)
    out["label_raw"] = df[label_col].astype(str)  # keep original label as string
    return out


def load_testset_x(path, dataset_name):
    """
    Load and clean a test dataset (no labels).
    """
    df = pd.read_csv(path)
    drop_cols = (EXCLUDE_COLS & set(df.columns))
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df[feature_cols] = df[feature_cols].replace({-9: np.nan, -8: np.nan, -999: np.nan, -7: 0})

    out = df[feature_cols].copy()
    out["dataset"] = dataset_name
    out = add_dataset_onehot(out, dataset_name)
    return out


# def compute_balanced_sample_weight(y):
#     """
#     Weight each class inversely to its frequency, normalized to mean weight 1.
#     """
#     classes, counts = np.unique(y, return_counts=True)
#     inv = 1.0 / counts.astype(np.float64)
#     weight_map = dict(zip(classes, inv))
#     w = np.array([weight_map[int(yi)] for yi in y], dtype=np.float64)
#     w *= (len(w) / w.sum())
#     return w.astype(np.float32)



def compute_balanced_sample_weight(y):
    """
    Weight each class inversely to its frequency, normalized to mean weight 1.
    """
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    return w



def xgb_grid():
    """
    Return a dictionary of parameters for grid search.
    """
    return {
        "max_depth": [3, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "n_estimators": [300, 600],
    }


def batched_call(fn, X, batch_size=2500):
    """
    Run predict / predict_proba in batches to avoid out-of-memory.
    """
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        chunks.append(fn(X[start:end]))
    return np.concatenate(chunks, axis=0)




def evaluate_model(clf, X, y_true, n_classes, batch_size=2500):
    y_pred = batched_call(clf.predict, X, batch_size=batch_size)

    # For multi-class problems, you must choose an averaging method.
    # "macro" = average F1 across classes equally (common default for multi-class).
    f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(n_classes),   # keeps class set consistent
        average="macro",
        zero_division=0                # avoids warnings if a class gets no predicted positives
    )
    return f1



def train_model(X_train, y_train, n_classes, balance_probabilities=False, param_overrides=None):
    """
    Train an XGBoost model (GPU if available).
    """
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"  # EDIT: don't force cuda on a CPU-only machine
    print(f"Using device: {device}")

    sample_weight = compute_balanced_sample_weight(y_train) if balance_probabilities else None

    common_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        missing=np.nan,
        verbosity=1,
        tree_method="hist",
        device=device,  # EDIT: use computed device
    )

    if param_overrides is not None:
        common_params.update(param_overrides)

    if n_classes == 2:
        objective_params = dict(objective="binary:logistic", eval_metric="auc")
    else:
        objective_params = dict(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)

    clf = XGBClassifier(**objective_params, **common_params)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def tune_with_grid(X_tr, y_tr, X_val, y_val, n_classes, balance_probabilities, param_grid):
    best_params = None
    best_f1 = -np.inf

    for params in ParameterGrid(param_grid):
        clf = train_model(X_tr, y_tr, n_classes, balance_probabilities, param_overrides=params)
        f1 = evaluate_model(clf, X_val, y_val, n_classes=n_classes, batch_size=2500)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    return best_params, best_f1



# Align any dataframe to training feature columns (same cols, same order)
def align_to_feature_cols(df, all_feature_cols):
    df2 = df.copy()
    for c in all_feature_cols:
        if c not in df2.columns:
            df2[c] = np.nan
    df2 = df2[all_feature_cols]
    return df2.to_numpy(dtype=np.float32)


def predict_constrained(model, X, allowed_classes, batch_size=2500):
    proba = batched_call(model.predict_proba, X, batch_size=batch_size)

    # Map class label -> column index in predict_proba output
    class_to_col = {c: i for i, c in enumerate(model.classes_)}
    allowed_classes = np.array(list(allowed_classes), dtype=np.int64)
    cols = [class_to_col[c] for c in allowed_classes]

    sub = proba[:, cols]
    best = np.argmax(sub, axis=1)
    return allowed_classes[best]


def run_pipeline(
    balance_probabilities=False,
    use_grid_search=True,
):
    """
    Standalone end-to-end pipeline:
    - load + align + merge training data
    - encode labels (0..10)
    - split train/val
    - (optional) grid search
    - train final model (on full training set)
    - load + align test sets
    - constrained predictions per dataset
    - write combined submission
    """
    print("=" * 75)

    # 1) Load all training datasets
    dfs = [
        load_dataset(COV_TRAIN,   "covtype", "Cover_Type"),
        load_dataset(HELOC_TRAIN, "heloc",   "RiskPerformance"),
        load_dataset(HIGGS_TRAIN, "higgs",   "Label"),
    ]

    # 2) Align feature columns (union, fill missing with NaN)
    all_feature_cols = sorted(set().union(*(set(d.columns) for d in dfs)) - {"dataset", "label_raw"})
    for d in dfs:
        for c in all_feature_cols:
            if c not in d.columns:
                d[c] = np.nan

    # 3) Merge
    combined = pd.concat(dfs, ignore_index=True)

    # 4) Encode labels automatically (dataset + label), no hardcoded ranges
    combined["label_norm"] = combined["label_raw"].astype(str).str.strip().str.lower()
    combined["label_key"] = combined["dataset"].astype(str) + "__" + combined["label_norm"]
    
    le = LabelEncoder()
    y_train = le.fit_transform(combined["label_key"]).astype(np.int64)

    # Allowed classes per dataset (for constrained prediction)
    allowed = {}
    for ds in combined["dataset"].unique():
        ds_labs = combined.loc[combined["dataset"].eq(ds), "label_norm"].unique()
        ds_keys = [f"{ds}__{lab}" for lab in ds_labs]
        allowed[ds] = le.transform(ds_keys)


    # 5) Final training matrix
    X_train = combined[all_feature_cols].to_numpy(dtype=np.float32)
    print(f"Used train shape: X={X_train.shape}, y={y_train.shape}")

    

    # # --- DEBUG: print contents of X_train and y_train BEFORE training ---
    # print("X_train shape/dtype:", X_train.shape, X_train.dtype)
    # print("y_train shape/dtype:", y_train.shape, y_train.dtype)

    # # Readable preview (recommended for large datasets)
    # X_preview = pd.DataFrame(X_train[:5], columns=all_feature_cols)
    # print("\nFirst 5 rows of X_train:")
    # print(X_preview.to_string(index=False))
    # print("\nFirst 20 values of y_train:")
    # print(y_train[:20])

    # 6) Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    print(f"Train split: X={X_tr.shape}, y={y_tr.shape}")
    print(f"Val   split: X={X_val.shape}, y={y_val.shape}")

    n_classes = len(np.unique(y_train))

    # 7) Grid search + train final model
    if use_grid_search:
        best_params, best_f1 = tune_with_grid(
            X_tr, y_tr, X_val, y_val,
            n_classes=n_classes,
            balance_probabilities=balance_probabilities,
            param_grid=xgb_grid(),
        )
        print("Best params:", best_params)
        print(f"Best val F1 (macro): {best_f1:.4f}")

        # EDIT: train the final model using best_params, and keep it in ONE variable: `model`
        model = train_model(
            X_train, y_train,  # train on full training set
            n_classes=n_classes,
            balance_probabilities=balance_probabilities,
            param_overrides=best_params,
        )
    else:
        print("Grid search disabled; using default XGBoost hyperparameters.")
        # EDIT: store the trained model in `model` (not `clf`), so prediction uses the correct object
        model = train_model(
            X_train, y_train,
            n_classes=n_classes,
            balance_probabilities=balance_probabilities,
            param_overrides=None,
        )


    # 8) Load and align test sets
    cov_test_df = load_testset_x(COV_TEST, "covtype")
    heloc_test_df = load_testset_x(HELOC_TEST, "heloc")
    higgs_test_df = load_testset_x(HIGGS_TEST, "higgs")

    X_cov_test = align_to_feature_cols(cov_test_df, all_feature_cols)
    X_heloc_test = align_to_feature_cols(heloc_test_df, all_feature_cols)
    X_higgs_test = align_to_feature_cols(higgs_test_df, all_feature_cols)

    # 9) Constrained predictions (critical for combined-label training)
    y_cov_pred  = predict_constrained(model, X_cov_test,  allowed["covtype"])
    y_heloc_pred = predict_constrained(model, X_heloc_test, allowed["heloc"])
    y_higgs_pred = predict_constrained(model, X_higgs_test, allowed["higgs"])

    # 10) Convert back to original label space
    def decode_raw(pred_ids):
        keys = le.inverse_transform(pred_ids)
        return np.array([k.split("__", 1)[1] for k in keys])

    y_cov_pred_orig = decode_raw(y_cov_pred).astype(int)                # "1".."7" -> 1..7
    y_heloc_pred_orig = (decode_raw(y_heloc_pred) == "good").astype(int)  # bad->0, good->1
    y_higgs_pred_orig = (decode_raw(y_higgs_pred) == "s").astype(int)     # b->0, s->1

    print("CoverType predictions (original 1-7):", np.unique(y_cov_pred_orig))
    print("HELOC predictions (original 0-1):    ", np.unique(y_heloc_pred_orig))
    print("HIGGS predictions (original 0-1):    ", np.unique(y_higgs_pred_orig))

    # 11) Build submission with contiguous IDs (more robust than hardcoding 3501/4547)
    # EDIT: compute starts from lengths
    cov_start = 1
    heloc_start = cov_start + len(y_cov_pred_orig)
    higgs_start = heloc_start + len(y_heloc_pred_orig)

    cov_df = pd.DataFrame({
        "ID": np.arange(cov_start, cov_start + len(y_cov_pred_orig)),
        "Prediction": y_cov_pred_orig
    })
    heloc_df = pd.DataFrame({
        "ID": np.arange(heloc_start, heloc_start + len(y_heloc_pred_orig)),
        "Prediction": y_heloc_pred_orig
    })
    higgs_df = pd.DataFrame({
        "ID": np.arange(higgs_start, higgs_start + len(y_higgs_pred_orig)),
        "Prediction": y_higgs_pred_orig
    })

    submission = pd.concat([cov_df, heloc_df, higgs_df], ignore_index=True)

    print("\nSubmission preview:")
    print(submission.head(10))
    print("...")
    print(submission.tail(10))
    print(f"\nTotal rows: {len(submission)}")

    submission_path = "combined_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\n Saved unified submission to: {submission_path}")

    return model


if __name__ == "__main__":
    run_pipeline(
        balance_probabilities=True,
        use_grid_search=True,
    )
