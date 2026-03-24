"""
src/data/splits.py

Patient-level stratified splits for BreakHis.

WHY PATIENT-LEVEL MATTERS
--------------------------
BreakHis contains up to 32 images per patient (4 magnifications x ~8 images
each). These images are highly correlated — they are literally the same biopsy
tissue viewed differently. Image-level splitting allows the same patient to
appear in both training and test, inflating every reported metric because the
model has effectively seen the patient before evaluation.

The BreakHis dataset ships with an official 5-fold patient-level split
(Folds.csv, released by Spanhol et al. 2016). Despite its availability,
the majority of published studies bypass it in favour of image-level
train_test_split — a pattern consistent with most published BreakHis literature,
but one that produces optimistic and non-reproducible results.

This implementation corrects that by:
  1. Extracting patient ID from the BreakHis filename convention
  2. Grouping all images by patient before any split
  3. Ensuring every image from a given patient appears in exactly one partition

BREAKHIS FILENAME CONVENTION
-----------------------------
SOB_B_A-14-22549AB-40-001.png
         ^  ^       ^   ^
         |  |       |   sequence number
         |  |       magnification (40|100|200|400)
         |  patient slide ID
         year

Patient key = everything before the magnification field:
  SOB_B_A-14-22549AB  <- unique per patient, shared across all magnifications
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


# --------------------------------------------------------------------------
# Patient ID extraction
# --------------------------------------------------------------------------

def extract_patient_id(filepath: str) -> str:
    """
    Extract patient identifier from a BreakHis image filepath.

    Given SOB_B_A-14-22549AB-40-001.png, returns 'SOB_B_A-14-22549AB'.
    This string is unique per patient and shared across all magnifications.

    Falls back to the patient directory name if the filename does not match
    the expected pattern (handles edge cases in some BreakHis distributions).

    Raises ValueError if no patient ID can be determined.
    """
    fname = Path(filepath).stem  # remove .png extension

    # Primary: match the standard BreakHis filename pattern
    match = re.match(
        r'^(SOB_[BM]_[A-Za-z]+-\d+-[A-Z0-9]+)-(?:40|100|200|400)-\d+$',
        fname
    )
    if match:
        return match.group(1)

    # Fallback: patient directory sits above the magnification directory
    # .../SOB/<type>/<patient_folder>/<mag>X/<image.png>
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part in ('40X', '100X', '200X', '400X'):
            if i > 0:
                return parts[i - 1]

    raise ValueError(
        f"Cannot extract patient ID from: {filepath}\n"
        f"Expected BreakHis convention: SOB_<C>_<T>_<yr>-<slideID>-<mag>-<seq>.png  (or hyphen variant)"
    )


def add_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'patient_id' column to a BreakHis DataFrame."""
    df = df.copy()
    df['patient_id'] = df['path'].apply(extract_patient_id)
    n_pat = df['patient_id'].nunique()
    print(f"Identified {n_pat} unique patients "
          f"({len(df) / n_pat:.1f} images/patient on average)")
    return df


# --------------------------------------------------------------------------
# Train / test split — patient level
# --------------------------------------------------------------------------

def train_test_split_by_patient(df: pd.DataFrame,
                                test_size: float = 0.20,
                                random_seed: int = 42) -> tuple:
    """
    Stratified train/test split at the patient level.

    Class ratio (benign:malignant) is approximately preserved in both
    partitions. No patient appears in both.

    Returns (train_df, test_df) with reset indices.
    """
    if 'patient_id' not in df.columns:
        df = add_patient_ids(df)

    # One representative label per patient (majority vote across their images)
    patient_labels = (
        df.groupby('patient_id')['label']
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_seed)
    tr_pat_idx, te_pat_idx = next(
        sss.split(patient_labels['patient_id'], patient_labels['label'])
    )

    train_patients = set(patient_labels.iloc[tr_pat_idx]['patient_id'])
    test_patients  = set(patient_labels.iloc[te_pat_idx]['patient_id'])

    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    test_df  = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

    _report("Train", train_df)
    _report("Test ", test_df)
    _assert_no_leakage(train_df, test_df, "train/test")

    return train_df, test_df


# --------------------------------------------------------------------------
# K-fold CV — patient level
# --------------------------------------------------------------------------

def kfold_splits_by_patient(train_df: pd.DataFrame,
                             n_folds: int = 5,
                             random_seed: int = 42):
    """
    Stratified k-fold cross-validation at the patient level.

    Yields (fold_index, fold_train_df, fold_val_df).
    No patient appears in both fold_train and fold_val.

    n_folds=5 matches the official BreakHis Folds.csv protocol.
    """
    if 'patient_id' not in train_df.columns:
        train_df = add_patient_ids(train_df)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                random_state=random_seed)

    X      = train_df['path'].values
    y      = train_df['label'].values
    groups = train_df['patient_id'].values

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val   = train_df.iloc[val_idx].reset_index(drop=True)

        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        _report("  Train", fold_train)
        _report("  Val  ", fold_val)
        _assert_no_leakage(fold_train, fold_val, f"fold {fold + 1}")

        yield fold, fold_train, fold_val


# --------------------------------------------------------------------------
# Class weights
# --------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Inverse-frequency class weights for weighted binary cross-entropy.
    Recomputed per fold from the fold's training partition.

    Returns {0: w_benign, 1: w_malignant}.
    """
    n       = len(labels)
    n_b     = (labels == 0).sum()
    n_m     = (labels == 1).sum()

    if n_b == 0 or n_m == 0:
        raise ValueError(
            f"Degenerate partition: benign={n_b}, malignant={n_m}. "
            "All patients of one class landed in the same fold — "
            "check dataset class distribution or reduce n_folds."
        )

    w_b = n / (2.0 * n_b)
    w_m = n / (2.0 * n_m)
    print(f"  Class weights — benign: {w_b:.4f}, malignant: {w_m:.4f}")
    return {0: w_b, 1: w_m}


# --------------------------------------------------------------------------
# Image-level split — for the ablation comparison only
# --------------------------------------------------------------------------

def train_test_split_by_image(df: pd.DataFrame,
                              test_size: float = 0.20,
                              random_seed: int = 42) -> tuple:
    """
    Image-level stratified split.

    This is the approach used in the majority of published BreakHis studies
    and in the original notebook series this reconstruction is based on.
    It allows the same patient to appear in both train and test, inflating
    all reported metrics.

    PROVIDED FOR COMPARISON ONLY to populate the split-protocol ablation
    table in the paper (Table: image-level vs patient-level metrics).
    DO NOT use for main experimental results.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_seed)
    tr_idx, te_idx = next(sss.split(df['path'], df['label']))
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    test_df  = df.iloc[te_idx].reset_index(drop=True)

    print("[IMAGE-LEVEL SPLIT — ablation comparison only]")
    _report("Train", train_df)
    _report("Test ", test_df)
    return train_df, test_df


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _report(label: str, df: pd.DataFrame):
    n_b   = (df['label'] == 0).sum()
    n_m   = (df['label'] == 1).sum()
    n_pat = df['patient_id'].nunique() if 'patient_id' in df.columns else '?'
    print(f"{label}: {len(df):>5} images  "
          f"({n_b} benign, {n_m} malignant)  "
          f"{n_pat} patients")


def _assert_no_leakage(df_a: pd.DataFrame, df_b: pd.DataFrame, context: str):
    """Raise ValueError if any patient ID appears in both partitions."""
    if 'patient_id' not in df_a.columns or 'patient_id' not in df_b.columns:
        return
    overlap = set(df_a['patient_id']) & set(df_b['patient_id'])
    if overlap:
        raise ValueError(
            f"PATIENT LEAKAGE in {context}: "
            f"{len(overlap)} patient(s) appear in both partitions.\n"
            f"First 5: {sorted(overlap)[:5]}"
        )
    print(f"  ✓ No patient leakage detected in {context}")
