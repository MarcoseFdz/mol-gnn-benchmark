"""Train and evaluate a single XGBoost baseline on BACE ECFP4 fingerprints."""

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.dataset import download_bace


MAX_ATOMS = 50
FINGERPRINT_SIZE = 2048
RANDOM_STATE = 42


def load_bace_smiles_splits(max_atoms=MAX_ATOMS):
    """Use the same valid-molecule filter and stratified split as load_graphs."""
    df = pd.read_csv(download_bace())[["mol", "Class"]].dropna()

    smiles, labels = [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["mol"])
        if mol is not None and mol.GetNumAtoms() <= max_atoms:
            smiles.append(row["mol"])
            labels.append(int(row["Class"]))

    smiles = np.array(smiles)
    labels = np.array(labels, dtype=np.int32)
    rng = np.random.default_rng(RANDOM_STATE)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_class(indices, n_train=1000, n_val=250, n_test=250):
        ratio = len(indices) / (n_train + n_val + n_test)
        n_train_class = round(n_train * ratio)
        n_val_class = round(n_val * ratio)
        n_test_class = round(n_test * ratio)
        return (
            indices[:n_train_class],
            indices[n_train_class:n_train_class + n_val_class],
            indices[n_train_class + n_val_class:n_train_class + n_val_class + n_test_class],
        )

    pos_train, pos_val, pos_test = split_class(pos_idx)
    neg_train, neg_val, neg_test = split_class(neg_idx)
    split_indices = []
    for indices in (
        np.concatenate([pos_train, neg_train]),
        np.concatenate([pos_val, neg_val]),
        np.concatenate([pos_test, neg_test]),
    ):
        rng.shuffle(indices)
        split_indices.append(indices)

    return tuple((smiles[idx], labels[idx]) for idx in split_indices)


def ecfp4_features(smiles, n_bits=FINGERPRINT_SIZE):
    """Generate radius-2 Morgan fingerprints (ECFP4) as float32 features."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return np.vstack([
        generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi)).astype(np.float32)
        for smi in smiles
    ])


def main():
    print("Loading BACE and generating ECFP4 fingerprints...")
    (smiles_train, y_train), (smiles_val, y_val), (smiles_test, y_test) = load_bace_smiles_splits()
    x_train = ecfp4_features(smiles_train)
    x_val = ecfp4_features(smiles_val)
    x_test = ecfp4_features(smiles_test)

    neg_count, pos_count = np.bincount(y_train, minlength=2)
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=neg_count / pos_count,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Training XGBoost once (early stopping monitors validation ROC-AUC)...")
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    best_iteration = model.best_iteration
    predict_args = {"iteration_range": (0, best_iteration + 1)} if best_iteration is not None else {}
    probs = model.predict_proba(x_test, **predict_args)[:, 1]
    preds = (probs > 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds, zero_division=0)

    os.makedirs("models/weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    model_path = "models/weights/XGBoost_ECFP4.json"
    results_path = "results/xgboost_benchmark_results.json"
    model.save_model(model_path)

    results = {
        "XGBoost_ECFP4": {
            "roc_auc_mean": float(roc_auc),
            "roc_auc_std": 0.0,
            "f1_mean": float(f1),
            "f1_std": 0.0,
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
        }
    }
    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\nFINAL XGBOOST RESULT")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Best iteration: {best_iteration + 1 if best_iteration is not None else 'all'}")
    print(f"  Model saved to: {model_path}")
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
