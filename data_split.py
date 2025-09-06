import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os, argparse, random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def make_splits(test_size=0.2, split_strategy="feature", feature_name="mean radius"):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    df = X.copy()
    df['target'] = y
    # Confirm target mapping
    target_names = load_breast_cancer().target_names  # ['malignant', 'benign']
    mapping_info = {0: target_names[0], 1: target_names[1]}
    # Hold-out test
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['target'], random_state=SEED)
    # Non-IID split
    if split_strategy == "feature":
        median_val = train_df[feature_name].median()
        hospA = train_df[train_df[feature_name] >= median_val].copy()
        hospB = train_df[train_df[feature_name] < median_val].copy()
    else:  # 'label' strategy
        # 60% malignant to A, 40% to B; vice-versa for benign
        A_parts, B_parts = [], []
        for label in [0,1]:
            label_df = train_df[train_df['target']==label].sample(frac=1.0, random_state=SEED)
            cut = int(0.6*len(label_df))
            if label == 0:  # malignant
                A_parts.append(label_df.iloc[:cut])
                B_parts.append(label_df.iloc[cut:])
            else:          # benign
                # flip
                B_parts.append(label_df.iloc[:cut])
                A_parts.append(label_df.iloc[cut:])
        hospA = pd.concat(A_parts).sample(frac=1.0, random_state=SEED)
        hospB = pd.concat(B_parts).sample(frac=1.0, random_state=SEED)
    return hospA, hospB, test_df, mapping_info

def main(outdir=".", test_size=0.2, split_strategy="feature"):
    os.makedirs(outdir, exist_ok=True)
    hospA, hospB, test_df, mapping = make_splits(test_size=test_size, split_strategy=split_strategy)
    hospA.to_csv(os.path.join(outdir, "hospital_A.csv"), index=False)
    hospB.to_csv(os.path.join(outdir, "hospital_B.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test_set.csv"), index=False)
    # Class distribution table
    dist = {
        "hospital": ["A","B","Test"],
        "n_samples": [len(hospA), len(hospB), len(test_df)],
        "positive(1)": [int((hospA['target']==1).sum()), int((hospB['target']==1).sum()), int((test_df['target']==1).sum())],
        "negative(0)": [int((hospA['target']==0).sum()), int((hospB['target']==0).sum()), int((test_df['target']==0).sum())],
    }
    dist_df = pd.DataFrame(dist)
    dist_df.to_csv(os.path.join(outdir, "class_distribution.csv"), index=False)
    with open(os.path.join(outdir, "target_mapping.json"), "w") as f:
        import json; json.dump(mapping, f, indent=2)
    print("Saved splits to:", outdir)
    print("Target mapping:", mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-strategy", type=str, choices=["feature","label"], default="feature")
    args = parser.parse_args()
    main(outdir=args.outdir, test_size=args.test_size, split_strategy=args.split_strategy)
