import os, argparse, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from src.utils import metric_dict, plot_roc_curves

SEED = 42

def load_data(test_size=0.2):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    return X_train, X_test, y_train, y_test

def train_and_eval(outdir=".", test_size=0.2):
    os.makedirs(outdir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(test_size=test_size)
    # Logistic Regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=500, random_state=SEED)
    lr.fit(X_train_s, y_train)
    lr_prob = lr.predict_proba(X_test_s)[:,1]
    lr_metrics = metric_dict(y_test.values, lr_prob)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:,1]
    rf_metrics = metric_dict(y_test.values, rf_prob)
    # Save best
    best_model = ("lr", lr, scaler) if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"] else ("rf", rf, None)
    joblib.dump(best_model, os.path.join(outdir, "centralized_best.joblib"))
    # ROC curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    plot_roc_curves(
        curves=[(lr_fpr, lr_tpr, f"LR AUC={lr_metrics['roc_auc']:.3f}"),
                (rf_fpr, rf_tpr, f"RF AUC={rf_metrics['roc_auc']:.3f}")],
        title="Centralized ROC Curves",
        outpath=os.path.join("results","centralized_roc.png")
    )
    # Save metrics table
    import pandas as pd
    df = pd.DataFrame([{"model":"LogReg", **lr_metrics}, {"model":"RandomForest", **rf_metrics}])
    df.to_csv(os.path.join("results","centralized_metrics.csv"), index=False)
    print("Saved:", os.path.join(outdir, "centralized_best.joblib"))
    print(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train_and_eval(outdir=args.outdir, test_size=args.test_size)
