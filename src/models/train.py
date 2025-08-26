import argparse, os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def metrics_dict(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    y = df["y"].values
    # feature columns = all numeric except y/date/match_id
    drop_cols = {"y","date","match_id"}
    X = df.drop(columns=[c for c in df.columns if c in drop_cols]).values

    # Temporal split: newest 20% as test
    cutoff = int(0.8 * len(df))
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:,1]
    lr_metrics = metrics_dict(y_test, lr_prob)

    # XGBoost + probability calibration
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42
    )
    xgb_cal = CalibratedClassifierCV(xgb, method="isotonic", cv=5)
    xgb_cal.fit(X_train, y_train)
    xgb_prob = xgb_cal.predict_proba(X_test)[:,1]
    xgb_metrics = metrics_dict(y_test, xgb_prob)

    # Save artifacts
    joblib.dump({"model": xgb_cal, "feature_names": [c for c in df.columns if c not in ("y","date","match_id")]},
                os.path.join(args.artifacts_dir, "model_xgb_calibrated.joblib"))
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump({"logreg": lr_metrics, "xgb_cal": xgb_metrics,
                   "n_train": int(len(y_train)), "n_test": int(len(y_test))}, f, indent=2)

    # Reliability plot
    plt.figure()
    CalibrationDisplay.from_predictions(y_test, lr_prob, name="LogReg")
    CalibrationDisplay.from_predictions(y_test, xgb_prob, name="XGB (calibrated)")
    plt.title("Reliability curve (temporal holdout)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts_dir, "calibration.png"), dpi=160)

    print("Baseline (LR):", lr_metrics)
    print("XGB (Calibrated):", xgb_metrics)
    print(f"Saved artifacts to: {args.artifacts_dir}")

if __name__ == "__main__":
    main()
