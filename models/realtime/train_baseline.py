"""
AgentGuard — Layer 1: Real-Time Risk Scorer (baseline)
========================================================
LightGBM binary classifier: legitimate_agent vs. any fraud archetype.

Split strategy matters here: a random shuffle would leak future agent
behavior into training. Instead we split by DAY (train on the first
~70% of days, test on the last ~30%) so evaluation mimics how the
model will actually be used — scoring transactions it has never seen,
from time periods after its training cutoff.

Usage:
    python models/realtime/train_baseline.py
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
)

FEATURE_COLS = [
    "txn_seq_num", "amount", "amount_zscore", "category_seen_before",
    "device_matches_primary", "day_txn_count", "amount_to_limit_ratio",
    "agent_age_at_txn", "token_scope_ok", "within_spending_limit",
]


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7):
    cutoff_day = df["day"].quantile(train_frac)
    train = df[df["day"] <= cutoff_day]
    test = df[df["day"] > cutoff_day]
    return train, test, cutoff_day


def train_and_evaluate(feature_path="features/feature_table.csv"):
    df = pd.read_csv(feature_path)
    df["is_fraud"] = (df["label"] != "legitimate_agent").astype(int)

    train, test, cutoff = time_based_split(df)
    print(f"Time-based split at day {cutoff:.0f}: "
          f"{len(train):,} train rows / {len(test):,} test rows")
    print(f"Train fraud rate: {train.is_fraud.mean():.2%} | "
          f"Test fraud rate: {test.is_fraud.mean():.2%}\n")

    X_train, y_train = train[FEATURE_COLS], train["is_fraud"]
    X_test, y_test = test[FEATURE_COLS], test["is_fraud"]

    # class imbalance handling
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    # --- ranking quality ---
    ap = average_precision_score(y_test, proba)
    print(f"Average Precision (AUC-PR): {ap:.4f}")
    print(f"(baseline / random-guess AUC-PR would be ~{y_test.mean():.4f} — the fraud rate itself)\n")

    # --- pick an operating threshold and report precision/recall there ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    # choose threshold that gets recall >= 0.85 with best precision at that recall
    target_recall = 0.85
    valid = recalls[:-1] >= target_recall
    if valid.any():
        best_idx = np.where(valid)[0][np.argmax(precisions[:-1][valid])]
        threshold = thresholds[best_idx]
    else:
        threshold = 0.5
    preds = (proba >= threshold).astype(int)

    print(f"Operating threshold (targeting {target_recall:.0%} recall): {threshold:.4f}\n")
    print("Confusion matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["legitimate", "fraud"]))

    # --- feature importance ---
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.to_string())

    # --- latency benchmark: single-row scoring time (real-time authorization path) ---
    sample = X_test.sample(1, random_state=0)
    timings = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        timings.append((time.perf_counter() - t0) * 1000)
    timings = np.array(timings)
    print(f"\nSingle-transaction scoring latency: "
          f"p50={np.percentile(timings, 50):.2f}ms  "
          f"p99={np.percentile(timings, 99):.2f}ms")

    model.booster_.save_model("models/realtime/lgbm_risk_model.txt")
    return model, (X_test, y_test, proba, threshold)


if __name__ == "__main__":
    train_and_evaluate()
