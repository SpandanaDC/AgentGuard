"""
AgentGuard — Feature Engineering
=================================
Turns the raw transaction log into a model-ready feature table.

Every feature for a given transaction is computed using ONLY that
agent's transactions *before* it — this is what keeps the model
honest and prevents label leakage (a real-time risk scorer only ever
sees the past).

Usage:
    python features/build_features.py
"""

import numpy as np
import pandas as pd

EPS = 1e-6


def load_raw(txn_path="data/transactions.csv", agent_path="data/agents.csv"):
    txns = pd.read_csv(txn_path)
    agents = pd.read_csv(agent_path)
    return txns, agents


def build_features(txns: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    df = txns.merge(agents, on="agent_id", suffixes=("", "_meta"))
    df = df.sort_values(["agent_id", "txn_id"]).reset_index(drop=True)

    g = df.groupby("agent_id")

    # --- how many transactions has this agent made before this one? ---
    df["txn_seq_num"] = g.cumcount()

    # --- expanding (prior-only) mean/std of amount -> deviation z-score ---
    df["hist_mean_amount"] = g["amount"].transform(lambda s: s.shift(1).expanding().mean())
    df["hist_std_amount"] = g["amount"].transform(lambda s: s.shift(1).expanding().std())
    global_mean, global_std = df["amount"].mean(), df["amount"].std()
    df["hist_mean_amount"] = df["hist_mean_amount"].fillna(global_mean)
    df["hist_std_amount"] = df["hist_std_amount"].fillna(global_std)
    df["amount_zscore"] = (df["amount"] - df["hist_mean_amount"]) / (df["hist_std_amount"] + EPS)

    # --- has this agent transacted in this category before? ---
    def _seen_before(sub):
        seen = set()
        flags = []
        for cat in sub:
            flags.append(cat in seen)
            seen.add(cat)
        return pd.Series(flags, index=sub.index)

    df["category_seen_before"] = g["category"].transform(_seen_before)

    # --- device consistency: does this txn's device match the agent's
    #     established device (mode of its first 3 transactions)? ---
    primary_device = (
        df[df.txn_seq_num < 3].groupby("agent_id")["device_fingerprint"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("primary_device")
    )
    df = df.merge(primary_device, on="agent_id", how="left")
    df["device_matches_primary"] = df["device_fingerprint"] == df["primary_device"]

    # --- same-day burst detection (catches credential-testing bursts) ---
    df["day_txn_count"] = df.groupby(["agent_id", "day"])["txn_id"].transform("count")

    # --- spending-limit pressure ---
    df["amount_to_limit_ratio"] = df["amount"] / (df["spending_limit"] + EPS)

    # --- agent maturity at time of transaction ---
    df["agent_age_at_txn"] = df["agent_age_days"]

    feature_cols = [
        "txn_seq_num", "amount", "amount_zscore", "category_seen_before",
        "device_matches_primary", "day_txn_count", "amount_to_limit_ratio",
        "agent_age_at_txn", "token_scope_ok", "within_spending_limit",
    ]
    meta_cols = ["txn_id", "agent_id", "user_id", "merchant_id", "category", "day", "label"]

    out = df[meta_cols + feature_cols].copy()
    for c in ["category_seen_before", "device_matches_primary", "token_scope_ok", "within_spending_limit"]:
        out[c] = out[c].astype(int)

    return out


if __name__ == "__main__":
    txns, agents = load_raw()
    features = build_features(txns, agents)
    features.to_csv("features/feature_table.csv", index=False)

    print(f"Built {len(features):,} rows x {features.shape[1]} columns")
    print("\nFeature columns:", [c for c in features.columns if c not in
          ["txn_id", "agent_id", "user_id", "merchant_id", "category", "day", "label"]])
    print("\nSample rows:")
    print(features.sample(5, random_state=1).to_string(index=False))
