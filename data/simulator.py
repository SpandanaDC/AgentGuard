"""
AgentGuard — Agentic Transaction Simulator
============================================
There is no public labeled dataset for agent-initiated payment fraud
(the underlying protocols only started shipping in 2025-2026), so this
simulator generates one grounded in four documented fraud archetypes:

  1. compromised_agent   — a hijacked agent reusing a trusted token, but
                            deviating from its own historical pattern
  2. impersonated_agent  — a new/fake agent identity mimicking a mature
                            agent's traffic shape without real history
  3. intent_drift        — agent transacts outside the merchant-category
                            scope the user actually authorized
  4. credential_testing  — rapid low-value probing across merchants,
                            routed through an agent to look legitimate

Everything else is labeled `legitimate_agent`.

Usage:
    python simulator.py --n_users 2000 --days 30 --fraud_rate 0.04
"""

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

RNG = np.random.default_rng(42)

MERCHANT_CATEGORIES = [
    "groceries", "electronics", "food_delivery", "travel",
    "subscriptions", "fashion", "utilities", "entertainment",
]

FRAUD_LABELS = [
    "compromised_agent", "impersonated_agent", "intent_drift", "credential_testing"
]


@dataclass
class Agent:
    agent_id: str
    user_id: str
    created_day: int                 # how many days ago the agent was delegated
    authorized_categories: list      # merchant categories the user actually approved
    spending_limit: float
    typical_amount: float
    typical_cadence_per_day: float   # avg transactions/day
    typical_device: str


def _make_population(n_users: int):
    """Create the underlying users/agents/merchants for the simulation."""
    merchants = pd.DataFrame({
        "merchant_id": [f"M{idx:04d}" for idx in range(200)],
        "category": RNG.choice(MERCHANT_CATEGORIES, size=200),
        "base_risk": RNG.beta(2, 20, size=200),  # most merchants are low risk
    })

    agents = []
    for i in range(n_users):
        user_id = f"U{i:05d}"
        agent_id = f"A{i:05d}"
        n_cats = RNG.integers(1, 3)
        authorized = list(RNG.choice(MERCHANT_CATEGORIES, size=n_cats, replace=False))
        agents.append(Agent(
            agent_id=agent_id,
            user_id=user_id,
            created_day=int(RNG.integers(1, 200)),
            authorized_categories=authorized,
            spending_limit=float(RNG.choice([2000, 5000, 10000, 25000])),
            typical_amount=float(RNG.gamma(3, 400)),
            typical_cadence_per_day=float(RNG.uniform(0.2, 2.0)),
            typical_device=f"D{RNG.integers(0, n_users * 2):06d}",
        ))
    return merchants, agents


def _legit_transaction(agent: Agent, merchants: pd.DataFrame, day: int, ts_id: int):
    cat = RNG.choice(agent.authorized_categories)
    merchant_row = merchants[merchants.category == cat].sample(1, random_state=int(RNG.integers(1e6))).iloc[0]
    amount = max(50, RNG.normal(agent.typical_amount, agent.typical_amount * 0.25))

    # Real legitimate behavior is messier than a clean template: occasional
    # secondary device (shared family device, new phone), occasional one-off
    # large purchase (annual renewal, gift). Without this, fraud is trivially
    # separable and the model looks unrealistically perfect.
    device = agent.typical_device
    if RNG.random() < 0.05:
        device = f"D{RNG.integers(0, 999999):06d}"
    if RNG.random() < 0.04:
        amount = amount * RNG.uniform(2.5, 5)

    return {
        "txn_id": f"T{ts_id:08d}",
        "agent_id": agent.agent_id,
        "user_id": agent.user_id,
        "merchant_id": merchant_row.merchant_id,
        "category": cat,
        "day": day,
        "amount": round(amount, 2),
        "agent_age_days": agent.created_day + day,
        "device_fingerprint": device,
        "token_scope_ok": True,
        "within_spending_limit": amount <= agent.spending_limit,
        "label": "legitimate_agent",
    }


def _fraud_transaction(agent: Agent, merchants: pd.DataFrame, day: int, ts_id: int, kind: str):
    base = _legit_transaction(agent, merchants, day, ts_id)

    if kind == "compromised_agent":
        # same trusted agent/token, but behavior deviates sharply from its own history
        off_cats = [c for c in MERCHANT_CATEGORIES if c not in agent.authorized_categories]
        cat = RNG.choice(off_cats) if off_cats else RNG.choice(MERCHANT_CATEGORIES)
        merchant_row = merchants[merchants.category == cat].sample(1).iloc[0]
        base.update({
            "merchant_id": merchant_row.merchant_id,
            "category": cat,
            "amount": round(agent.typical_amount * RNG.uniform(3, 8), 2),
            "device_fingerprint": f"D{RNG.integers(0, 999999):06d}",  # new device
            "token_scope_ok": True,  # token itself is still "valid" — that's the danger
        })

    elif kind == "impersonated_agent":
        # claims to be a mature agent but behaves like a brand-new, hyperactive one
        base.update({
            "agent_age_days": int(RNG.integers(0, 2)),  # freshly minted
            "amount": round(agent.typical_amount * RNG.uniform(1.5, 4), 2),
            "device_fingerprint": f"D{RNG.integers(0, 999999):06d}",
            "token_scope_ok": True,
        })

    elif kind == "intent_drift":
        # transacts outside the user-authorized category scope
        off_cats = [c for c in MERCHANT_CATEGORIES if c not in agent.authorized_categories]
        cat = RNG.choice(off_cats) if off_cats else RNG.choice(MERCHANT_CATEGORIES)
        merchant_row = merchants[merchants.category == cat].sample(1).iloc[0]
        base.update({
            "merchant_id": merchant_row.merchant_id,
            "category": cat,
            "token_scope_ok": False,
        })

    elif kind == "credential_testing":
        # small probing amount, will be emitted in a rapid burst by the caller
        base.update({
            "amount": round(RNG.uniform(1, 50), 2),
            "device_fingerprint": f"D{RNG.integers(0, 999999):06d}",
            "token_scope_ok": True,
        })

    base["label"] = kind
    base["within_spending_limit"] = base["amount"] <= agent.spending_limit
    return base


def simulate(n_users: int, days: int, fraud_rate: float):
    merchants, agents = _make_population(n_users)
    rows = []
    ts_id = 0

    for agent in agents:
        for day in range(days):
            n_txns = np.random.poisson(agent.typical_cadence_per_day)
            for _ in range(n_txns):
                if RNG.random() < fraud_rate:
                    kind = RNG.choice(FRAUD_LABELS)
                    if kind == "credential_testing":
                        # burst of 4-10 rapid small transactions same day
                        for _ in range(RNG.integers(4, 10)):
                            rows.append(_fraud_transaction(agent, merchants, day, ts_id, kind))
                            ts_id += 1
                        continue
                    rows.append(_fraud_transaction(agent, merchants, day, ts_id, kind))
                else:
                    rows.append(_legit_transaction(agent, merchants, day, ts_id))
                ts_id += 1

    df = pd.DataFrame(rows).sort_values(["agent_id", "txn_id"]).reset_index(drop=True)
    return df, merchants, agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=2000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--fraud_rate", type=float, default=0.04)
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    args = parser.parse_args()

    txns, merchants, agents = simulate(args.n_users, args.days, args.fraud_rate)
    txns.to_csv(args.out, index=False)
    merchants.to_csv("data/merchants.csv", index=False)

    # persist agent metadata (spending limits, authorized scope) so downstream
    # feature engineering doesn't need to re-run the simulator internals
    agents_df = pd.DataFrame([{
        "agent_id": a.agent_id,
        "user_id": a.user_id,
        "created_day": a.created_day,
        "authorized_categories": ";".join(a.authorized_categories),
        "spending_limit": a.spending_limit,
        "typical_amount": a.typical_amount,
        "typical_cadence_per_day": a.typical_cadence_per_day,
        "typical_device": a.typical_device,
    } for a in agents])
    agents_df.to_csv("data/agents.csv", index=False)

    print(f"Generated {len(txns):,} transactions for {args.n_users:,} agents over {args.days} days")
    print("\nLabel distribution:")
    print(txns.label.value_counts())
    print(f"\nFraud rate (actual): {(txns.label != 'legitimate_agent').mean():.2%}")
