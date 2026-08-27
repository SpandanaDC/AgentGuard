# AgentGuard
**Real-time risk & trust scoring for agent-initiated payments**
*Razorpay AI Builder Internship 2026 — Track 2: AI Risk Manager*

## The problem

AI agents are starting to hold delegated, pre-authorized payment credentials and
transact on a user's behalf. When an agent is compromised or impersonated, the
resulting transactions reuse a trusted token and a familiar spending pattern —
they don't look like fraud to models trained on human-fraud signals. AgentGuard
is a risk-scoring layer purpose-built for agent behavior itself, with an
explainable rationale for every flagged transaction.

See [`docs/`](docs/) for the full architecture writeup.

## Status

| Layer | Component | Status |
|---|---|---|
| Data | Agentic Transaction Simulator | ✅ built (`data/simulator.py`) |
| Data | Feature engineering (leakage-free) | ✅ built (`features/build_features.py`) |
| 1 | Real-time risk scorer (LightGBM baseline) | ✅ built, **eval caveat documented** — see [`docs/eval_notes.md`](docs/eval_notes.md) |
| 2 | Agent trust graph (GNN) | ⏳ next |
| 3 | Explainability layer (SHAP + LLM rationale) | ⏳ next |
| — | Risk-ops dashboard | ⏳ next |

## Quick start

```bash
pip install -r requirements.txt

# 1. generate the synthetic labeled dataset
python data/simulator.py --n_users 2000 --days 30 --fraud_rate 0.04

# 2. build leakage-free behavioral features
python features/build_features.py

# 3. train + evaluate the baseline real-time risk scorer
python models/realtime/train_baseline.py
```

Generated data files (`data/*.csv`, `features/*.csv`) are not committed —
regenerate them with the commands above (they're deterministic, seeded with a
fixed random state, so results are reproducible).

## Why a simulator instead of real data

No public labeled dataset for agent-initiated payment fraud exists yet — the
underlying delegation protocols only started shipping in 2025-2026. `data/simulator.py`
generates realistic legitimate-agent behavior and injects four documented fraud
archetypes (`compromised_agent`, `impersonated_agent`, `intent_drift`,
`credential_testing`) on top of it. Full rationale in [`docs/`](docs/).

## Repo structure

```
agentguard/
├── data/            # simulator + generated (gitignored) datasets
├── features/        # feature engineering pipeline
├── models/
│   ├── realtime/     # LightGBM risk scorer
│   └── graph/        # GNN trust-graph model (next)
├── explainability/   # SHAP + LLM rationale generator (next)
├── dashboard/        # risk-ops console (next)
├── eval/             # cost-benefit / financial-case analysis (next)
└── docs/             # architecture notes, eval notes
```
