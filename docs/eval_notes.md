# Evaluation Notes — Baseline Model (Layer 1)

**Result:** AUC-PR = 0.9732, near-perfect precision/recall on the time-based holdout.

**Don't trust this number at face value — here's why, and what it actually tells us.**

Digging into feature-vs-label correlation on the raw feature table:

```
token_scope_ok by label:
  intent_drift          0.0   <- always False
  everything else       1.0   <- always True

agent_age_at_txn by label:
  impersonated_agent    mean 0.49 days (range 0-1)
  everything else       mean ~115 days (range 1-228)
```

Two of the four fraud archetypes — `intent_drift` and `impersonated_agent` — have a
**deterministic, single-feature tell** baked into how the simulator generates them.
That's an artifact of the data-generation logic, not evidence the model has learned
anything subtle about agent risk. A classifier doesn't need to be smart to hit 100%
recall on a rule it was implicitly told the answer to.

**Why this matters for the pitch:** a risk model that looks flawless on synthetic
data is a yellow flag, not a green one — real fraud doesn't announce itself with a
single deterministic field, and adversaries adapt specifically to whatever the
last obvious tell was. Presenting 99.99% AUC-PR without this caveat would
undersell the actual hard problem (which is `compromised_agent` — the archetype
that reuses a *valid* token and *only* shows up as a behavioral deviation, and is
the one closest to real-world compromised-agent fraud).

**Next iteration (before the pitch video):**
- Make `token_scope_ok` probabilistic rather than a hard rule for `intent_drift`
  (some off-category purchases should occasionally still show `token_scope_ok=True`,
  mirroring token-scope bugs/edge cases in real delegation systems).
- Widen the `impersonated_agent` age distribution so it overlaps with genuinely new
  legitimate agents (a legitimate agent's first week should not be a free tell).
- Re-run the eval and report per-archetype recall separately, not just the pooled
  binary number — `compromised_agent` recall is the metric that actually matters,
  since it's the archetype identical to how real agent-hijack fraud behaves.

This file exists so that anyone reviewing the repo (or asking about the metrics in
the pitch video Q&A) sees this was caught and understood, not missed.
