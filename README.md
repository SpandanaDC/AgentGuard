# 🛡️ ImposterAgent (AgentGuard)
> **Real-Time Trust & Risk Scoring Engine for Agentic Commerce**  


---

## 🚀 Executive Summary
As autonomous AI agents take over consumer financial workflows (delegated checkouts, pre-authorized spending limits), traditional fraud detection models fail because compromised agents reuse trusted tokens and saved cards, leaving a completely "clean" fraud signature[cite: 1]. 

**ImposterAgent** solves this by introducing a purpose-built, real-time risk scoring engine optimized for payment authorization latencies[cite: 1], featuring a **true closed-loop adversarial red-teaming system** that hardens the model against adaptive zero-day attacks.

---

## 🏛️ System Architecture
```mermaid
graph TD
    User["Agent Payment Request"] --> FastAPI["FastAPI / Scoring Layer"]
    FastAPI --> Layer1["<b>Layer 1: Real-Time Risk Scorer</b> <br>• LightGBM Model (<15ms P99) <br>• Z-Scores & Spending Limit Ratios"]
    FastAPI --> Layer2["<b>Layer 2: Agent Trust Graph</b> <br>• GNN / Entity Clustering <br>• Fan-out Abuse Tracking"]
    Layer1 --> Layer3["<b>Layer 3: Explainability & Rationale</b> <br>• SHAP Values & LLM Rationale Audit Trail"]
    Layer1 --> Decision{"Risk Threshold"}
    Decision -->|Low Risk| Greenlight["Approve & Uplift Conversion"]
    Decision -->|High Risk| Flag["Escalate to Risk-Ops Dashboard"]
