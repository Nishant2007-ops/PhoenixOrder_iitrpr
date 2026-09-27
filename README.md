# 🔥 PhoenixOrder_iitrpr

### AI-Powered Predictive Cyber Defence Using Temporal Network Behaviour

> **SIH 2026 — Problem Statement: SIH26153**

PhoenixOrder is an **active cyber-defence prototype** for representing network behaviour over time, flagging attack-like windows, and presenting evidence for analyst review. The longer-term project goal is to learn how attacks evolve and anticipate their likely next stage.

Instead of only asking **"Is this network traffic malicious?"**, PhoenixOrder aims to answer a more important question:

> **"What is the attacker likely to do next?"**

The current prototype converts flow records into **temporal network states** and uses a multitask Delta-LSTM to predict the next-state feature change and classify the next window as attack or benign. It does not currently predict the next attack stage.

> **Prototype benchmark:** the current classifier benchmark has a high false-positive rate. On the fixed 218-window test set, the Logistic Regression FPR is **0.9695** and the Multitask Delta-LSTM FPR is **0.7500**. PhoenixOrder is a research prototype, not a production IDS. The current LSTM attack output is an attack probability; it does not directly predict an attack stage.

---

## 🎯 1. Problem

Traditional Intrusion Detection Systems mainly focus on identifying whether current network traffic is:

* Normal
* Suspicious
* Malicious

However, cyberattacks are not isolated events. They usually follow a sequence of actions.

For example:

```text
Reconnaissance
      ↓
Initial Access
      ↓
Execution
      ↓
Persistence
      ↓
Privilege Escalation
      ↓
Lateral Movement
      ↓
Command & Control
      ↓
Impact
```

If a system can understand this **progression over time**, it can potentially warn defenders before the attacker reaches the next stage.

This is the core motivation behind PhoenixOrder.

---

# 🚀 2. Our Solution

PhoenixOrder transforms raw network traffic into meaningful **temporal network states**.

### Our core pipeline:

```text
Network Traffic
      ↓
Data Ingestion
      ↓
Data Profiling & Validation
      ↓
Preprocessing
      ↓
Canonical Feature Extraction
      ↓
Temporal Network States
      ↓
AI/ML Temporal Model
      ↓
Attack Progression Prediction
      ↓
MITRE ATT&CK Mapping
      ↓
Explainable Alert
      ↓
Proactive Cyber Defence
```

The current prototype includes an offline flow-CSV inference pipeline and a multitask temporal classifier. Predicting the next *attack stage* remains a future goal; the current stage output is a separate rule-based interpretation of behavioral evidence.

---

# 🧠 3. What Makes PhoenixOrder Different?

The key idea is **time**.

Instead of treating every network flow independently:

```text
Flow 1 → Malicious
Flow 2 → Malicious
Flow 3 → Malicious
```

we aim to understand the sequence:

```text
State 1 → State 2 → State 3 → State 4
   ↓         ↓         ↓         ↓
Recon     Access    Execution   C2
```

Temporal context helps represent how network behaviour changes over time. The current pipeline aggregates flows into five-minute states and uses twelve contiguous states (about one hour) as context. The model learns next-state feature change and attack classification; the stage label shown by the prototype is generated separately by rules.

Our long-term goal is to move from:

> **Detection → Prediction → Proactive Defence**

---

# 📊 4. Datasets

We use publicly available cybersecurity datasets to develop and validate the system.

### CSE-CIC-IDS2018

The dataset provides network traffic containing both benign and attack traffic.

We perform dataset profiling to understand:

* Available files
* Feature availability
* Column consistency
* Missing values
* Feature types
* Dataset compatibility

### CTU-13

CTU-13 provides network traffic from botnet scenarios and is used as an additional source for studying network behaviour.

Using multiple datasets helps us investigate whether our feature representation can work across different network environments.

---

# ⚙️ 5. Current Technical Pipeline

## Step 1 — Data Ingestion

Network traffic datasets are loaded into the processing pipeline.

The system is designed to handle differences between dataset files instead of assuming that every file has exactly the same schema.

---

## Step 2 — Dataset Profiling

Before training models, we inspect the datasets.

Our profiling stage checks:

* Number of files
* Number of features
* Feature names
* Schema differences
* Available traffic information
* Dataset compatibility

This helps prevent incorrect assumptions during preprocessing.

---

## Step 3 — Feature Standardization

Different datasets can contain different column names and structures.

PhoenixOrder uses a **canonical feature schema** to define the features required by the system.

The canonical schema contains **44 canonical features** for common dataset representation. The trained multitask model uses **32 temporal features** from aggregated network states. Their exact order is stored in `multitask_delta_lstm_features.json`; the saved training scaler is `multitask_delta_lstm_scaler.joblib`.

This provides a common representation for downstream processing.

```text
Dataset A ──┐
            │
Dataset B ──┼──→ Canonical Feature Schema
            │
Dataset C ──┘
                    ↓
             Common Representation
```

---

# ⏱️ 6. Temporal Network States

This is one of the most important components of PhoenixOrder.

Raw network flows are transformed into **time-aware states**.

Instead of looking at a single network event, the system considers sequences of events:

```text
t1 → t2 → t3 → t4 → t5
```

Each time step represents the observed network behaviour during a particular period.

The current pipeline aggregates flows into **five-minute network states**. It builds a context from **12 consecutive states** (about one hour when the states are contiguous) and aligns the model target to the following state. Sequences do not bridge session gaps. Flow labels are retained as `attack_ratio`; the benchmark binary label marks a window as attack when `attack_ratio > 0`.

The multitask Delta-LSTM receives 12 states with 32 features per state. Its shared LSTM representation feeds two heads: a 32-value next-state delta predictor and a binary attack classifier. Training combines next-state delta regression loss with binary classification loss and uses a training-derived positive class weight. The scaler and feature order are saved and loaded for inference.

---

# 🤖 7. AI/ML Approach

PhoenixOrder is designed around temporal machine learning.

The architecture can use models such as:

### LSTM

LSTM networks are useful for learning patterns from sequential data.

```text
State 1
   ↓
State 2
   ↓
State 3
   ↓
State 4
   ↓
Prediction
```

### Transformer-based Temporal Models

Transformers can capture relationships between different points in a sequence and are a possible future approach for modelling attack progression.

Our development direction is to compare and improve temporal models based on the behaviour of the available data.

---

# 🛡️ 8. Attack Progression Prediction

The ultimate objective is not simply to classify traffic.

We want to forecast the **next likely attack stage**.

For example:

```text
Observed:
Reconnaissance → Initial Access → Execution

Prediction:
                  ↓
          Possible next stage
```

The current LSTM predicts attack probability and next-state feature delta; it does **not** predict the next attack stage. Current `attack_stage` is a **rule-based interpretation** of observed evidence and may be `Unknown` when the evidence is insufficient. Compatible rule mappings can attach MITRE ATT&CK context. The defense recommender provides advisory suggestions; autonomous traffic blocking is not implemented.

---

# 🗺️ 9. MITRE ATT&CK Integration

The current rule-based stage mapper attaches **MITRE ATT&CK** references for a limited set of compatible stages. This is not a learned technique classifier or verified incident attribution; broader mapping is future work.

Conceptually:

```text
Network Behaviour
       ↓
Detected Pattern
       ↓
Attack Technique
       ↓
MITRE ATT&CK Mapping
       ↓
Defensive Recommendation
```

This makes the prediction more understandable to cybersecurity analysts and helps connect machine-learning output with established security knowledge.

---

# 🔍 10. Explainability

A cybersecurity system should not simply say:

> "Attack predicted."

The defender should also understand **why** the system generated the prediction.

The offline inference output keeps three signals distinct: `attack_probability` is the neural classifier output; `behavioral_change_score` is the separate behavioral detector's network-state change signal; `attack_stage` is a rule-based interpretation, not an LSTM prediction. Significant feature changes and mapped context support analyst review. `Unknown` is valid when evidence is insufficient.

This is intended to make the system more useful for human analysts rather than acting as a completely opaque AI model.

---

# 🧪 11. Current Prototype Status

PhoenixOrder is currently a **working prototype under active development**, rather than a fully deployed production cybersecurity platform.

### Implemented / validated

* Dataset ingestion
* CSE-CIC-IDS2018 profiling
* CTU-13 dataset profiling
* Dataset schema comparison
* Canonical feature schema
* Feature availability analysis
* Initial preprocessing pipeline
* Project architecture
* Testing and validation of core modules
* Five-minute state aggregation and 12-state sequence construction
* Logistic Regression baseline and multitask Delta-LSTM benchmark
* Offline inference using the saved checkpoint, scaler and feature ordering
* Streamlit prototype for local CSV upload and result review
* Behavioral-change analysis, limited rule-based stage/MITRE mapping and defense suggestions

### Currently under development

The project is extending its temporal representation and investigating better ways to reduce false alarms and learn attack progression. The existing benchmark and model remain as documented; current implementation should not be interpreted as stage prediction.

### Future scope

PCAP ingestion, real-time streaming, learned attack-stage prediction, broader MITRE ATT&CK coverage, and analyst-approved response automation are planned. Autonomous traffic blocking is not implemented.

---

# 📈 12. Prototype Evaluation

We evaluate the system at multiple levels instead of relying on a single metric.

### Data-level evaluation

* Feature availability
* Schema consistency
* Dataset compatibility
* Data quality

### Software-level evaluation

* Automated tests
* Module validation
* Pipeline correctness

### Frozen ML benchmark

The benchmark uses six CSE-CIC-IDS2018 sessions for training, two for validation, and two held-out sessions for test. Validation data selects the multitask classification threshold; test labels are used only for final evaluation. Both models use the same 218 test windows and binary ground-truth labels (164 benign, 54 attack).

| Model | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2535 | 1.0000 | 0.4045 | 0.9695 |
| Multitask Delta-LSTM | 0.3051 | 1.0000 | 0.4675 | 0.7500 |

Multitask Delta-LSTM confusion matrix, ordered `[[TN, FP], [FN, TP]]`:

```text
[[41, 123],
 [ 0,  54]]
```

The FPR remains high: the multitask model flags 123 of 164 benign test windows. These results are prototype evidence on a limited dataset, not a production performance claim.

---

# ⚠️ 13. Current Limitations

Our current prototype has several limitations.

### 1. Offline datasets

The present system primarily works with recorded datasets rather than live enterprise network traffic.

### 2. Limited attack progression labels

Datasets do not always directly provide clean sequential labels representing every stage of an attack.

### 3. Cross-dataset differences

Different datasets contain different features, schemas and traffic characteristics.

### 4. High false-positive rate and limited stage coverage

The current benchmark produces many false alarms. The stage output is rule-based and supports only limited evidence mappings; the model does not forecast an attack stage.

### 5. Explainability and real-time deployment

PCAP ingestion and real-time deployment are planned extensions. Autonomous traffic blocking is not implemented.

---

# 🔮 14. Future Scope

Our future development is focused on extending the current prototype toward a complete predictive cyber-defence platform.

Planned technical directions include stronger temporal representation learning, Transformers, GNN/network-graph modelling, richer latent world models, multi-step future-state prediction, broader MITRE ATT&CK stage coverage, learned attack-stage transitions, PCAP ingestion, real-time streaming inference, SOC alert correlation, and analyst-approved response automation.

### Phase 1 — Data Foundation

```text
✓ Dataset Profiling
✓ Feature Availability
✓ Canonical Feature Schema
✓ Data Validation
```

### Phase 2 — Temporal Intelligence

```text
→ Temporal State Construction
→ LSTM / Transformer Models
→ Attack Progression Learning
→ Next-Stage Prediction
```

### Phase 3 — Cybersecurity Intelligence

```text
→ MITRE ATT&CK Mapping
→ Explainable Predictions
→ Attack Graph Representation
→ Analyst-Oriented Alerts
```

### Phase 4 — Proactive Defence

```text
→ Real-Time Network Monitoring
→ Continuous Prediction
→ Early Warning
→ Defensive Recommendations
→ Integration with Security Infrastructure
```

Our long-term vision is:

```text
Detect
  ↓
Understand
  ↓
Predict
  ↓
Explain
  ↓
Act
```

---

# 🏗️ 15. Project Structure

```text
PhoenixOrder_iitrpr/
│
├── config/
│   └── feature_schema.yaml
│
├── src/
│   ├── analysis/
│   ├── features/
│   ├── models/
│   ├── profiling/
│   ├── preprocessing/
│   └── temporal/
│
├── demo/
│   └── app.py
│
├── dashboard/
│   └── dashboard.py
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

The repository is organized into analysis, feature, model, profiling, preprocessing, and temporal modules, alongside the demo, dashboard, configuration, and reports. Dataset files and `*.pth` checkpoint binaries are intentionally excluded from Git.

---

# 💻 16. Setup

## Clone the repository

```bash
git clone <repository-url>
cd PhoenixOrder_iitrpr
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ 17. Running the Project

Run the local Streamlit prototype from the repository root:

```bash
streamlit run demo/app.py
```

After this command, the PhoenixOrder prototype opens in your browser. It accepts a CSE-CIC-IDS2018-style flow CSV and calls the offline inference pipeline. The matching saved checkpoint, scaler, and feature-order file must be available locally. No cloud API or key is required.

Dataset profiling is available for locally placed CSE-CIC-IDS2018 and CTU-13 files at the paths configured in `src/profiling/profile_datasets.py`:

```bash
python src/profiling/profile_datasets.py
```

Offline inference from a CSV is available as:

```bash
python -m src.analysis.run_offline_inference /path/to/flows.csv --output /path/to/predictions.csv
```

> **Note:** Large datasets are intentionally not included in this repository. Dataset download and placement instructions will be provided separately to keep the repository lightweight.

---

# 👥 18. Team

### Team: PhoenixOrder_iitrpr

**Smart India Hackathon 2026**

**Problem Statement:** SIH26153

Our team combines interests in:

* Artificial Intelligence / Machine Learning
* Data Science
* Cybersecurity
* Software Development
* Network Behaviour Analysis

---

# 🌟 19. Our Vision

Cyber defence should not only react to attacks after they occur.

It should understand how an attack is evolving and provide defenders with information early enough to respond.

**PhoenixOrder aims to build that capability.**

```text
        CURRENT SYSTEMS
              │
              ▼
        Detect Attacks
              │
              ▼
       PhoenixOrder
              │
              ▼
      Learn Behaviour
              │
              ▼
      Understand Sequence
              │
              ▼
       Predict Next Stage
              │
              ▼
       Enable Early Action
```

> **PhoenixOrder — From detecting attacks to anticipating their progression.**

---

## 📌 Project Status

**Prototype Status: Active Development — offline flow-CSV inference is implemented; not production-ready.**

The current implementation includes a saved multitask Delta-LSTM, an offline inference path, behavioral analysis, limited rule-based stage/MITRE interpretation, and advisory recommendations. Its false-positive rate is high. PCAP/live streaming, learned stage transitions, expanded ATT&CK coverage, and response automation remain roadmap items.

**From detecting attacks to anticipating their progression.**
