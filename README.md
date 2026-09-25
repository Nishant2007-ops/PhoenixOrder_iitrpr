# PhoenixOrder_iitrpr — SIH 2026 | SIH26153

**AI-driven proactive cyber defence using temporal network behaviour and world models**

PhoenixOrder is a research prototype for turning flow-level network telemetry into short-horizon temporal states, identifying suspicious behaviour, and presenting evidence that can help an analyst investigate. It combines a multitask Delta-LSTM with a separate behavioral-change analysis and rule-based attack-stage interpretation. The current primary input is a CSE-CIC-IDS2018-style flow CSV.

> **Prototype status:** this is not a production intrusion-detection system (IDS). On the frozen 218-window test set, the multitask model has a 75.00% false-positive rate. The results are reported as measured; this README does not claim production readiness.

## 1. Project overview

The project explores a world-model-inspired workflow for proactive cyber defence: encode recent network behaviour as a sequence of states, predict the next-state change, and jointly estimate whether the next window is an attack. Separate analysis components describe behavioral changes, map evidence to a coarse attack stage, and offer defensive recommendations for human review.

The saved inference path is designed to run locally and offline. It does not send traffic or data to cloud APIs.

## 2. Problem being solved

Flow records are numerous and individually hard to interpret. Aggregating them into time-ordered states can expose changes in volume, direction, packet timing, and TCP flags. PhoenixOrder investigates whether a temporal model can use those patterns to flag attack windows early enough to support investigation.

The project evaluates a limited labeled benchmark. It does not establish effectiveness on unseen networks, generalize to all attack families, or replace analyst judgment.

## 3. Key features

- CSE-CIC-IDS2018-style flow CSV preprocessing and label normalization.
- Five-minute aggregation into network-behavior states.
- Twelve-state temporal contexts (about one hour where states are contiguous).
- A multitask Delta-LSTM for next-state delta prediction and attack classification.
- A Logistic Regression baseline using the benchmark's temporal features.
- Independent behavioral-change analysis, rule-based stage mapping, MITRE ATT&CK references, and deterministic defense recommendations.
- A Streamlit demonstration and reusable offline inference pipeline.
- Validation-based threshold selection and a frozen test evaluation.

## 4. End-to-end architecture

```text
CSE-CIC-IDS2018-style flow CSV
              |
              v
  normalize fields and labels
              |
              v
  aggregate flows into 5-minute states
              |
              v
  order states and split at session gaps
              |
              v
  12-state context (12 x 32 features)
              |
              v
       saved feature scaler
              |
              v
       shared LSTM encoder
          /             \
         v               v
  next-state delta   attack classifier
     prediction       probability/score
         |               |
         +-------+-------+
                 v
  behavioral change, stage interpretation,
  MITRE context, and defense recommendations
                 |
                 v
        local Streamlit / CSV output
```

The classifier score, behavioral-change score, and interpreted attack stage are separate outputs; see [Behavioral change analysis](#11-behavioural-change-analysis) and [Attack-stage interpretation](#12-attack-stage-interpretation).

## 5. Repository structure

The tree below reflects the current repository files. Dataset files and model binaries are local artifacts and are intentionally not listed as source files.

```text
.
├── config/
│   └── feature_schema.yaml
├── dashboard/
│   └── dashboard.py
├── demo/
│   └── app.py
├── reports/
│   ├── attack_detection_benchmark.csv
│   ├── attack_detection_comparison_with_multitask.csv
│   ├── attack_threshold_analysis.md
│   ├── final_benchmark.csv
│   └── multitask_threshold_sensitivity.md
├── src/
│   ├── analysis/       # offline inference and result/timeline utilities
│   ├── features/       # behavior, transitions, stages, defense suggestions
│   ├── models/         # LSTM, training, evaluation, benchmark scripts
│   ├── preprocessing/  # CSE/CTU normalization and schema checks
│   ├── profiling/      # feature and dataset profiling utilities
│   └── temporal/       # state aggregation, session building, sequences
├── .gitignore
├── README.md
└── requirements.txt
```

The Python directories contain the implementation files corresponding to their responsibilities. Large datasets (`datasets/`, CSV, PCAP, and related archive formats) and `*.pth` model binaries are ignored by Git. The local saved model assets used by inference are `multitask_delta_lstm_checkpoint.pth`, `multitask_delta_lstm_scaler.joblib`, and `multitask_delta_lstm_features.json`.

## 6. Preprocessing

The CSE normalization code in `src/preprocessing/normalize_cse.py` standardizes the dataset's flow columns for the downstream pipeline, parses timestamps and numeric values, handles duration units and derived flow quantities, and normalizes attack labels through the label-mapping utilities. Schema validation and canonicalization helpers are in `src/preprocessing/`.

The inference path uses the same project preprocessing and temporal feature construction as the model pipeline. Model features are selected in the saved ordering, transformed with the saved scaler, and are not re-fit during inference. The label/statistical columns used to form evaluation targets are not classifier input features.

## 7. Temporal aggregation into five-minute network states

Flows are grouped into five-minute time bins by the temporal aggregation code (`src/temporal/aggregate_states.py`, `src/temporal/build_multisession_dataset.py`). Each state summarizes activity such as packet and byte totals/means, directional traffic, packet and inter-arrival timing, TCP flag counts, and flow counts. States are time ordered and session boundaries are respected so sequences do not bridge large gaps.

Ground-truth `attack_ratio` is derived from the original flow labels within each time bin: the fraction of flows labeled as attacks. The benchmark's binary target treats a window with `attack_ratio > 0` as an attack. Retaining the ratio preserves the distinction between a window with a small attack fraction and one dominated by attack flows, even though the reported benchmark uses a binary target.

## 8. Twelve-state sequence construction

The model consumes twelve consecutive five-minute states, approximately one hour of recent context when the states are contiguous. The sequence builder constructs a context from the prior twelve states and aligns its targets with the following state. Sequences are built within their session/segment rather than across discontinuities.

```text
state[t-11] ... state[t]  ──> LSTM context (12 states)
                                      |
                                      +──> predict delta to state[t+1]
                                      +──> classify state[t+1] as attack/benign
```

## 9. Multitask Delta-LSTM architecture

The multitask model in `src/models/lstm_model.py` keeps a shared LSTM representation and branches into two heads. Its input has 12 time steps and 32 feature values per step. The next-state head predicts a 32-dimensional change; the attack head produces a binary-classification logit.

```text
12 x 32 temporal input
         |
       LSTM
         |
 shared final hidden representation
       /                 \
      v                   v
  delta head          attack head
  32 outputs           1 logit
```

## 10. Next-state delta prediction and attack classification

The delta objective predicts the difference between the next state's scaled feature vector and the current state's vector. The classification objective predicts the binary attack label aligned to that next state. Training combines a delta regression loss (MSE) and a binary classification loss (BCE with logits); the positive-class weight is computed from training labels only to address class imbalance. Validation data is used for model selection and for the saved classification threshold. Test labels are reserved for final evaluation.

## 11. Behavioural change analysis

`behavioral_change_score` is produced by the separate behavioral detector from transition/change evidence between network states. It summarizes how unusual or significant the state change appears under that detector's rules. The score is **not** the attack probability and is not the classifier head output. Significant feature changes are surfaced as supporting evidence for review.

## 12. Attack-stage interpretation

`attack_stage` is a rule-based interpretation of behavioral evidence and available flow context. It is **not directly predicted by the LSTM**. The mapper can return **Unknown** when the evidence is insufficient; an unknown stage is preferable to asserting a stage without support.

## 13. MITRE ATT&CK mapping

Where the current rule-based stage mapper has a compatible mapping, the interface can show associated MITRE ATT&CK tactic/technique context. This is an evidence-linked reference, not a learned ATT&CK classifier and not proof that a technique occurred. Mapping coverage is limited and may be empty for an unknown or unsupported stage.

## 14. Defensive recommendations

The defense recommender provides deterministic, rule-driven suggestions based on the interpreted evidence. Recommendations are for analyst consideration. The system does not automatically block traffic, change firewall rules, quarantine hosts, or execute a response action.

## 15. Streamlit demo

`demo/app.py` provides a local interface for uploading a CSE-CIC-IDS2018-style CSV and viewing inference results, the probability timeline, and flagged-window evidence. The demo calls the shared inference pipeline instead of defining a second model path. Run it from the repository root:

```bash
streamlit run demo/app.py --server.address 127.0.0.1
```

The uploaded file is processed locally. No cloud service or API key is required. Keep the saved model assets available at the paths expected by the inference module.

## 16. Offline inference

`src/analysis/run_offline_inference.py` loads the saved checkpoint, scaler, feature ordering, and checkpoint threshold. It processes a local flow CSV and emits one record per valid prediction, including timestamp, classifier probability and class, behavioral-change information, significant feature changes, interpreted stage, MITRE context, and defense recommendations.

The 32 input features are listed in `multitask_delta_lstm_features.json` and that file defines their exact order:

```text
flow_duration_mean, flow_duration_std, total_packets_sum, total_packets_mean,
total_bytes_sum, total_bytes_mean, forward_packets_sum, forward_packets_mean,
backward_packets_sum, backward_packets_mean, forward_bytes_sum, forward_bytes_mean,
backward_bytes_sum, backward_bytes_mean, bytes_per_packet_mean,
packet_length_mean_mean, packet_length_mean_std, flow_iat_mean_mean,
flow_iat_std_mean, forward_iat_mean_mean, backward_iat_mean_mean, syn_count_sum,
ack_count_sum, fin_count_sum, rst_count_sum, psh_count_sum, urg_count_sum,
down_up_ratio_mean, forward_packet_ratio_mean, active_mean_mean, idle_mean_mean,
flow_count
```

`multitask_delta_lstm_scaler.joblib` is the saved training scaler. Inference loads it; it does not fit a new scaler. The `.pth` checkpoint stores the trained network and validation-selected threshold. Current input is flow CSV; PCAP ingestion is a future extension. Inference is offline and does not require cloud APIs.

## 17. Benchmark methodology

The frozen benchmark uses date-separated CSE-CIC-IDS2018 sessions:

- **Train:** Friday-02-03-2018, Thursday-01-03-2018, Wednesday-28-02-2018, Friday-23-02-2018, Thursday-22-02-2018, Wednesday-21-02-2018.
- **Validation:** Friday-16-02-2018, Thursday-15-02-2018.
- **Test:** Wednesday-14-02-2018, Thuesday-20-02-2018.

The split and model feature construction are shared by the benchmarked models. The Logistic Regression baseline uses the benchmark temporal states/features with class balancing. The multitask model uses the saved checkpoint. Its attack threshold was selected using validation data (saved threshold approximately `0.39090657`) and then frozen before test evaluation. The test set is used only for the final reported metrics, not for fitting, tuning, or threshold choice.

Evaluation compares both classifiers over the same 218 test windows and the same binary ground truth. False-positive rate is `FP / (FP + TN)`. Reports and the reproducible runner are under `reports/` and `src/models/`.

## 18. Exact benchmark results

Frozen test results (positive class: attack; 218 windows total):

| Model | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2535 | 1.0000 | 0.4045 | 0.9695 |
| Multitask Delta-LSTM | 0.3051 | 1.0000 | 0.4675 | 0.7500 |

Test class distribution: **164 benign**, **54 attack**. The multitask confusion matrix in `[[TN, FP], [FN, TP]]` order is:

```text
[[ 41, 123],
 [  0,  54]]
```

The high false-positive rate means many benign windows are flagged. Perfect recall on this test set does not make the model operationally reliable; these are prototype results on a limited benchmark, not a production IDS guarantee.

## 19. Explainability

The demo presents distinct, inspectable outputs:

- **`attack_probability`** — the neural-network classifier's sigmoid probability for the attack class.
- **`behavioral_change_score`** — the behavioral detector's network-state change signal.
- **`attack_stage`** — a rule-based interpretation from available evidence; it can be `Unknown`.
- Significant feature changes, mapped tactic/technique context, and suggested defenses provide additional investigation context.

These signals have different meanings and should not be treated as interchangeable confidence values or causal explanations.

## 20. Limitations

- The measured false-positive rates are high: 96.95% for Logistic Regression and 75.00% for the multitask Delta-LSTM on the current test split.
- The benchmark covers a limited collection of sessions and attack/benign windows from one dataset; traffic from other networks may differ substantially.
- A binary label marks a window as attack when its flow-level `attack_ratio` is nonzero, including windows with only a small attack fraction. This target choice affects class labels and metrics.
- Five-minute aggregation loses within-window ordering and detail; twelve states are only approximately one hour and may be shorter near session boundaries.
- Stage and MITRE outputs are rule-based with limited coverage, not model predictions or verified incident attribution.
- Recommendations are advisory only. There is no autonomous traffic blocking or response automation.
- PCAP ingestion and live streaming are not implemented; the current primary input is a flow CSV.
- Performance numbers should be interpreted as prototype benchmark evidence, not as a security or production guarantee.

## 21. Future roadmap

- Stronger temporal representation learning.
- Transformer-based sequence models.
- GNN and network-graph modelling.
- Richer latent world models.
- Multi-step future-state prediction.
- Broader MITRE ATT&CK stage coverage.
- Learned attack-stage transitions.
- PCAP ingestion.
- Real-time streaming inference.
- SOC alert correlation.
- Analyst-approved response automation.

## 22. Reproducibility

Reproducible inference depends on the exact saved checkpoint, scaler, and ordered feature list. The benchmark runner and split definitions are retained in the repository; the final benchmark result is saved separately. The threshold is selected from validation data and kept fixed for test scoring. Training/benchmark reproduction additionally requires obtaining the original CSE-CIC-IDS2018 data files locally.

Large dataset files and `*.pth` model binaries are intentionally excluded from Git, so a fresh clone may not contain everything needed for inference or training. Keep the checkpoint, scaler, and feature-order JSON from the same model release together. Do not replace the scaler or feature ordering independently.

## 23. Team and project information

- **Project:** PhoenixOrder_iitrpr
- **Event:** Smart India Hackathon 2026
- **Problem statement:** SIH26153
- **Theme:** AI-driven proactive cyber defence using temporal network behaviour and world models
- **Institution/team:** IIT Ropar project (team and contributor details can be added here by the project owners).

## 24. Quick start

From the repository root, create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the matching saved model assets where `src/analysis/run_offline_inference.py` expects them, then run the local demo:

```bash
streamlit run demo/app.py --server.address 127.0.0.1
```

For inference from the command line with a local flow CSV:

```bash
python -m src.analysis.run_offline_inference /path/to/flows.csv --output /path/to/predictions.csv
```

Training and benchmark reproduction require the CSE-CIC-IDS2018 input files locally; datasets are not bundled in this repository. No command in the demo needs cloud access. **Do not interpret this prototype as an autonomous or production IDS.**
