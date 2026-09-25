# Saved multitask Delta-LSTM threshold sensitivity

No training was performed. The saved checkpoint, scaler, and feature order were loaded without modification. The current threshold is 0.39090657.

Threshold choices and approximate recall operating points below are derived from validation scores only. Test labels are used only after cutoffs are frozen, to calculate test evaluation metrics.

Validation: 134 windows (107 benign, 27 attack). Test: 218 windows (164 benign, 54 attack).

## Validation threshold sweep

The complete set of attainable score cutoffs from unique validation probabilities plus the current threshold is saved in the CSV. This summary includes the current threshold and the threshold nearest each requested recall target.

| Operating point | Threshold | Precision | Recall | F1 | FPR | Predicted attacks |
|---|---:|---:|---:|---:|---:|---:|
| Current | 0.39090657 | 0.2651 | 0.8148 | 0.4000 | 0.5701 | 83 |
| Nearest 80% recall | 0.3909065723 | 0.2651 | 0.8148 | 0.4000 | 0.5701 | 83 |
| Nearest 85% recall | 0.3325196803 | 0.2500 | 0.8519 | 0.3866 | 0.6449 | 92 |
| Nearest 90% recall | 0.2909415662 | 0.2474 | 0.8889 | 0.3871 | 0.6822 | 97 |
| Nearest 95% recall | 0.173660174 | 0.2453 | 0.9630 | 0.3910 | 0.7477 | 106 |
| Nearest 100% recall | 0.07791650295 | 0.2288 | 1.0000 | 0.3724 | 0.8505 | 118 |

## Test evaluation using fixed validation-derived thresholds

These cutoffs were fixed before evaluating test labels. Test metrics did not affect threshold selection.

| Operating point | Fixed threshold | Precision | Recall | F1 | FPR | Predicted attacks |
|---|---:|---:|---:|---:|---:|---:|
| Current | 0.39090657 | 0.3051 | 1.0000 | 0.4675 | 0.7500 | 177 |
| Nearest 80% recall | 0.3909065723 | 0.3051 | 1.0000 | 0.4675 | 0.7500 | 177 |
| Nearest 85% recall | 0.3325196803 | 0.3051 | 1.0000 | 0.4675 | 0.7500 | 177 |
| Nearest 90% recall | 0.2909415662 | 0.2967 | 1.0000 | 0.4576 | 0.7805 | 182 |
| Nearest 95% recall | 0.173660174 | 0.2857 | 1.0000 | 0.4444 | 0.8232 | 189 |
| Nearest 100% recall | 0.07791650295 | 0.2571 | 1.0000 | 0.4091 | 0.9512 | 210 |

The CSV contains each validation-derived cutoff evaluated on both splits, along with confusion-matrix counts. The test split is labeled `test evaluation`; no threshold was selected from its results.
