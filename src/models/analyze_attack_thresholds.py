"""Validation-only diagnostic for attack-ratio and Delta-LSTM score cutoffs."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.models.benchmark_attack_detection import (
    DEVICE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    SEQUENCE_LENGTH,
    _behavior_scores,
    _get_files,
)
from src.models.lstm_model import NetworkStateLSTM
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import prepare_multisession_delta_lstm_data


CHECKPOINT = Path("multisession_delta_lstm_model.pth")
REPORT = Path("reports/attack_threshold_analysis.md")
CURRENT_SCORE_THRESHOLD = 0.04545917953664258
LABEL_THRESHOLDS = (0.01, 0.05, 0.10, 0.25)


def ratio_bins(ratios):
    return {
        "attack_ratio == 0": int(np.sum(ratios == 0)),
        "0 < attack_ratio <= 0.01": int(np.sum((ratios > 0) & (ratios <= 0.01))),
        "0.01 < attack_ratio <= 0.05": int(np.sum((ratios > 0.01) & (ratios <= 0.05))),
        "0.05 < attack_ratio <= 0.10": int(np.sum((ratios > 0.05) & (ratios <= 0.10))),
        "0.10 < attack_ratio <= 0.25": int(np.sum((ratios > 0.10) & (ratios <= 0.25))),
        "attack_ratio > 0.25": int(np.sum(ratios > 0.25)),
    }


def metrics(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "predicted_attack": int(pred.sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def validation_cutoff(y, scores, minimum_recall=0.80):
    """Highest score cutoff retaining at least the stated validation recall."""
    candidates = np.unique(np.concatenate((
        [np.nextafter(np.min(scores), -np.inf)], scores
    )))
    eligible = [t for t in candidates
                if recall_score(y, scores >= t, zero_division=0) >= minimum_recall]
    return float(max(eligible))


def fmt_distribution(ratios):
    values = ratio_bins(ratios)
    total = len(ratios)
    return "\n".join(
        f"| {name} | {count} | {count / total:.1%} |"
        for name, count in values.items()
    )


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Saved checkpoint not found: {CHECKPOINT}")

    train_files, validation_files, test_files = _get_files()
    train, validation, test = build_multisession_states(
        train_files, validation_files, test_files
    )
    (_, _, _, _, _, _, feature_columns, scaler) = prepare_multisession_delta_lstm_data(
        train, validation, test, sequence_length=SEQUENCE_LENGTH
    )
    model = NetworkStateLSTM(
        input_size=len(feature_columns),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(feature_columns),
    ).to(DEVICE)
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    validation_scores, validation_y_any, validation_ratios = _behavior_scores(
        model, validation, scaler, feature_columns
    )
    _, _, test_ratios = _behavior_scores(
        model, test, scaler, feature_columns
    )
    if len(validation_scores) != len(validation_ratios):
        raise RuntimeError("Validation score and label windows do not align.")

    lines = [
        "# Attack-ratio threshold analysis",
        "",
        "## Scope and ground-truth rule",
        "",
        "This is an offline diagnostic. No LSTM training was run. Ground-truth positives "
        "for each row use `attack_ratio >= label threshold`; the existing benchmark rule "
        "is `attack_ratio > 0`. Empty time bins are excluded as targets because they have "
        "no observed flows. Test ratios below are descriptive only and were not used for "
        "any cutoff choice.",
        "",
        f"The benchmark's exact LSTM weights were trained in memory and were not saved. "
        f"The score distributions below use the available saved checkpoint `{CHECKPOINT}`. "
        f"Its score scale may differ from the ephemeral model used for the benchmark CSV; "
        f"the old benchmark's validation/test confusion counts are separately quoted below.",
        "",
        "## Attack-ratio distributions",
        "",
        f"Validation windows: {len(validation_ratios)}; test windows: {len(test_ratios)}.",
        "",
        "| Attack-ratio range | Validation count | Validation share | Test count | Test share |",
        "|---|---:|---:|---:|---:|",
    ]
    val_bins, test_bins = ratio_bins(validation_ratios), ratio_bins(test_ratios)
    for label in val_bins:
        lines.append(
            f"| {label} | {val_bins[label]} | {val_bins[label] / len(validation_ratios):.1%} "
            f"| {test_bins[label]} | {test_bins[label] / len(test_ratios):.1%} |"
        )

    lines.extend([
        "",
        "## Validation class distributions and score operating points",
        "",
        "For each candidate ground-truth definition, the diagnostic score cutoff is the "
        "highest validation cutoff that retains at least 80% recall. This makes the "
        "recall requirement explicit and then minimizes false positives among cutoffs "
        "meeting it; it does not select the cutoff by highest F1. The 80% target is an "
        "illustrative operating constraint, not a universal optimum.",
        "",
        "| Label positive rule | Benign | Attack | Score cutoff | Precision | Recall | FPR | F1 | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label_threshold in LABEL_THRESHOLDS:
        y = (validation_ratios >= label_threshold).astype(np.int8)
        cutoff = validation_cutoff(y, validation_scores)
        result = metrics(y, validation_scores, cutoff)
        lines.append(
            f"| attack_ratio >= {label_threshold:.2f} | {(y == 0).sum()} | {(y == 1).sum()} "
            f"| {cutoff:.8g} | {result['precision']:.3f} | {result['recall']:.3f} "
            f"| {result['fpr']:.3f} | {result['f1']:.3f} | {result['tn']} | {result['fp']} "
            f"| {result['fn']} | {result['tp']} |"
        )

    benign_scores = validation_scores[validation_y_any == 0]
    attack_scores = validation_scores[validation_y_any == 1]
    lines.extend([
        "",
        "## Delta-LSTM behavioral-score distribution (saved checkpoint)",
        "",
        "Benign/attack here uses the existing rule `attack_ratio > 0`.",
        "",
        "| Group | Windows | Mean | Std | Min | Q25 | Median | Q75 | Q90 | Q95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, values in (("Benign", benign_scores), ("Attack", attack_scores)):
        q = np.quantile(values, [0.25, 0.5, 0.75, 0.90, 0.95])
        lines.append(
            f"| {group} | {len(values)} | {values.mean():.6g} | {values.std():.6g} "
            f"| {values.min():.6g} | {q[0]:.6g} | {q[1]:.6g} | {q[2]:.6g} "
            f"| {q[3]:.6g} | {q[4]:.6g} | {values.max():.6g} |"
        )

    old_validation = metrics(validation_y_any, validation_scores, CURRENT_SCORE_THRESHOLD)
    lines.extend([
        "",
        "## Is the current behavioral-score cutoff predicting nearly all windows as attacks?",
        "",
        f"The previous benchmark selected score cutoff `{CURRENT_SCORE_THRESHOLD:.8g}` on "
        f"validation. Its saved benchmark confusion matrix reports {24 + 82}/134 "
        f"validation windows predicted attack (79.1%) and {49 + 160}/218 test windows "
        f"predicted attack (95.9%). Thus it did not classify nearly all validation windows "
        f"as attacks, but it did classify nearly all test windows that way.",
        "",
        f"Applied to the available saved checkpoint (not the exact ephemeral benchmark "
        f"weights), the same numeric cutoff predicts {old_validation['predicted_attack']}/"
        f"{len(validation_scores)} validation windows as attacks "
        f"({old_validation['predicted_attack'] / len(validation_scores):.1%}); "
        f"precision={old_validation['precision']:.3f}, recall={old_validation['recall']:.3f}, "
        f"FPR={old_validation['fpr']:.3f}. A score cutoff is model-specific, so this "
        f"checkpoint-specific result is diagnostic rather than a replacement benchmark.",
        "",
        "## Interpretation and tradeoffs",
        "",
        "Raising the attack-ratio label cutoff removes low-intensity mixed windows from "
        "the positive class and changes both class balance and the meaning of a false "
        "positive. A score cutoff selected for high recall accepts more false positives; "
        "raising it generally improves precision and FPR at the cost of missed attacks. "
        "The validation table makes that operating choice visible. The prior test FPR "
        "cannot be fixed by relabeling test windows after seeing its result; any chosen "
        "label rule and score cutoff must be fixed from domain requirements and validation "
        "before a future test evaluation.",
        "",
        "No test label was used for threshold selection.",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {REPORT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
