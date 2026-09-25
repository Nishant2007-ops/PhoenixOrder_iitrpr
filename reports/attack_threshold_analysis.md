# Attack-ratio threshold analysis

## Scope and ground-truth rule

This is an offline diagnostic. No LSTM training was run. Ground-truth positives for each row use `attack_ratio >= label threshold`; the existing benchmark rule is `attack_ratio > 0`. Empty time bins are excluded as targets because they have no observed flows. Test ratios below are descriptive only and were not used for any cutoff choice.

The benchmark's exact LSTM weights were trained in memory and were not saved. The score distributions below use the available saved checkpoint `multisession_delta_lstm_model.pth`. Its score scale may differ from the ephemeral model used for the benchmark CSV; the old benchmark's validation/test confusion counts are separately quoted below.

## Attack-ratio distributions

Validation windows: 134; test windows: 218.

| Attack-ratio range | Validation count | Validation share | Test count | Test share |
|---|---:|---:|---:|---:|
| attack_ratio == 0 | 107 | 79.9% | 164 | 75.2% |
| 0 < attack_ratio <= 0.01 | 1 | 0.7% | 0 | 0.0% |
| 0.01 < attack_ratio <= 0.05 | 0 | 0.0% | 0 | 0.0% |
| 0.05 < attack_ratio <= 0.10 | 7 | 5.2% | 0 | 0.0% |
| 0.10 < attack_ratio <= 0.25 | 5 | 3.7% | 4 | 1.8% |
| attack_ratio > 0.25 | 14 | 10.4% | 50 | 22.9% |

## Validation class distributions and score operating points

For each candidate ground-truth definition, the diagnostic score cutoff is the highest validation cutoff that retains at least 80% recall. This makes the recall requirement explicit and then minimizes false positives among cutoffs meeting it; it does not select the cutoff by highest F1. The 80% target is an illustrative operating constraint, not a universal optimum.

| Label positive rule | Benign | Attack | Score cutoff | Precision | Recall | FPR | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attack_ratio >= 0.01 | 108 | 26 | 0.033638173 | 0.350 | 0.808 | 0.361 | 0.488 | 69 | 39 | 5 | 21 |
| attack_ratio >= 0.05 | 108 | 26 | 0.033638173 | 0.350 | 0.808 | 0.361 | 0.488 | 69 | 39 | 5 | 21 |
| attack_ratio >= 0.10 | 115 | 19 | 0.033638173 | 0.267 | 0.842 | 0.383 | 0.405 | 71 | 44 | 3 | 16 |
| attack_ratio >= 0.25 | 120 | 14 | 0.037550371 | 0.222 | 0.857 | 0.350 | 0.353 | 78 | 42 | 2 | 12 |

## Delta-LSTM behavioral-score distribution (saved checkpoint)

Benign/attack here uses the existing rule `attack_ratio > 0`.

| Group | Windows | Mean | Std | Min | Q25 | Median | Q75 | Q90 | Q95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 107 | 0.154454 | 0.211703 | 0.0196071 | 0.0243502 | 0.0290849 | 0.280716 | 0.427002 | 0.616332 | 0.786934 |
| Attack | 27 | 0.166378 | 0.192115 | 0.0268768 | 0.0358392 | 0.0396949 | 0.315718 | 0.39192 | 0.589264 | 0.655307 |

## Is the current behavioral-score cutoff predicting nearly all windows as attacks?

The previous benchmark selected score cutoff `0.04545918` on validation. Its saved benchmark confusion matrix reports 106/134 validation windows predicted attack (79.1%) and 209/218 test windows predicted attack (95.9%). Thus it did not classify nearly all validation windows as attacks, but it did classify nearly all test windows that way.

Applied to the available saved checkpoint (not the exact ephemeral benchmark weights), the same numeric cutoff predicts 46/134 validation windows as attacks (34.3%); precision=0.217, recall=0.370, FPR=0.336. A score cutoff is model-specific, so this checkpoint-specific result is diagnostic rather than a replacement benchmark.

## Interpretation and tradeoffs

Raising the attack-ratio label cutoff removes low-intensity mixed windows from the positive class and changes both class balance and the meaning of a false positive. A score cutoff selected for high recall accepts more false positives; raising it generally improves precision and FPR at the cost of missed attacks. The validation table makes that operating choice visible. The prior test FPR cannot be fixed by relabeling test windows after seeing its result; any chosen label rule and score cutoff must be fixed from domain requirements and validation before a future test evaluation.

No test label was used for threshold selection.
