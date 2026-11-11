# Baseline vs Transformer comparison

Fill this table using measured outputs from Task 3 and `transformer_metrics.json`.
Do not copy example metrics from the assessment brief as if they were measured.

| Model | Accuracy | Macro F1 | Training time (s) | Inference time/example (s) |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | Record Task 3 output | Record Task 3 output | Measure Task 3 fit time | Measure prediction time |
| Fine-tuned DistilBERT | `eval_accuracy` | `eval_macro_f1` | `training_time_seconds` | `inference_time_seconds_per_example` |

Note: Task 3's generated feedback corpus and Task 4's SST-2 subset are different datasets. For a scientifically fair comparison, fine-tune DistilBERT on the same customer-feedback dataset and same train/test split before making a direct performance claim.
