# Fine-tuned LLM vs rule vs learned models (PRONOUN vs explicit)

model: Qwen/Qwen2.5-1.5B 4-bit QLoRA; 10 previous AD lines, target name masked; protocol lodo, splits run: 3 of 3; epochs 1; lr 0.0001; batch 4x4; max_len 512.
finetuned@prior: threshold set per fold so the predicted pronoun rate equals the training fold's rate.
Reference models: out-of-fold predictions under the same protocol (lodo) on the same rows.

| subset | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 13990 | 15.0 | rule | 0.792 | 0.514 | 27.9 |
| overall | 13990 | 15.0 | lgbm+role (lodo) | 0.847 | 0.393 | 10.2 |
| overall | 13990 | 15.0 | stack+role (lodo) | 0.849 | 0.401 | 10.2 |
| overall | 13990 | 15.0 | finetuned@0.5 | 0.799 | 0.269 | 12.5 |
| overall | 13990 | 15.0 | finetuned@prior | 0.802 | 0.346 | 15.2 |
| subject | 8668 | 13.1 | rule | 0.770 | 0.460 | 29.5 |
| subject | 8668 | 13.1 | lgbm+role (lodo) | 0.847 | 0.219 | 6.5 |
| subject | 8668 | 13.1 | stack+role (lodo) | 0.850 | 0.223 | 6.2 |
| subject | 8668 | 13.1 | finetuned@0.5 | 0.781 | 0.047 | 9.9 |
| subject | 8668 | 13.1 | finetuned@prior | 0.800 | 0.178 | 11.3 |
| object | 773 | 22.8 | rule | 0.854 | 0.699 | 25.7 |
| object | 773 | 22.8 | lgbm+role (lodo) | 0.824 | 0.558 | 17.1 |
| object | 773 | 22.8 | stack+role (lodo) | 0.820 | 0.567 | 18.8 |
| object | 773 | 22.8 | finetuned@0.5 | 0.806 | 0.436 | 11.6 |
| object | 773 | 22.8 | finetuned@prior | 0.784 | 0.507 | 21.1 |
| possessive | 1172 | 48.2 | rule | 0.774 | 0.750 | 42.2 |
| possessive | 1172 | 48.2 | lgbm+role (lodo) | 0.590 | 0.574 | 48.0 |
| possessive | 1172 | 48.2 | stack+role (lodo) | 0.591 | 0.579 | 49.0 |
| possessive | 1172 | 48.2 | finetuned@0.5 | 0.711 | 0.675 | 40.8 |
| possessive | 1172 | 48.2 | finetuned@prior | 0.657 | 0.647 | 48.9 |
| prep. object | 1454 | 14.9 | rule | 0.869 | 0.629 | 20.5 |
| prep. object | 1454 | 14.9 | lgbm+role (lodo) | 0.863 | 0.494 | 12.1 |
| prep. object | 1454 | 14.9 | stack+role (lodo) | 0.867 | 0.506 | 12.0 |
| prep. object | 1454 | 14.9 | finetuned@0.5 | 0.845 | 0.347 | 8.9 |
| prep. object | 1454 | 14.9 | finetuned@prior | 0.816 | 0.397 | 15.5 |
| other | 1923 | 0.2 | rule | 0.817 | 0.006 | 18.2 |
| other | 1923 | 0.2 | lgbm+role (lodo) | 0.998 | 0.000 | 0.0 |
| other | 1923 | 0.2 | stack+role (lodo) | 0.998 | 0.000 | 0.0 |
| other | 1923 | 0.2 | finetuned@0.5 | 0.898 | 0.000 | 10.0 |
| other | 1923 | 0.2 | finetuned@prior | 0.900 | 0.010 | 10.0 |
| dataset cmdad | 5114 | 19.7 | rule | 0.827 | 0.630 | 27.1 |
| dataset cmdad | 5114 | 19.7 | lgbm+role (lodo) | 0.800 | 0.362 | 11.6 |
| dataset cmdad | 5114 | 19.7 | stack+role (lodo) | 0.801 | 0.367 | 11.8 |
| dataset cmdad | 5114 | 19.7 | finetuned@0.5 | 0.811 | 0.287 | 6.7 |
| dataset cmdad | 5114 | 19.7 | finetuned@prior | 0.805 | 0.391 | 12.3 |
| dataset madeval | 5322 | 8.9 | rule | 0.725 | 0.313 | 31.1 |
| dataset madeval | 5322 | 8.9 | lgbm+role (lodo) | 0.875 | 0.281 | 8.5 |
| dataset madeval | 5322 | 8.9 | stack+role (lodo) | 0.882 | 0.307 | 8.1 |
| dataset madeval | 5322 | 8.9 | finetuned@0.5 | 0.764 | 0.240 | 22.1 |
| dataset madeval | 5322 | 8.9 | finetuned@prior | 0.788 | 0.234 | 18.7 |
| dataset tvad | 3554 | 17.2 | rule | 0.841 | 0.614 | 24.1 |
| dataset tvad | 3554 | 17.2 | lgbm+role (lodo) | 0.872 | 0.545 | 10.9 |
| dataset tvad | 3554 | 17.2 | stack+role (lodo) | 0.869 | 0.538 | 11.2 |
| dataset tvad | 3554 | 17.2 | finetuned@0.5 | 0.833 | 0.297 | 6.4 |
| dataset tvad | 3554 | 17.2 | finetuned@prior | 0.820 | 0.427 | 14.2 |