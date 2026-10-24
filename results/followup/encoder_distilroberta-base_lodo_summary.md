# Encoder fine-tuning (distilroberta-base_lodo), 15250 evaluated rows, leave-one-dataset-out

> Transcribed from the Kaggle log (notebook version 3, run on code at 2ef1dba): the run finished but its push was rejected, so the .json and _oof.npy files of this run are not in the repo. Labels: text-only chains BEFORE the pronoun-label fix (4c4a53d) and without the syntactic-position features.
device=cuda, epochs=3, bs=16, lr=3e-05, max_len=192
| model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.751 | 0.456 | 0.751 | 0.544 | 0.000 | 30.6 |
| lgbm_feat | 0.768 | 0.465 | 0.774 | 0.451 | 0.091 | 17.2 |
| encoder (distilroberta-base_lodo) | 0.736 | 0.435 | 0.741 | 0.483 | 0.000 | 26.2 |
| stack_lgbm (features + distilroberta-base_lodo) | 0.773 | 0.461 | 0.779 | 0.461 | 0.065 | 17.0 |
gold PRN% = 24.0
## Per dataset (each scored by models trained on the other two)
| dataset | n | model | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| cmdad | 5752 | rule | 0.775 | 0.634 | 30.8 | 30.8 |
| cmdad | 5752 | lgbm_feat | 0.738 | 0.460 | 17.7 | 30.8 |
| cmdad | 5752 | encoder | 0.743 | 0.503 | 21.0 | 30.8 |
| cmdad | 5752 | stack_lgbm | 0.748 | 0.424 | 12.9 | 30.8 |
| madeval | 5739 | rule | 0.694 | 0.411 | 33.9 | 18.0 |
| madeval | 5739 | lgbm_feat | 0.778 | 0.370 | 17.3 | 18.0 |
| madeval | 5739 | encoder | 0.700 | 0.448 | 36.4 | 18.0 |
| madeval | 5739 | stack_lgbm | 0.776 | 0.427 | 21.0 | 18.0 |
| tvad | 3759 | rule | 0.802 | 0.586 | 25.2 | 22.7 |
| tvad | 3759 | lgbm_feat | 0.824 | 0.546 | 16.1 | 22.7 |
| tvad | 3759 | encoder | 0.799 | 0.513 | 18.5 | 22.7 |
| tvad | 3759 | stack_lgbm | 0.829 | 0.572 | 17.3 | 22.7 |
