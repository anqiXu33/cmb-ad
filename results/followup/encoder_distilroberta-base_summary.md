# Encoder fine-tuning (distilroberta-base), 15250 evaluated rows, 5-fold GroupKFold, folds run=5

> Transcribed from the Kaggle log (notebook version 3, run on code at 2ef1dba): the run finished but its push was rejected, so the .json and _oof.npy files of this run are not in the repo. Labels: text-only chains BEFORE the pronoun-label fix (4c4a53d) and without the syntactic-position features.
device=cuda, epochs=3, bs=16, lr=3e-05, max_len=192
| model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.751 | 0.456 | 0.751 | 0.544 | 0.000 | 30.6 |
| lgbm_feat | 0.802 | 0.598 | 0.805 | 0.533 | 0.387 | 17.8 |
| encoder (distilroberta-base) | 0.823 | 0.506 | 0.828 | 0.633 | 0.000 | 22.8 |
| stack_lgbm (features + distilroberta-base) | 0.858 | 0.646 | 0.861 | 0.703 | 0.328 | 22.7 |
gold PRN% = 24.0
## Per dataset
| dataset | n | model | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| cmdad | 5752 | rule | 0.775 | 0.634 | 30.8 | 30.8 |
| cmdad | 5752 | lgbm_feat | 0.786 | 0.641 | 28.8 | 30.8 |
| cmdad | 5752 | encoder | 0.762 | 0.591 | 27.5 | 30.8 |
| cmdad | 5752 | stack_lgbm | 0.809 | 0.688 | 30.2 | 30.8 |
| madeval | 5739 | rule | 0.694 | 0.411 | 33.9 | 18.0 |
| madeval | 5739 | lgbm_feat | 0.808 | 0.212 | 6.4 | 18.0 |
| madeval | 5739 | encoder | 0.902 | 0.741 | 20.1 | 18.0 |
| madeval | 5739 | stack_lgbm | 0.915 | 0.757 | 17.0 | 18.0 |
| tvad | 3759 | rule | 0.802 | 0.586 | 25.2 | 22.7 |
| tvad | 3759 | lgbm_feat | 0.828 | 0.582 | 18.4 | 22.7 |
| tvad | 3759 | encoder | 0.819 | 0.572 | 19.7 | 22.7 |
| tvad | 3759 | stack_lgbm | 0.859 | 0.670 | 20.1 | 22.7 |
