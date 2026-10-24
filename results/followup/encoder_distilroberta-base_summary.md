# Encoder fine-tuning (distilroberta-base), 13990 evaluated rows, 5-fold GroupKFold, folds run=5

device=cuda, epochs=3, bs=16, lr=3e-05, max_len=192

| model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.792 | 0.459 | 0.792 | 0.514 | 0.000 | 27.9 |
| lgbm_feat | 0.898 | 0.686 | 0.901 | 0.652 | 0.464 | 13.4 |
| encoder (distilroberta-base) | 0.867 | 0.490 | 0.873 | 0.549 | 0.000 | 13.2 |
| stack_lgbm (features + distilroberta-base) | 0.907 | 0.704 | 0.910 | 0.682 | 0.484 | 13.3 |

gold PRN% = 15.0

## Per dataset

| dataset | n | model | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| cmdad | 5114 | rule | 0.827 | 0.630 | 27.1 | 19.7 |
| cmdad | 5114 | lgbm_feat | 0.853 | 0.609 | 17.8 | 19.7 |
| cmdad | 5114 | encoder | 0.828 | 0.506 | 15.1 | 19.7 |
| cmdad | 5114 | stack_lgbm | 0.870 | 0.656 | 18.0 | 19.7 |
| madeval | 5322 | rule | 0.725 | 0.313 | 31.1 | 8.9 |
| madeval | 5322 | lgbm_feat | 0.966 | 0.797 | 8.0 | 8.9 |
| madeval | 5322 | encoder | 0.929 | 0.626 | 10.0 | 8.9 |
| madeval | 5322 | stack_lgbm | 0.966 | 0.796 | 7.8 | 8.9 |
| tvad | 3554 | rule | 0.841 | 0.614 | 24.1 | 17.2 |
| tvad | 3554 | lgbm_feat | 0.874 | 0.612 | 15.3 | 17.2 |
| tvad | 3554 | encoder | 0.853 | 0.547 | 15.2 | 17.2 |
| tvad | 3554 | stack_lgbm | 0.884 | 0.639 | 14.7 | 17.2 |