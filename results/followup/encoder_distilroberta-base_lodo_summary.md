# Encoder fine-tuning (distilroberta-base_lodo), 13990 evaluated rows, leave-one-dataset-out

device=cuda, epochs=3, bs=16, lr=3e-05, max_len=192

| model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.792 | 0.459 | 0.792 | 0.514 | 0.000 | 27.9 |
| lgbm_feat | 0.840 | 0.453 | 0.847 | 0.393 | 0.058 | 10.2 |
| encoder (distilroberta-base_lodo) | 0.790 | 0.420 | 0.796 | 0.386 | 0.000 | 18.2 |
| stack_lgbm (features + distilroberta-base_lodo) | 0.822 | 0.478 | 0.837 | 0.357 | 0.181 | 10.4 |

gold PRN% = 15.0

## Per dataset (each scored by models trained on the other two)

| dataset | n | model | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| cmdad | 5114 | rule | 0.827 | 0.630 | 27.1 | 19.7 |
| cmdad | 5114 | lgbm_feat | 0.800 | 0.362 | 11.6 | 19.7 |
| cmdad | 5114 | encoder | 0.818 | 0.486 | 15.8 | 19.7 |
| cmdad | 5114 | stack_lgbm | 0.804 | 0.369 | 11.3 | 19.7 |
| madeval | 5322 | rule | 0.725 | 0.313 | 31.1 | 8.9 |
| madeval | 5322 | lgbm_feat | 0.875 | 0.281 | 8.5 | 8.9 |
| madeval | 5322 | encoder | 0.746 | 0.260 | 25.3 | 8.9 |
| madeval | 5322 | stack_lgbm | 0.851 | 0.223 | 10.3 | 8.9 |
| tvad | 3554 | rule | 0.841 | 0.614 | 24.1 | 17.2 |
| tvad | 3554 | lgbm_feat | 0.872 | 0.545 | 10.9 | 17.2 |
| tvad | 3554 | encoder | 0.841 | 0.436 | 11.0 | 17.2 |
| tvad | 3554 | stack_lgbm | 0.862 | 0.481 | 9.3 | 17.2 |