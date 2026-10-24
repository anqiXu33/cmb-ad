# Encoder fine-tuning (distilroberta-base), 15653 evaluated rows, folds=5

device=cuda, epochs=3, bs=16, lr=3e-05, max_len=192

| model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.733 | 0.444 | 0.733 | 0.513 | 0.000 | 32.5 |
| lgbm_feat | 0.769 | 0.460 | 0.800 | 0.501 | 0.016 | 17.7 |
| encoder (distilroberta-base) | 0.791 | 0.553 | 0.822 | 0.604 | 0.186 | 22.6 |
| stack_lgbm (features + distilroberta-base) | 0.824 | 0.582 | 0.854 | 0.669 | 0.185 | 21.7 |

gold PRN% = 22.4