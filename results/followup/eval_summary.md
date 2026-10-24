# Learned selector vs rule planner (English silver labels)

## protocol_A_pooled_cv

| split | n | model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 11446 | majority | 0.682 | 0.270 | 0.723 | 0.000 | 0.0 | 27.7 |
| ALL | 11446 | rule | 0.657 |  | 0.657 | 0.385 | 28.0 | 27.7 |
| ALL | 11446 | lr | 0.547 | 0.429 | 0.702 | 0.444 | 25.9 | 27.7 |
| ALL | 11446 | lgbm | 0.736 | 0.448 | 0.764 | 0.496 | 19.2 | 27.7 |
| cmdad | 3738 | majority | 0.522 | 0.229 | 0.581 | 0.000 | 0.0 | 41.9 |
| cmdad | 3738 | rule | 0.623 |  | 0.623 | 0.463 | 28.3 | 41.9 |
| cmdad | 3738 | lr | 0.413 | 0.380 | 0.609 | 0.463 | 30.8 | 41.9 |
| cmdad | 3738 | lgbm | 0.630 | 0.437 | 0.661 | 0.574 | 37.7 | 41.9 |
| madeval | 4780 | majority | 0.786 | 0.293 | 0.826 | 0.000 | 0.0 | 17.4 |
| madeval | 4780 | rule | 0.647 |  | 0.647 | 0.268 | 30.8 | 17.4 |
| madeval | 4780 | lr | 0.583 | 0.405 | 0.781 | 0.330 | 15.3 | 17.4 |
| madeval | 4780 | lgbm | 0.805 | 0.414 | 0.839 | 0.350 | 7.4 | 17.4 |
| tvad | 2928 | majority | 0.718 | 0.279 | 0.737 | 0.000 | 0.0 | 26.3 |
| tvad | 2928 | rule | 0.718 |  | 0.718 | 0.431 | 23.2 | 26.3 |
| tvad | 2928 | lr | 0.659 | 0.460 | 0.692 | 0.513 | 37.0 | 26.3 |
| tvad | 2928 | lgbm | 0.756 | 0.431 | 0.772 | 0.444 | 14.7 | 26.3 |

## protocol_B_leave_one_dataset_out

| split | n | model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmdad | 3738 | majority | 0.522 | 0.229 | 0.581 | 0.000 | 0.0 | 41.9 |
| cmdad | 3738 | rule | 0.623 |  | 0.623 | 0.463 | 28.3 | 41.9 |
| cmdad | 3738 | lr | 0.299 | 0.288 | 0.601 | 0.230 | 9.9 | 41.9 |
| cmdad | 3738 | lgbm | 0.558 | 0.399 | 0.648 | 0.375 | 14.3 | 41.9 |
| madeval | 4780 | majority | 0.786 | 0.293 | 0.826 | 0.000 | 0.0 | 17.4 |
| madeval | 4780 | rule | 0.647 |  | 0.647 | 0.268 | 30.8 | 17.4 |
| madeval | 4780 | lr | 0.745 | 0.458 | 0.839 | 0.325 | 6.4 | 17.4 |
| madeval | 4780 | lgbm | 0.759 | 0.392 | 0.791 | 0.317 | 13.2 | 17.4 |
| tvad | 2928 | majority | 0.718 | 0.279 | 0.737 | 0.000 | 0.0 | 26.3 |
| tvad | 2928 | rule | 0.718 |  | 0.718 | 0.431 | 23.2 | 26.3 |
| tvad | 2928 | lr | 0.573 | 0.406 | 0.583 | 0.497 | 56.5 | 26.3 |
| tvad | 2928 | lgbm | 0.732 | 0.408 | 0.745 | 0.391 | 15.5 | 26.3 |

## LightGBM feature importance (top 12, mean split count over folds)

- ad_dur: 6349
- gap_sec: 4412
- n_prior_appear: 3603
- name_ratio_prior: 3444
- n_prior_name: 2491
- log_gap_sec: 2330
- n_prior_prn: 2041
- consecutive: 1431
- gap_ads: 1241
- shot_scale: 1175
- n_chars_cur: 1114
- n_chars_prev: 1051