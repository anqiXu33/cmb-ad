# Learned selector vs rule planner (text-only chains silver labels)

## protocol_A_pooled_cv

| split | n | model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 15653 | majority | 0.731 | 0.282 | 0.776 | 0.000 | 0.0 | 22.4 |
| ALL | 15653 | rule | 0.733 |  | 0.733 | 0.513 | 32.5 | 22.4 |
| ALL | 15653 | lr | 0.623 | 0.488 | 0.747 | 0.495 | 27.7 | 22.4 |
| ALL | 15653 | lgbm | 0.771 | 0.463 | 0.802 | 0.509 | 17.9 | 22.4 |
| cmdad | 5888 | majority | 0.658 | 0.264 | 0.712 | 0.000 | 0.0 | 28.8 |
| cmdad | 5888 | rule | 0.750 |  | 0.750 | 0.598 | 33.5 | 28.8 |
| cmdad | 5888 | lr | 0.590 | 0.489 | 0.724 | 0.505 | 27.1 | 28.8 |
| cmdad | 5888 | lgbm | 0.742 | 0.486 | 0.770 | 0.604 | 29.1 | 28.8 |
| madeval | 5950 | majority | 0.787 | 0.294 | 0.839 | 0.000 | 0.0 | 16.1 |
| madeval | 5950 | rule | 0.675 |  | 0.675 | 0.370 | 35.4 | 16.1 |
| madeval | 5950 | lr | 0.585 | 0.442 | 0.765 | 0.364 | 20.9 | 16.1 |
| madeval | 5950 | lgbm | 0.773 | 0.353 | 0.818 | 0.174 | 5.9 | 16.1 |
| tvad | 3815 | majority | 0.758 | 0.288 | 0.777 | 0.000 | 0.0 | 22.3 |
| tvad | 3815 | rule | 0.796 |  | 0.796 | 0.582 | 26.4 | 22.3 |
| tvad | 3815 | lr | 0.732 | 0.490 | 0.755 | 0.601 | 39.3 | 22.3 |
| tvad | 3815 | lgbm | 0.814 | 0.489 | 0.826 | 0.582 | 19.3 | 22.3 |

## protocol_B_leave_one_dataset_out

| split | n | model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmdad | 5888 | majority | 0.658 | 0.264 | 0.712 | 0.000 | 0.0 | 28.8 |
| cmdad | 5888 | rule | 0.750 |  | 0.750 | 0.598 | 33.5 | 28.8 |
| cmdad | 5888 | lr | 0.503 | 0.403 | 0.711 | 0.290 | 11.9 | 28.8 |
| cmdad | 5888 | lgbm | 0.667 | 0.395 | 0.721 | 0.263 | 9.0 | 28.8 |
| madeval | 5950 | majority | 0.787 | 0.294 | 0.839 | 0.000 | 0.0 | 16.1 |
| madeval | 5950 | rule | 0.675 |  | 0.675 | 0.370 | 35.4 | 16.1 |
| madeval | 5950 | lr | 0.727 | 0.387 | 0.831 | 0.075 | 2.2 | 16.1 |
| madeval | 5950 | lgbm | 0.706 | 0.388 | 0.740 | 0.333 | 22.9 | 16.1 |
| tvad | 3815 | majority | 0.758 | 0.288 | 0.777 | 0.000 | 0.0 | 22.3 |
| tvad | 3815 | rule | 0.796 |  | 0.796 | 0.582 | 26.4 | 22.3 |
| tvad | 3815 | lr | 0.728 | 0.481 | 0.743 | 0.591 | 40.5 | 22.3 |
| tvad | 3815 | lgbm | 0.813 | 0.479 | 0.826 | 0.551 | 16.4 | 22.3 |

## LightGBM feature importance (top 12, mean split count over folds)

- ad_dur: 6575
- gap_sec: 5393
- name_ratio_prior: 3526
- n_prior_mentions: 3350
- n_prior_name: 3158
- log_gap_sec: 2682
- n_prior_prn: 2424
- consecutive: 1426
- shot_scale: 1143
- gap_ads: 871
- n_same_gender_last3: 796
- thread_switch: 743