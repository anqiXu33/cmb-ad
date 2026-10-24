# Zero-shot LLM vs rule vs learned models (PRONOUN vs explicit)

model: Qwen/Qwen2.5-7B-Instruct 4-bit; 10 previous AD lines with names visible; 3000 random rows of 13990.
zeroshot@prior threshold = 0.912 (pronoun rate of the unscored rows = 15.0%).
zeroshot@otherds: per dataset, predicted pronoun rate = pronoun rate of the other two datasets.
Reference models: (cv) = 5-fold out-of-fold, (lodo) = trained on the other two datasets; same rows.

| subset | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 3000 | 15.1 | rule | 0.788 | 0.512 | 28.3 |
| overall | 3000 | 15.1 | lgbm+role (cv) | 0.896 | 0.633 | 13.4 |
| overall | 3000 | 15.1 | stack+role (cv) | 0.897 | 0.642 | 13.6 |
| overall | 3000 | 15.1 | lgbm+role (lodo) | 0.845 | 0.385 | 10.2 |
| overall | 3000 | 15.1 | stack+role (lodo) | 0.845 | 0.382 | 10.0 |
| overall | 3000 | 15.1 | zeroshot@0.5 | 0.530 | 0.289 | 51.0 |
| overall | 3000 | 15.1 | zeroshot@prior | 0.777 | 0.259 | 15.0 |
| overall | 3000 | 15.1 | zeroshot@otherds | 0.777 | 0.257 | 15.0 |
| subject | 1880 | 14.1 | rule | 0.771 | 0.482 | 30.1 |
| subject | 1880 | 14.1 | lgbm+role (cv) | 0.890 | 0.573 | 11.5 |
| subject | 1880 | 14.1 | stack+role (cv) | 0.898 | 0.608 | 12.0 |
| subject | 1880 | 14.1 | lgbm+role (lodo) | 0.839 | 0.241 | 7.1 |
| subject | 1880 | 14.1 | stack+role (lodo) | 0.843 | 0.242 | 6.6 |
| subject | 1880 | 14.1 | zeroshot@0.5 | 0.523 | 0.255 | 49.8 |
| subject | 1880 | 14.1 | zeroshot@prior | 0.791 | 0.228 | 13.0 |
| subject | 1880 | 14.1 | zeroshot@otherds | 0.790 | 0.230 | 13.2 |
| object | 152 | 19.1 | rule | 0.868 | 0.688 | 23.0 |
| object | 152 | 19.1 | lgbm+role (cv) | 0.816 | 0.517 | 19.1 |
| object | 152 | 19.1 | stack+role (cv) | 0.803 | 0.483 | 19.1 |
| object | 152 | 19.1 | lgbm+role (lodo) | 0.829 | 0.552 | 19.1 |
| object | 152 | 19.1 | stack+role (lodo) | 0.829 | 0.567 | 20.4 |
| object | 152 | 19.1 | zeroshot@0.5 | 0.480 | 0.336 | 59.2 |
| object | 152 | 19.1 | zeroshot@prior | 0.678 | 0.290 | 26.3 |
| object | 152 | 19.1 | zeroshot@otherds | 0.691 | 0.277 | 23.7 |
| possessive | 245 | 46.1 | rule | 0.767 | 0.730 | 40.0 |
| possessive | 245 | 46.1 | lgbm+role (cv) | 0.829 | 0.817 | 47.8 |
| possessive | 245 | 46.1 | stack+role (cv) | 0.812 | 0.802 | 48.6 |
| possessive | 245 | 46.1 | lgbm+role (lodo) | 0.596 | 0.556 | 44.9 |
| possessive | 245 | 46.1 | stack+role (lodo) | 0.584 | 0.549 | 46.1 |
| possessive | 245 | 46.1 | zeroshot@0.5 | 0.580 | 0.623 | 65.3 |
| possessive | 245 | 46.1 | zeroshot@prior | 0.584 | 0.407 | 24.1 |
| possessive | 245 | 46.1 | zeroshot@otherds | 0.588 | 0.423 | 25.3 |
| prep. object | 305 | 14.1 | rule | 0.872 | 0.629 | 20.3 |
| prep. object | 305 | 14.1 | lgbm+role (cv) | 0.885 | 0.568 | 12.5 |
| prep. object | 305 | 14.1 | stack+role (cv) | 0.875 | 0.513 | 11.5 |
| prep. object | 305 | 14.1 | lgbm+role (lodo) | 0.879 | 0.519 | 11.1 |
| prep. object | 305 | 14.1 | stack+role (lodo) | 0.869 | 0.474 | 10.8 |
| prep. object | 305 | 14.1 | zeroshot@0.5 | 0.554 | 0.284 | 48.2 |
| prep. object | 305 | 14.1 | zeroshot@prior | 0.784 | 0.298 | 16.7 |
| prep. object | 305 | 14.1 | zeroshot@otherds | 0.777 | 0.244 | 15.4 |
| other | 418 | 0.5 | rule | 0.785 | 0.022 | 21.5 |
| other | 418 | 0.5 | lgbm+role (cv) | 0.995 | 0.000 | 0.0 |
| other | 418 | 0.5 | stack+role (cv) | 0.995 | 0.000 | 0.0 |
| other | 418 | 0.5 | lgbm+role (lodo) | 0.995 | 0.000 | 0.0 |
| other | 418 | 0.5 | stack+role (lodo) | 0.995 | 0.000 | 0.0 |
| other | 418 | 0.5 | zeroshot@0.5 | 0.533 | 0.010 | 46.7 |
| other | 418 | 0.5 | zeroshot@prior | 0.861 | 0.000 | 13.4 |
| other | 418 | 0.5 | zeroshot@otherds | 0.859 | 0.000 | 13.6 |
| dataset cmdad | 1151 | 19.4 | rule | 0.829 | 0.628 | 26.6 |
| dataset cmdad | 1151 | 19.4 | lgbm+role (cv) | 0.850 | 0.591 | 17.4 |
| dataset cmdad | 1151 | 19.4 | stack+role (cv) | 0.848 | 0.594 | 18.1 |
| dataset cmdad | 1151 | 19.4 | lgbm+role (lodo) | 0.804 | 0.369 | 11.7 |
| dataset cmdad | 1151 | 19.4 | stack+role (lodo) | 0.805 | 0.370 | 11.6 |
| dataset cmdad | 1151 | 19.4 | zeroshot@0.5 | 0.499 | 0.339 | 56.5 |
| dataset cmdad | 1151 | 19.4 | zeroshot@prior | 0.736 | 0.306 | 18.7 |
| dataset cmdad | 1151 | 19.4 | zeroshot@otherds | 0.772 | 0.280 | 12.3 |
| dataset madeval | 1089 | 8.7 | rule | 0.709 | 0.306 | 33.2 |
| dataset madeval | 1089 | 8.7 | lgbm+role (cv) | 0.964 | 0.780 | 7.5 |
| dataset madeval | 1089 | 8.7 | stack+role (cv) | 0.961 | 0.761 | 7.4 |
| dataset madeval | 1089 | 8.7 | lgbm+role (lodo) | 0.873 | 0.242 | 8.0 |
| dataset madeval | 1089 | 8.7 | stack+role (lodo) | 0.879 | 0.250 | 7.4 |
| dataset madeval | 1089 | 8.7 | zeroshot@0.5 | 0.500 | 0.207 | 54.4 |
| dataset madeval | 1089 | 8.7 | zeroshot@prior | 0.806 | 0.216 | 16.0 |
| dataset madeval | 1089 | 8.7 | zeroshot@otherds | 0.788 | 0.217 | 18.4 |
| dataset tvad | 760 | 17.6 | rule | 0.839 | 0.614 | 23.9 |
| dataset tvad | 760 | 17.6 | lgbm+role (cv) | 0.867 | 0.601 | 15.7 |
| dataset tvad | 760 | 17.6 | stack+role (cv) | 0.880 | 0.640 | 15.7 |
| dataset tvad | 760 | 17.6 | lgbm+role (lodo) | 0.866 | 0.532 | 11.1 |
| dataset tvad | 760 | 17.6 | stack+role (lodo) | 0.858 | 0.509 | 11.3 |
| dataset tvad | 760 | 17.6 | zeroshot@0.5 | 0.622 | 0.318 | 37.8 |
| dataset tvad | 760 | 17.6 | zeroshot@prior | 0.799 | 0.215 | 8.0 |
| dataset tvad | 760 | 17.6 | zeroshot@otherds | 0.767 | 0.272 | 14.3 |