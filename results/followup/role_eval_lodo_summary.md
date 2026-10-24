# Syntactic position: does it help, and where is it hard? (text-only chains, leave one dataset out: each dataset scored by models trained on the other two)

| position | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| all | 13990 | 15.0 | rule | 0.792 | 0.514 | 27.9 |
| all | 13990 | 15.0 | lgbm | 0.852 | 0.302 | 6.1 |
| all | 13990 | 15.0 | lgbm+role | 0.847 | 0.393 | 10.2 |
| all | 13990 | 15.0 | stack | 0.850 | 0.337 | 7.7 |
| all | 13990 | 15.0 | stack+role | 0.846 | 0.405 | 10.9 |
| subject | 8668 | 13.1 | rule | 0.770 | 0.460 | 29.5 |
| subject | 8668 | 13.1 | lgbm | 0.866 | 0.301 | 6.1 |
| subject | 8668 | 13.1 | lgbm+role | 0.847 | 0.219 | 6.5 |
| subject | 8668 | 13.1 | stack | 0.856 | 0.285 | 7.0 |
| subject | 8668 | 13.1 | stack+role | 0.843 | 0.221 | 7.1 |
| object | 773 | 22.8 | rule | 0.854 | 0.699 | 25.7 |
| object | 773 | 22.8 | lgbm | 0.796 | 0.319 | 7.2 |
| object | 773 | 22.8 | lgbm+role | 0.824 | 0.558 | 17.1 |
| object | 773 | 22.8 | stack | 0.811 | 0.421 | 9.8 |
| object | 773 | 22.8 | stack+role | 0.829 | 0.593 | 19.1 |
| possessive | 1172 | 48.2 | rule | 0.774 | 0.750 | 42.2 |
| possessive | 1172 | 48.2 | lgbm | 0.601 | 0.316 | 10.2 |
| possessive | 1172 | 48.2 | lgbm+role | 0.590 | 0.574 | 48.0 |
| possessive | 1172 | 48.2 | stack | 0.634 | 0.441 | 17.2 |
| possessive | 1172 | 48.2 | stack+role | 0.598 | 0.592 | 50.2 |
| prep. object | 1454 | 14.9 | rule | 0.869 | 0.629 | 20.5 |
| prep. object | 1454 | 14.9 | lgbm | 0.860 | 0.334 | 6.1 |
| prep. object | 1454 | 14.9 | lgbm+role | 0.863 | 0.494 | 12.1 |
| prep. object | 1454 | 14.9 | stack | 0.866 | 0.396 | 7.3 |
| prep. object | 1454 | 14.9 | stack+role | 0.869 | 0.521 | 12.5 |
| other | 1923 | 0.2 | rule | 0.817 | 0.006 | 18.2 |
| other | 1923 | 0.2 | lgbm | 0.962 | 0.000 | 3.6 |
| other | 1923 | 0.2 | lgbm+role | 0.998 | 0.000 | 0.0 |
| other | 1923 | 0.2 | stack | 0.957 | 0.000 | 4.1 |
| other | 1923 | 0.2 | stack+role | 0.998 | 0.000 | 0.0 |
| ds:cmdad | 5114 | 19.7 | rule | 0.827 | 0.630 | 27.1 |
| ds:cmdad | 5114 | 19.7 | lgbm | 0.813 | 0.165 | 2.6 |
| ds:cmdad | 5114 | 19.7 | lgbm+role | 0.800 | 0.362 | 11.6 |
| ds:cmdad | 5114 | 19.7 | stack | 0.816 | 0.244 | 4.6 |
| ds:cmdad | 5114 | 19.7 | stack+role | 0.803 | 0.373 | 11.8 |
| ds:madeval | 5322 | 8.9 | rule | 0.725 | 0.313 | 31.1 |
| ds:madeval | 5322 | 8.9 | lgbm | 0.885 | 0.221 | 5.8 |
| ds:madeval | 5322 | 8.9 | lgbm+role | 0.875 | 0.281 | 8.5 |
| ds:madeval | 5322 | 8.9 | stack | 0.873 | 0.260 | 8.2 |
| ds:madeval | 5322 | 8.9 | stack+role | 0.871 | 0.303 | 9.5 |
| ds:tvad | 3554 | 17.2 | rule | 0.841 | 0.614 | 24.1 |
| ds:tvad | 3554 | 17.2 | lgbm | 0.860 | 0.516 | 11.7 |
| ds:tvad | 3554 | 17.2 | lgbm+role | 0.872 | 0.545 | 10.9 |
| ds:tvad | 3554 | 17.2 | stack | 0.864 | 0.521 | 11.3 |
| ds:tvad | 3554 | 17.2 | stack+role | 0.869 | 0.550 | 11.9 |