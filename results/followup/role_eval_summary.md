# Syntactic position: does it help, and where is it hard? (text-only chains, 5-fold by sequence)

| position | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| all | 13990 | 15.0 | rule | 0.792 | 0.514 | 27.9 |
| all | 13990 | 15.0 | lgbm | 0.878 | 0.515 | 10.2 |
| all | 13990 | 15.0 | lgbm+role | 0.903 | 0.656 | 13.3 |
| all | 13990 | 15.0 | stack | 0.879 | 0.530 | 10.7 |
| all | 13990 | 15.0 | stack+role | 0.903 | 0.659 | 13.6 |
| subject | 8668 | 13.1 | rule | 0.770 | 0.460 | 29.5 |
| subject | 8668 | 13.1 | lgbm | 0.901 | 0.594 | 11.3 |
| subject | 8668 | 13.1 | lgbm+role | 0.902 | 0.595 | 11.1 |
| subject | 8668 | 13.1 | stack | 0.897 | 0.573 | 10.9 |
| subject | 8668 | 13.1 | stack+role | 0.901 | 0.591 | 11.0 |
| object | 773 | 22.8 | rule | 0.854 | 0.699 | 25.7 |
| object | 773 | 22.8 | lgbm | 0.824 | 0.469 | 10.3 |
| object | 773 | 22.8 | lgbm+role | 0.832 | 0.604 | 19.7 |
| object | 773 | 22.8 | stack | 0.828 | 0.513 | 12.5 |
| object | 773 | 22.8 | stack+role | 0.842 | 0.637 | 20.7 |
| possessive | 1172 | 48.2 | rule | 0.774 | 0.750 | 42.2 |
| possessive | 1172 | 48.2 | lgbm | 0.620 | 0.406 | 15.7 |
| possessive | 1172 | 48.2 | lgbm+role | 0.824 | 0.818 | 48.4 |
| possessive | 1172 | 48.2 | stack | 0.654 | 0.499 | 20.8 |
| possessive | 1172 | 48.2 | stack+role | 0.825 | 0.823 | 50.5 |
| prep. object | 1454 | 14.9 | rule | 0.869 | 0.629 | 20.5 |
| prep. object | 1454 | 14.9 | lgbm | 0.867 | 0.417 | 7.8 |
| prep. object | 1454 | 14.9 | lgbm+role | 0.882 | 0.559 | 11.9 |
| prep. object | 1454 | 14.9 | stack | 0.872 | 0.475 | 9.4 |
| prep. object | 1454 | 14.9 | stack+role | 0.878 | 0.567 | 13.2 |
| other | 1923 | 0.2 | rule | 0.817 | 0.006 | 18.2 |
| other | 1923 | 0.2 | lgbm | 0.959 | 0.025 | 4.1 |
| other | 1923 | 0.2 | lgbm+role | 0.998 | 0.000 | 0.0 |
| other | 1923 | 0.2 | stack | 0.961 | 0.026 | 3.8 |
| other | 1923 | 0.2 | stack+role | 0.998 | 0.000 | 0.0 |