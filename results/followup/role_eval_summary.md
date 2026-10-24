# Syntactic position: does it help, and where is it hard? (text-only chains, 5-fold by sequence)

| position | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |
| --- | --- | --- | --- | --- | --- | --- |
| all | 15250 | 24.0 | rule | 0.751 | 0.544 | 30.6 |
| all | 15250 | 24.0 | lgbm | 0.805 | 0.534 | 17.8 |
| all | 15250 | 24.0 | lgbm+role | 0.872 | 0.730 | 23.4 |
| all | 15250 | 24.0 | stack | 0.828 | 0.618 | 21.0 |
| all | 15250 | 24.0 | stack+role | 0.876 | 0.739 | 23.4 |
| subject | 9077 | 18.7 | rule | 0.737 | 0.475 | 31.2 |
| subject | 9077 | 18.7 | lgbm | 0.849 | 0.597 | 18.8 |
| subject | 9077 | 18.7 | lgbm+role | 0.861 | 0.603 | 16.3 |
| subject | 9077 | 18.7 | stack | 0.849 | 0.596 | 18.6 |
| subject | 9077 | 18.7 | stack+role | 0.866 | 0.616 | 16.2 |
| object | 895 | 36.5 | rule | 0.793 | 0.693 | 30.8 |
| object | 895 | 36.5 | lgbm | 0.728 | 0.501 | 17.9 |
| object | 895 | 36.5 | lgbm+role | 0.799 | 0.739 | 40.6 |
| object | 895 | 36.5 | stack | 0.758 | 0.608 | 25.3 |
| object | 895 | 36.5 | stack+role | 0.794 | 0.725 | 38.3 |
| possessive | 1838 | 68.6 | rule | 0.669 | 0.709 | 45.2 |
| possessive | 1838 | 68.6 | lgbm | 0.502 | 0.468 | 25.1 |
| possessive | 1838 | 68.6 | lgbm+role | 0.839 | 0.888 | 74.9 |
| possessive | 1838 | 68.6 | stack | 0.658 | 0.700 | 45.2 |
| possessive | 1838 | 68.6 | stack+role | 0.850 | 0.896 | 75.6 |
| prep. object | 1554 | 23.2 | rule | 0.833 | 0.641 | 23.2 |
| prep. object | 1554 | 23.2 | lgbm | 0.819 | 0.520 | 14.7 |
| prep. object | 1554 | 23.2 | lgbm+role | 0.866 | 0.705 | 22.3 |
| prep. object | 1554 | 23.2 | stack | 0.847 | 0.633 | 18.5 |
| prep. object | 1554 | 23.2 | stack+role | 0.869 | 0.721 | 23.8 |
| other | 1886 | 0.4 | rule | 0.810 | 0.022 | 19.0 |
| other | 1886 | 0.4 | lgbm | 0.916 | 0.036 | 8.3 |
| other | 1886 | 0.4 | lgbm+role | 0.996 | 0.000 | 0.0 |
| other | 1886 | 0.4 | stack | 0.909 | 0.044 | 9.1 |
| other | 1886 | 0.4 | stack+role | 0.996 | 0.000 | 0.0 |