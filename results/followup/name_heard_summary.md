# Name heard in dialogue vs AD reference (REFRAMED, US AD, train+validation)

196 movies, 50200 AD lines, 53598 subtitle lines, labels {'NAME': 18353, 'PRN': 12801, 'DESC': 99}

## A. First AD naming vs first time the name is spoken

characters named in AD: 1458

- spoken before first AD naming: 35.3% (515)
- AD names before it is spoken: 35.8% (522)
- never spoken in subtitles: 28.9% (421)
- delay spoken -> first AD naming: median 65s (IQR 17-1346s)

## B. PRN vs explicit, 5-fold GroupKFold by movie

| features | rows | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |
| --- | --- | --- | --- | --- | --- |
| discourse | 31253 | 0.791 | 0.756 | 44.8 | 41.0 |
| discourse+heard | 31253 | 0.791 | 0.756 | 44.5 | 41.0 |
| discourse (name already spoken) | 21457 | 0.781 | 0.741 | 43.8 | 40.6 |
| discourse+heard (name already spoken) | 21457 | 0.782 | 0.742 | 43.6 | 40.6 |