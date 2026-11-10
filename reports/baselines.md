# Step 2: time split and baselines (SYNTHETIC data)

> All data is synthetic. These numbers validate the pipeline, not real-world performance.

## Split (by booking time)

| part | bookings | from | to | cold_start_share |
|---|---|---|---|---|
| train | 17710 | 2023-09-05 | 2025-03-31 | 0.972 |
| val | 4585 | 2025-04-01 | 2025-06-30 | 0.949 |
| test | 6430 | 2025-07-01 | 2025-12-28 | 0.929 |

Train covers 17,710 baskets. Evaluated carts: val 4,579.

## Tuning

Profile-popularity smoothing m, chosen by validation map@5: m=5: 0.3980, m=20: 0.3983, m=50: 0.3981, m=200: 0.3977 -> **m=20**.

## Val: all services

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.648 | 0.280 | 0.781 | 0.410 | 0.317 |
| profile_popularity | 0.763 | 0.353 | 0.848 | 0.479 | 0.398 |
| co_occurrence | 0.677 | 0.294 | 0.793 | 0.417 | 0.325 |

## Val: excluding $0 items furniture_dolly, hand_truck_dolly

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.656 | 0.313 | 0.768 | 0.445 | 0.340 |
| profile_popularity | 0.755 | 0.381 | 0.831 | 0.497 | 0.416 |
| co_occurrence | 0.663 | 0.317 | 0.783 | 0.454 | 0.344 |

### Val by visible cart size (map@5, all services)

| cart size | carts | global_popularity | profile_popularity | co_occurrence |
|---|---|---|---|---|
| 0 (empty) | 766 | 0.327 | 0.419 | 0.327 |
| 1-2 | 1462 | 0.317 | 0.408 | 0.327 |
| 3-5 | 1528 | 0.312 | 0.387 | 0.322 |
| 6+ | 823 | 0.316 | 0.384 | 0.324 |

_Test set sealed. It is evaluated once, in step 6._
