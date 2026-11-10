# Step 4: move-profile cosine kNN (SYNTHETIC data)

> All data is synthetic. These numbers validate the pipeline, not real-world performance.

Each booking's move profile (home size, labor-only, log distance, log lead time, origin/destination access, move month as sin/cos) is standardised, balanced so every feature group carries equal variance, and compared by cosine similarity. A service's score is the similarity-weighted share of the k nearest training bookings that bought it. The cart is ignored, so this model is aimed at empty carts.

## Tuning (validation; selected on empty carts, n=766)

| weighting | k | map@5 empty carts | map@5 all carts |
|---|---|---|---|
| equal | 25 | 0.393 | 0.376 |
| equal | 50 | 0.402 | 0.384 |
| equal | 100 | 0.404 | 0.388 |
| equal | 200 | 0.409 | 0.390 |
| equal | 400 | 0.412 | 0.390 |
| equal | 800 | 0.407 | 0.389 |
| core_x2 | 25 | 0.394 | 0.374 |
| core_x2 | 50 | 0.407 | 0.390 |
| core_x2 | 100 | 0.418 | 0.394 |
| core_x2 | 200 | 0.422 | 0.396 |
| core_x2 | 400 | 0.424 | 0.399 |
| core_x2 | 800 | 0.423 | 0.399 |

Chosen: k=400, weighting=core_x2.

## Val: all services

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.648 | 0.280 | 0.781 | 0.410 | 0.317 |
| profile_popularity | 0.763 | 0.353 | 0.848 | 0.479 | 0.398 |
| co_occurrence | 0.677 | 0.294 | 0.793 | 0.417 | 0.325 |
| implicit_als | 0.655 | 0.280 | 0.767 | 0.395 | 0.303 |
| content_knn | 0.765 | 0.353 | 0.845 | 0.477 | 0.399 |

## Val: excluding $0 items furniture_dolly, hand_truck_dolly

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.656 | 0.313 | 0.768 | 0.445 | 0.340 |
| profile_popularity | 0.755 | 0.381 | 0.831 | 0.497 | 0.416 |
| co_occurrence | 0.663 | 0.317 | 0.783 | 0.454 | 0.344 |
| implicit_als | 0.637 | 0.298 | 0.751 | 0.415 | 0.310 |
| content_knn | 0.757 | 0.378 | 0.827 | 0.494 | 0.411 |

### Val by visible cart size (map@5, all services)

| cart size | carts | global_popularity | profile_popularity | co_occurrence | implicit_als | content_knn |
|---|---|---|---|---|---|---|
| 0 (empty) | 766 | 0.327 | 0.419 | 0.327 | 0.327 | 0.424 |
| 1-2 | 1462 | 0.317 | 0.408 | 0.327 | 0.276 | 0.407 |
| 3-5 | 1528 | 0.312 | 0.387 | 0.322 | 0.303 | 0.386 |
| 6+ | 823 | 0.316 | 0.384 | 0.324 | 0.325 | 0.385 |

## Is content kNN better than profile popularity? (paired bootstrap, map@5)

| carts | kNN minus profile pop | 95% interval |
|---|---|---|
| all validation carts | +0.0006 | [-0.0022, +0.0035] |
| empty validation carts | +0.0057 | [-0.0007, +0.0119] |

An interval that contains 0 means no detectable difference.

## Niche services: mean rank when hidden (all validation carts, 1 = top)

Low-attach services rarely reach the top 5 under any model, so MAP can't show whether profile signals (stairs, lead time, season) were learned. Rank can.

| service | profile_popularity | content_knn |
|---|---|---|
| stairs_protector | 24.036 | 19.969 |
| day_of_packing | 25.794 | 25.833 |
| u_box_5x7 | 18.407 | 17.004 |
| storage_10x10 | 25.744 | 26.565 |
| full_packing | 11.437 | 11.046 |
| mattress_bag | 9.529 | 9.784 |
| piano | 38.196 | 38.391 |

## Empty carts: per-service recall@5, content kNN vs profile popularity

Share of empty validation carts in which the service was hidden AND ranked in the top 5. Services hidden in fewer than 20 empty carts are left out.

Largest gains:

| service | profile_popularity | content_knn | difference |
|---|---|---|---|
| tape_clear | 0.240 | 0.587 | 0.346 |
| hand_truck_dolly | 0.341 | 0.414 | 0.073 |
| u_box_5x7 | 0.123 | 0.175 | 0.053 |
| truck_16ft | 0.464 | 0.514 | 0.050 |
| tape_brown | 0.000 | 0.036 | 0.036 |
| full_packing | 0.000 | 0.028 | 0.028 |
| storage_10x10 | 0.000 | 0.000 | 0.000 |
| storage_5x10 | 0.000 | 0.000 | 0.000 |

Largest losses:

| service | profile_popularity | content_knn | difference |
|---|---|---|---|
| furniture_dolly | 0.895 | 0.721 | -0.174 |
| small_box | 0.910 | 0.802 | -0.108 |
| bubble_wrap | 0.041 | 0.000 | -0.041 |
| dish_wrap | 0.171 | 0.135 | -0.035 |
| mattress_bag | 0.020 | 0.000 | -0.020 |

_Test set sealed. It is evaluated once, in step 6._
