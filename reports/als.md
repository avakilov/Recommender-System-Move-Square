# Step 3: implicit ALS + fold-in (SYNTHETIC data)

> All data is synthetic. These numbers validate the pipeline, not real-world performance.

Rows of the training matrix are bookings (almost every customer is new), columns are the 45 services. At evaluation time each partial cart is folded in with the item factors fixed (a ridge least-squares solve), then services are ranked by x . y.

## Grid search (validation map@5, 27 configs, 15 iterations)

Top 8:

| factors | reg | alpha | map@5 |
|---|---|---|---|
| 4 | 10.000 | 1 | 0.303 |
| 4 | 1.000 | 1 | 0.297 |
| 4 | 10.000 | 5 | 0.297 |
| 4 | 0.100 | 1 | 0.295 |
| 4 | 1.000 | 5 | 0.294 |
| 4 | 0.100 | 5 | 0.294 |
| 4 | 10.000 | 20 | 0.285 |
| 8 | 10.000 | 1 | 0.281 |

Worst config: 0.142. Chosen: factors=4, reg=10.0, alpha=1.0.

Re-fit with init seeds [0, 1, 2]: map@5 = 0.3025, 0.2990, 0.3019 (spread 0.0035).

Caveat: ALS had a 27-point grid on validation, the baselines had at most 4, so ALS's validation numbers are slightly optimistic. The sealed test set settles it.

## Val: all services

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.648 | 0.280 | 0.781 | 0.410 | 0.317 |
| profile_popularity | 0.763 | 0.353 | 0.848 | 0.479 | 0.398 |
| co_occurrence | 0.677 | 0.294 | 0.793 | 0.417 | 0.325 |
| implicit_als | 0.655 | 0.280 | 0.767 | 0.395 | 0.303 |

## Val: excluding $0 items furniture_dolly, hand_truck_dolly

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.656 | 0.313 | 0.768 | 0.445 | 0.340 |
| profile_popularity | 0.755 | 0.381 | 0.831 | 0.497 | 0.416 |
| co_occurrence | 0.663 | 0.317 | 0.783 | 0.454 | 0.344 |
| implicit_als | 0.637 | 0.298 | 0.751 | 0.415 | 0.310 |

### Val by visible cart size (map@5, all services)

| cart size | carts | global_popularity | profile_popularity | co_occurrence | implicit_als |
|---|---|---|---|---|---|
| 0 (empty) | 766 | 0.327 | 0.419 | 0.327 | 0.327 |
| 1-2 | 1462 | 0.317 | 0.408 | 0.327 | 0.276 |
| 3-5 | 1528 | 0.312 | 0.387 | 0.322 | 0.303 |
| 6+ | 823 | 0.316 | 0.384 | 0.324 | 0.325 |

## What the item factors learned

| service | nearest by item-factor cosine |
|---|---|
| full_packing | unpacking (1.00), wardrobe_box_25in (0.96), hand_truck_dolly (0.86), furniture_disassembly_reassembly (0.84) |
| small_box | tape_brown (1.00), medium_box (1.00), large_box (1.00), speed_pack_regular (0.99) |
| storage_10x10 | wardrobe_box_25in_short (0.93), strap_elastic (0.90), u_box_5x7 (0.89), store_pickup_delivery (0.89) |
| piano | plastic_wrap_13in (1.00), u_box_5x7 (0.99), plastic_wrap_17in (0.99), artwork (0.99) |
| u_box_5x7 | plastic_wrap_13in (1.00), store_pickup_delivery (0.99), piano (0.99), strap_regular (0.99) |
| mattress_bag | bubble_wrap (0.99), strap_regular (0.97), plastic_wrap_15in (0.96), floor_protector_paper (0.96) |

## Diagnostics

| check | value |
|---|---|
| in-sample precision@5 (full training baskets folded in) | 0.646 |
| mean training attach rate of top-5 recs, 1-2 item carts: ALS | 0.324 |
| same, global popularity | 0.460 |
| mean training attach rate of the hidden targets, 1-2 item carts | 0.288 |

_Test set sealed. It is evaluated once, in step 6._
