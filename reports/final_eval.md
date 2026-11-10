# Step 6: final evaluation on the sealed test period (SYNTHETIC data)

> All data is synthetic. These numbers validate the pipeline, not real-world performance. Offline evaluation only; no A/B test.

## Protocol

- Test = bookings made 2025-07-01 to 2025-12-28 (summer + fall), after every training and validation booking. Most moves fall in seasons the validation period (spring) never saw.
- 6,428 partial carts: a random subset of each booking's services is visible (possibly none), the rest are hidden targets. Same seeded procedure as validation.
- Services already in the cart, and services the catalog rules forbid (a second truck, a truck on a labor-only move, a second storage unit), are never recommended.
- Models refit on train + val (22,295 bookings) with hyperparameters and hybrid weights chosen on validation. **Nothing was tuned on test.**
- Long-tail view drops the 13 services bought in at least 20% of bookings: truck_10ft, truck_16ft, truck_26ft, small_box, medium_box, large_box, bubble_wrap, dish_wrap, tape_clear, tape_brown, furniture_dolly, hand_truck_dolly, mattress_bag.

## Test: all services

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.666 | 0.292 | 0.791 | 0.423 | 0.330 |
| profile_popularity | 0.777 | 0.364 | 0.852 | 0.487 | 0.409 |
| co_occurrence | 0.692 | 0.303 | 0.801 | 0.430 | 0.338 |
| implicit_als | 0.664 | 0.292 | 0.780 | 0.406 | 0.316 |
| content_knn | 0.778 | 0.363 | 0.856 | 0.490 | 0.408 |
| cooc_lift | 0.440 | 0.147 | 0.560 | 0.223 | 0.139 |
| hybrid | 0.782 | 0.368 | 0.858 | 0.494 | 0.413 |

## Test: excluding $0 items (furniture_dolly, hand_truck_dolly)

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.677 | 0.330 | 0.785 | 0.457 | 0.355 |
| profile_popularity | 0.771 | 0.396 | 0.838 | 0.510 | 0.430 |
| co_occurrence | 0.684 | 0.333 | 0.797 | 0.467 | 0.359 |
| implicit_als | 0.652 | 0.314 | 0.770 | 0.434 | 0.330 |
| content_knn | 0.772 | 0.393 | 0.837 | 0.509 | 0.426 |
| cooc_lift | 0.442 | 0.162 | 0.565 | 0.247 | 0.150 |
| hybrid | 0.777 | 0.398 | 0.839 | 0.514 | 0.431 |

## Test: long tail: excluding services in >= 20% of bookings

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.377 | 0.205 | 0.480 | 0.305 | 0.165 |
| profile_popularity | 0.395 | 0.225 | 0.517 | 0.332 | 0.189 |
| co_occurrence | 0.390 | 0.217 | 0.506 | 0.324 | 0.190 |
| implicit_als | 0.370 | 0.214 | 0.484 | 0.309 | 0.183 |
| content_knn | 0.388 | 0.223 | 0.533 | 0.344 | 0.196 |
| cooc_lift | 0.261 | 0.146 | 0.363 | 0.219 | 0.126 |
| hybrid | 0.410 | 0.235 | 0.532 | 0.344 | 0.203 |

## Test by visible cart size (map@5, all services)

| cart size | carts | global_popularity | profile_popularity | co_occurrence | implicit_als | content_knn | cooc_lift | hybrid |
|---|---|---|---|---|---|---|---|---|
| 0 (empty) | 1116 | 0.349 | 0.440 | 0.349 | 0.349 | 0.440 | 0.175 | 0.442 |
| 1-2 | 2018 | 0.321 | 0.406 | 0.331 | 0.272 | 0.407 | 0.141 | 0.411 |
| 3-5 | 2213 | 0.324 | 0.401 | 0.332 | 0.325 | 0.398 | 0.131 | 0.405 |
| 6+ | 1081 | 0.342 | 0.397 | 0.353 | 0.348 | 0.396 | 0.115 | 0.406 |

## Test by move season (map@5, all services)

| move season | carts | global_popularity | profile_popularity | co_occurrence | implicit_als | content_knn | cooc_lift | hybrid |
|---|---|---|---|---|---|---|---|---|
| summer | 2229 | 0.364 | 0.440 | 0.371 | 0.341 | 0.442 | 0.150 | 0.444 |
| fall | 3261 | 0.313 | 0.392 | 0.322 | 0.303 | 0.389 | 0.133 | 0.396 |
| winter | 938 | 0.313 | 0.395 | 0.318 | 0.305 | 0.392 | 0.136 | 0.401 |

## Validation vs test (map@5, all services)

| model | validation | test | change |
|---|---|---|---|
| global_popularity | 0.317 | 0.330 | +0.013 |
| profile_popularity | 0.398 | 0.409 | +0.011 |
| co_occurrence | 0.325 | 0.338 | +0.013 |
| implicit_als | 0.303 | 0.316 | +0.014 |
| content_knn | 0.399 | 0.408 | +0.009 |
| cooc_lift | 0.139 | 0.139 | +0.000 |
| hybrid | 0.402 | 0.413 | +0.011 |

## Paired bootstrap on test (map@5, 2,000 resamples of carts)

| view | comparison | difference | 95% interval | detectable |
|---|---|---|---|---|
| all services | hybrid - profile_popularity | +0.0045 | [+0.0026, +0.0065] | yes |
| all services | hybrid - global_popularity | +0.0831 | [+0.0778, +0.0886] | yes |
| all services | profile_popularity - global_popularity | +0.0786 | [+0.0735, +0.0841] | yes |
| all services | content_knn - profile_popularity | -0.0009 | [-0.0033, +0.0015] | no |
| all services | implicit_als - global_popularity | -0.0140 | [-0.0199, -0.0081] | yes |
| all services | implicit_als - co_occurrence | -0.0218 | [-0.0272, -0.0163] | yes |
| long tail | hybrid - profile_popularity | +0.0140 | [+0.0100, +0.0179] | yes |
| long tail | hybrid - global_popularity | +0.0378 | [+0.0311, +0.0448] | yes |
| long tail | profile_popularity - global_popularity | +0.0238 | [+0.0177, +0.0303] | yes |
| long tail | content_knn - profile_popularity | +0.0071 | [+0.0021, +0.0118] | yes |
| long tail | implicit_als - global_popularity | +0.0186 | [+0.0120, +0.0257] | yes |
| long tail | implicit_als - co_occurrence | -0.0071 | [-0.0136, +0.0000] | no |
| cart size 0 (empty) | hybrid - profile_popularity | +0.0019 | [-0.0018, +0.0054] | no |
| cart size 1-2 | hybrid - profile_popularity | +0.0048 | [+0.0012, +0.0083] | yes |
| cart size 3-5 | hybrid - profile_popularity | +0.0034 | [-0.0000, +0.0069] | no |
| cart size 6+ | hybrid - profile_popularity | +0.0091 | [+0.0039, +0.0146] | yes |

## Hybrid ablations on test (weights chosen on validation)

| variant | w_profile | w_content | w_cooc_lift | w_als | map@5 all | map@5 long tail |
|---|---|---|---|---|---|---|
| full hybrid | 2.000 | 2.000 | 0.250 | 0.000 | 0.413 | 0.203 |
| without profile | 0.000 | 2.000 | 0.250 | 0.000 | 0.409 | 0.208 |
| without content | 2.000 | 0.000 | 0.250 | 0.000 | 0.409 | 0.202 |
| without cooc_lift | 0.250 | 0.250 | 0.000 | 0.000 | 0.411 | 0.193 |
| without als | 2.000 | 2.000 | 0.250 | 0.000 | 0.413 | 0.203 |
| profile + content only | 0.250 | 0.250 | 0.000 | 0.000 | 0.411 | 0.193 |
| profile + cooc_lift only | 2.000 | 0.000 | 0.250 | 0.000 | 0.409 | 0.202 |

_Test run logged to reports/test_runs.log (previous runs: 0)._
