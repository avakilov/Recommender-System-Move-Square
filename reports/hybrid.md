# Step 5: weighted hybrid (SYNTHETIC data)

> All data is synthetic. These numbers validate the pipeline, not real-world performance.

hybrid(j) = sum over components of weight x row-standardised score. Profile and content scores are logged first (probabilities multiply, so logs add). The co-occurrence component here is the **lift** version, mean log P(j|i)/P(j) over cart items, so it carries the cart signal without counting popularity twice. Cart components are flat on empty carts and abstain there.

## Weight search (624 combinations, validation map@5)

Top 8:

| profile | content | cooc_lift | als | map@5 |
|---|---|---|---|---|
| 2.000 | 2.000 | 0.250 | 0.000 | 0.402 |
| 2.000 | 1.000 | 0.250 | 0.000 | 0.402 |
| 2.000 | 0.500 | 0.250 | 0.000 | 0.402 |
| 1.000 | 2.000 | 0.250 | 0.000 | 0.402 |
| 0.500 | 2.000 | 0.250 | 0.000 | 0.401 |
| 0.250 | 2.000 | 0.250 | 0.000 | 0.401 |
| 0.250 | 0.250 | 0.000 | 0.000 | 0.401 |
| 0.500 | 0.500 | 0.000 | 0.000 | 0.401 |

Chosen weights: profile=2.0, content=2.0, cooc_lift=0.25, als=0.0 -> validation map@5 0.4025.

## Honest estimate: 2-fold cross-fit inside validation

Weights tuned on one half of the validation carts, scored on the other half.

| tuned on half | scored on half | w_profile | w_content | w_cooc_lift | w_als | hybrid map@5 | profile_pop map@5 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 2.000 | 1.000 | 0.250 | 0.000 | 0.401 | 0.398 |
| 2 | 1 | 1.000 | 2.000 | 0.250 | 0.000 | 0.401 | 0.399 |

## Ablations (validation, remaining weights re-tuned)

| variant | w_profile | w_content | w_cooc_lift | w_als | map@5 |
|---|---|---|---|---|---|
| full hybrid | 2.000 | 2.000 | 0.250 | 0.000 | 0.402 |
| without profile | 0.000 | 2.000 | 0.250 | 0.000 | 0.400 |
| without content | 2.000 | 0.000 | 0.250 | 0.000 | 0.399 |
| without cooc_lift | 0.250 | 0.250 | 0.000 | 0.000 | 0.401 |
| without als | 2.000 | 2.000 | 0.250 | 0.000 | 0.402 |
| profile + cooc_lift only | 2.000 | 0.000 | 0.250 | 0.000 | 0.399 |
| profile + content only | 0.250 | 0.250 | 0.000 | 0.000 | 0.401 |

## Val: all services

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.648 | 0.280 | 0.781 | 0.410 | 0.317 |
| profile_popularity | 0.763 | 0.353 | 0.848 | 0.479 | 0.398 |
| co_occurrence | 0.677 | 0.294 | 0.793 | 0.417 | 0.325 |
| implicit_als | 0.655 | 0.280 | 0.767 | 0.395 | 0.303 |
| content_knn | 0.765 | 0.353 | 0.845 | 0.477 | 0.399 |
| cooc_lift | 0.438 | 0.145 | 0.554 | 0.221 | 0.139 |
| hybrid | 0.769 | 0.357 | 0.850 | 0.482 | 0.402 |

## Val: excluding $0 items furniture_dolly, hand_truck_dolly

| model | hit@3 | recall@3 | hit@5 | recall@5 | map@5 |
|---|---|---|---|---|---|
| global_popularity | 0.656 | 0.313 | 0.768 | 0.445 | 0.340 |
| profile_popularity | 0.755 | 0.381 | 0.831 | 0.497 | 0.416 |
| co_occurrence | 0.663 | 0.317 | 0.783 | 0.454 | 0.344 |
| implicit_als | 0.637 | 0.298 | 0.751 | 0.415 | 0.310 |
| content_knn | 0.757 | 0.378 | 0.827 | 0.494 | 0.411 |
| cooc_lift | 0.442 | 0.161 | 0.557 | 0.243 | 0.149 |
| hybrid | 0.760 | 0.382 | 0.831 | 0.499 | 0.415 |

### Val by visible cart size (map@5, all services)

| cart size | carts | global_popularity | profile_popularity | co_occurrence | implicit_als | content_knn | cooc_lift | hybrid |
|---|---|---|---|---|---|---|---|---|
| 0 (empty) | 766 | 0.327 | 0.419 | 0.327 | 0.327 | 0.424 | 0.180 | 0.422 |
| 1-2 | 1462 | 0.317 | 0.408 | 0.327 | 0.276 | 0.407 | 0.155 | 0.410 |
| 3-5 | 1528 | 0.312 | 0.387 | 0.322 | 0.303 | 0.386 | 0.122 | 0.392 |
| 6+ | 823 | 0.316 | 0.384 | 0.324 | 0.325 | 0.385 | 0.105 | 0.392 |

## Hybrid minus profile popularity (paired bootstrap, map@5)

| part | cart size | carts | difference | 95% interval |
|---|---|---|---|---|
| val | all | 4579 | +0.0042 | [+0.0018, +0.0065] |
| val | 0 (empty) | 766 | +0.0037 | [-0.0005, +0.0079] |
| val | 1-2 | 1462 | +0.0019 | [-0.0026, +0.0061] |
| val | 3-5 | 1528 | +0.0049 | [+0.0008, +0.0092] |
| val | 6+ | 823 | +0.0072 | [+0.0003, +0.0141] |

An interval that excludes 0 is a detectable difference.

_Test set sealed. It is evaluated once, in step 6._
