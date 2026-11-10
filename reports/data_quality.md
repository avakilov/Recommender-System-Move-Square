# Data-quality report: SYNTHETIC moving bookings

> **Every row here is synthetic**, generated from `config/assumptions.yaml`. These numbers describe the simulator, not any real business.

## Volume and cold start

| metric | value |
|---|---|
| bookings | 28,725 |
| customers | 27,523 |
| booking dates | 2023-09-05 to 2025-12-28 |
| move dates | 2024-01-01 to 2025-12-31 |
| customers with exactly one booking | 95.8% |
| bookings with no prior customer history (cold start) | 95.8% |

Booking dates can fall before the window start because a move early in the window may have been booked weeks earlier.

## Services per booking

Mean 7.20, median 7, 90th percentile 11, max 22. Empty baskets: 0.1%.

| services | bookings | share |
|---|---|---|
| 0 | 28 | 0.1% |
| 1 | 161 | 0.6% |
| 2 | 562 | 2.0% |
| 3 | 1342 | 4.7% |
| 4 | 2407 | 8.4% |
| 5 | 3435 | 12.0% |
| 6 | 4253 | 14.8% |
| 7 | 4182 | 14.6% |
| 8 | 3839 | 13.4% |
| 9 | 3032 | 10.6% |
| 10 | 2189 | 7.6% |
| 11 | 1428 | 5.0% |
| 12+ | 1867 | 6.5% |

## Attach rate per service

Attach rate = share of all bookings that include the service. `free` = $0 line item (adds no revenue; recommend only as a reminder). `rare` = under 1% of bookings.

| service | group | attach | flags |
|---|---|---|---|
| medium_box | boxes_wrap | 55.6% |  |
| small_box | boxes_wrap | 50.4% |  |
| large_box | boxes_wrap | 47.1% |  |
| furniture_dolly | equipment | 41.6% | free |
| tape_clear | tape | 41.4% |  |
| hand_truck_dolly | equipment | 36.2% | free |
| truck_16ft | truck | 33.7% |  |
| dish_wrap | boxes_wrap | 29.7% |  |
| bubble_wrap | boxes_wrap | 25.9% |  |
| tape_brown | tape | 25.8% |  |
| truck_10ft | truck | 25.1% |  |
| truck_26ft | truck | 23.3% |  |
| mattress_bag | other | 22.2% |  |
| full_packing | packing_service | 17.5% |  |
| floor_protector_paper | surface_protection | 16.9% |  |
| furniture_disassembly_reassembly | furniture | 16.8% |  |
| unpacking | packing_service | 14.4% |  |
| wardrobe_box_25in | wardrobe_box | 14.1% |  |
| tv_protector | surface_protection | 13.9% |  |
| plastic_wrap_15in | plastic_wrap | 13.2% |  |
| strap_regular | equipment | 11.0% |  |
| floor_protector_carpet | surface_protection | 10.9% |  |
| speed_pack_regular | speed_pack | 9.0% |  |
| plastic_wrap_17in | plastic_wrap | 8.3% |  |
| u_box_5x7 | container | 8.1% |  |
| stairs_protector | surface_protection | 8.0% |  |
| strap_heavy_duty | equipment | 7.8% |  |
| day_of_packing | packing_service | 7.8% |  |
| glass_protector | surface_protection | 7.7% |  |
| wardrobe_box_20in | wardrobe_box | 7.5% |  |
| strap_elastic | equipment | 7.4% |  |
| store_pickup_delivery | other | 7.4% |  |
| plastic_wrap_13in | plastic_wrap | 7.2% |  |
| storage_10x10 | storage | 6.1% |  |
| storage_5x10 | storage | 5.4% |  |
| large_appliance | specialty | 5.2% |  |
| speed_pack_large | speed_pack | 5.1% |  |
| wardrobe_box_25in_short | wardrobe_box | 5.1% |  |
| storage_10x15 | storage | 5.0% |  |
| heavy_item_other | specialty | 4.3% |  |
| storage_10x20 | storage | 3.3% |  |
| artwork | specialty | 3.0% |  |
| piano | specialty | 2.0% |  |
| storage_5x5 | storage | 1.6% |  |
| gun_safe | specialty | 1.5% |  |

**Revenue flag:** furniture_dolly, hand_truck_dolly are $0 items. Hit-rate metrics count them like any other service, so they can make a recommender look better without adding revenue. Later steps report metrics with and without them.

## Seasonality (by move month)

```
2024-01    754  #################
2024-02    762  #################
2024-03   1010  #######################
2024-04   1076  #########################
2024-05   1261  #############################
2024-06   1552  ####################################
2024-07   1633  #####################################
2024-08   1594  ####################################
2024-09   1229  ############################
2024-10   1133  ##########################
2024-11   1020  #######################
2024-12    969  ######################
2025-01    836  ###################
2025-02    806  ##################
2025-03   1031  ########################
2025-04   1143  ##########################
2025-05   1419  ################################
2025-06   1747  ########################################
2025-07   1733  ########################################
2025-08   1745  ########################################
2025-09   1250  #############################
2025-10   1098  #########################
2025-11    986  #######################
2025-12    938  #####################
```

| season | bookings | storage attach |
|---|---|---|
| winter | 5065 | 19.4% |
| spring | 6940 | 19.8% |
| summer | 10004 | 25.2% |
| fall | 6716 | 18.8% |

## Rare combinations

| metric | value |
|---|---|
| distinct service pairs seen | 977 |
| pairs seen fewer than 5 times | 3 (0.3%) |
| distinct full baskets | 25,475 |
| bookings whose exact basket is unique | 83.6% |

## Drift over time (observable)

| period | full_packing attach | labor-only share | long-distance share |
|---|---|---|---|
| 2024-H1 | 16.6% | 17.4% | 14.0% |
| 2024-H2 | 16.9% | 18.8% | 13.0% |
| 2025-H1 | 18.3% | 17.9% | 13.8% |
| 2025-H2 | 18.2% | 17.5% | 13.6% |

## Hidden segment mix (generator diagnostics only)

These labels are hidden from every model. They are shown only to confirm the drift built into the generator.

| period | budget_diy | busy_professional | downsizer | family_upsizer | long_distance_relocator | student_renter |
|---|---|---|---|---|---|---|
| 2024-H1 | 26.9% | 18.9% | 9.2% | 19.8% | 8.4% | 16.9% |
| 2024-H2 | 26.6% | 19.7% | 9.8% | 18.0% | 7.5% | 18.4% |
| 2025-H1 | 25.1% | 21.8% | 9.8% | 19.2% | 8.1% | 16.0% |
| 2025-H2 | 24.2% | 23.0% | 9.7% | 18.4% | 7.9% | 16.7% |

## Assumptions that need attention before publishing

- `monthly_move_weights`: public source not yet verified
- `free_services`: from the project spec; needs a public source or removal before publishing
- `truck_size_by_home_size`: public source not yet verified
- `storage_size_by_home_size`: public source not yet verified
- `business_rules_not_simulated`: from the project spec; needs a public source or removal before publishing
