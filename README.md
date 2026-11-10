# Add-on recommender for moving bookings (synthetic data)

> **All data in this repository is SYNTHETIC.** It is generated from documented
> assumptions in [`config/assumptions.yaml`](config/assumptions.yaml) and does not
> come from, or mirror, any real company. Results validate the **pipeline**, not
> real-world performance. Evaluation is offline only. There is no A/B test.

When a customer books a move, recommend the add-ons they are likely to want:
truck size, packing services, boxes and wrap, surface protection, equipment,
specialty handling, storage. There are 45 services. Almost every customer books
only once, so there is no purchase history to lean on: **cold start is the
normal case, not the edge case.**

## Results at a glance

Sealed test period (bookings made Jul–Dec 2025), 6,428 partial carts, MAP@5.
Models were refit on train + val with settings chosen on validation. The test
set was evaluated once ([`reports/test_runs.log`](reports/test_runs.log)).

| model | all services | long tail¹ | uses |
|---|---|---|---|
| global popularity | 0.330 | 0.165 | nothing |
| co-occurrence | 0.338 | 0.190 | cart |
| implicit ALS + fold-in | 0.316 | 0.183 | cart |
| content kNN | 0.408 | 0.196 | move profile |
| profile popularity | 0.409 | 0.189 | move profile |
| **hybrid** | **0.413** | **0.203** | profile + content + cart lift |

¹ Long tail = the 13 services bought in at least 20% of bookings (trucks,
boxes, tape, wraps, dollies, mattress bag) are removed from both targets and
candidates. It asks whether a model finds the less obvious add-ons.

Hybrid minus profile popularity (paired bootstrap, 95% interval):
**+0.0045 [+0.003, +0.007]** on all services, **+0.014 [+0.010, +0.018]** on the
long tail.

**In one sentence:** knowing the move (home size, truck vs labor-only,
distance) is worth about 0.08 MAP@5 over global popularity. Adding the cart on
top of that is worth a small, real extra gain, mostly on less common services.
Matrix factorisation adds nothing here.

## Quick start

```bash
pip install -r requirements.txt
pytest                                  # 63 tests, ~3 s
python scripts/generate_data.py         # synthetic CSVs + data-quality report, ~3 s
python scripts/run_baselines.py         # step 2, validation only, ~2 s
python scripts/run_als.py               # step 3, ~40 s
python scripts/run_content.py           # step 4, ~1 min
python scripts/run_hybrid.py            # step 5, ~1 min
python scripts/run_final_eval.py --dry-run-on-val   # check step 6 without opening test
```

Each step script writes a Markdown report to `reports/` and its validation-chosen
settings to `reports/params/`. The final test evaluation reads those settings:

```bash
python scripts/run_final_eval.py        # opens the test set; every run is logged
```

Generation is deterministic (seed 0), so these commands reproduce every number
in this README.

## Repo layout

```
config/
  assumptions.yaml          every simulation assumption, with its source (frozen)
  experiment.yaml           split dates, eval seed, k values, tuning grids
src/moverec/
  assumptions.py            load + validate the assumptions file
  catalog.py                the 45-service catalog
  generator.py              synthetic bookings and baskets
  report.py                 data-quality report
  data.py                   load public CSVs, build the booking x service matrix
  split.py                  time-based train / val / test split
  evaluate.py               partial carts, catalog rules, metrics, slices, bootstrap
  baselines.py              global popularity, profile popularity, co-occurrence, lift
  als.py                    implicit ALS in NumPy + least-squares fold-in
  content.py                move-profile vectors + cosine kNN
  hybrid.py                 weighted blend of standardised component scores
  experiment.py             shared setup so every script sees identical carts
scripts/                    one script per step (see Quick start)
reports/                    generated reports, chosen params, test-run log
tests/                      unit tests, including a leakage guard
CHANGELOG.md                any change to the frozen assumptions must be logged here
```

## 1. The synthetic data

### How a booking is generated

1. **Customer and date.** Move dates follow a seasonal curve that peaks in summer.
   Each customer gets a **hidden segment**: budget DIY, busy professional,
   family upsizer, student renter, downsizer, or long-distance relocator. 92%
   of customers are set to book once; repeat bookings come at least 4 months later.
2. **Move profile (observed).** The segment shapes home size, distance band,
   truck vs labor-only, and lead time. Access (ground / stairs / elevator) is
   random at each end.
3. **Latent needs (hidden).** Five needs (DIY packing, pro packing, protection,
   heavy items, storage) = segment offset + profile effects + noise.
4. **Purchases.** Each service is bought with probability sigmoid(logit). The
   logit combines the needs, **segment-only boosts that no observed feature
   explains**, direct profile effects (stairs protector needs stairs),
   items already in the basket (boxes make tape likelier), and noise. The truck
   (exactly one for truck moves) and storage unit (at most one) are picked from
   size tables keyed on home size.
5. **Basket noise.** 4% of bookings get one random extra service and 3% lose
   one. This creates rare combinations.

### Assumptions file

Every number lives in [`config/assumptions.yaml`](config/assumptions.yaml). Each
entry has a `value`, a `source_type` (`public`, `generic_reasoning`,
`user_spec`, `illustrative`) and a `source`. Public sources also carry
`verified: true/false`. The loader rejects any entry without a source, and the
data-quality report lists every unverified source. Most numbers are honestly
labelled `illustrative`: they are there to make the simulation non-trivial, not
to describe any real business.

The file was **frozen before any model was run**. Nothing in it was tuned after
seeing model results ([`CHANGELOG.md`](CHANGELOG.md)).

### Guarding against circular evaluation

A synthetic benchmark is easy to win by accident if the model mirrors the
generator. To make that harder:

- **Hidden segments** drive behaviour, but no model can read them. A test
  ([`tests/test_step2.py`](tests/test_step2.py)) fails if any modelling code
  refers to the hidden folder.
- **Segment-only boosts** put signal in the data that only collaborative
  filtering could recover, giving ALS a fair chance.
- **Noise everywhere:** on the latent needs, on every purchase decision, and on
  whole baskets. 84% of bookings have a basket no other booking has.
- **Cold start:** 95.8% of customers book exactly once.
- **Drift:** the segment mix shifts over the two years, and students move more
  in summer, so the test period is not a copy of training.

### What the data looks like ([`reports/data_quality.md`](reports/data_quality.md))

| | |
|---|---|
| bookings / customers | 28,725 / 27,523 |
| customers with one booking | 95.8% |
| services per booking | mean 7.2, median 7 |
| busiest vs quietest months | Jun–Aug have about 2x the moves of Jan–Feb |
| bookings with a unique basket | 84% |
| $0 services | `furniture_dolly`, `hand_truck_dolly` (reminders, no revenue) |

## 2. Evaluation protocol

**Time-based split on booking time** (when the recommendation would be made):

| part | bookings | period | cold-start share |
|---|---|---|---|
| train | 17,710 | up to Mar 2025 | 97.2% |
| validation | 4,585 | Apr–Jun 2025 (spring) | 94.9% |
| test | 6,430 | Jul–Dec 2025 (summer, fall, winter) | 92.9% |

**Partial carts.** For each held-out booking, a random subset of its services is
visible (possibly none) and the rest are hidden targets. At least one is always
hidden. The carts are drawn once from a fixed seed, so every model is scored on
identical carts.

**Ranking rules, the same for every model:**
- Services already in the cart are never recommended.
- Services the catalog rules forbid are never recommended: a second truck, a
  truck on a labor-only move, or a second storage unit.

**Metrics:** hit rate@3/5 (at least one hidden service in the top k), recall@3/5,
and MAP@5. Every metric is reported three ways: all services, without the $0
items, and long tail. Results are also sliced by cart size and by move season.

**Statistics:** model differences use a paired bootstrap over carts (2,000
resamples, 95% interval).

**Discipline:**
- All tuning happened on validation.
- The test set was opened once, by a script that refits on train + val, reads
  the saved settings, tunes nothing, and appends each run to a log.
- A `--dry-run-on-val` mode exercised that script before the test set was opened.

## 3. Models and design choices

| model | idea | why |
|---|---|---|
| **Global popularity** | rank by training attach rate | the floor every model must beat |
| **Profile popularity** | attach rate within move type × home size × distance band; small segments shrink toward their parent level (m = 20 pseudo-bookings) | the honest "segment" baseline. It uses observed segments only, since the hidden ones are off limits |
| **Co-occurrence** | mean of P(j \| i) over cart items | the simplest cart-aware model |
| **Implicit ALS** | Hu–Koren–Volinsky weighted factorisation on bookings × services, in NumPy. A partial cart is **folded in** by solving a ridge least-squares problem with item factors fixed. `fold_in_lstsq` is the readable `np.linalg.lstsq` version, tested equal to the batched solver | almost every booking is a new user, so user factors from training are useless. Fold-in is the only way to score a new cart |
| **Content kNN** | standardised move-profile vector (ordinal home size, labor-only, log distance, log lead time, access one-hots, month as sin/cos), each feature group scaled to equal variance; cosine similarity to the 400 nearest training moves | covers the empty cart, the moment when nothing but the move is known |
| **Co-occurrence lift** | mean log P(j \| i) / P(j) over cart items, with P(j \| i) shrunk toward P(j) | the cart signal **with popularity divided out**, built to be added to a popularity-type score |
| **Hybrid** | Σ weight × row-standardised score. Profile and content scores are logged first; cart components are flat on empty carts and drop out there | weights picked from a 624-point grid on validation: profile 2, content 2, lift 0.25, ALS 0 |

## 4. Results

All numbers below are from the test period unless marked otherwise. The full
tables are in [`reports/final_eval.md`](reports/final_eval.md).

### By visible cart size (MAP@5, all services)

| cart size | carts | global pop. | profile pop. | co-occ. | ALS | content kNN | hybrid |
|---|---|---|---|---|---|---|---|
| 0 (empty) | 1,116 | 0.349 | 0.440 | 0.349 | 0.349 | 0.440 | 0.442 |
| 1–2 | 2,018 | 0.321 | 0.406 | 0.331 | 0.272 | 0.407 | 0.411 |
| 3–5 | 2,213 | 0.324 | 0.401 | 0.332 | 0.325 | 0.398 | 0.405 |
| 6+ | 1,081 | 0.342 | 0.397 | 0.353 | 0.348 | 0.396 | 0.406 |

The hybrid's edge over profile popularity grows with cart size: +0.002 on empty
carts (not significant) up to +0.009 on carts with 6+ items (significant). That
is what you'd expect if the cart is the extra information.

### Hybrid ablations (test; weights for each variant re-chosen on validation)

| variant | MAP@5 all | MAP@5 long tail |
|---|---|---|
| full hybrid | 0.413 | 0.203 |
| without co-occurrence lift | 0.411 | **0.193** |
| without content kNN | 0.409 | 0.202 |
| without profile popularity | 0.409 | 0.208 |
| without ALS | 0.413 | 0.203 |

- **The lift component drives the long-tail gain.**
- **Profile popularity and content kNN can largely stand in for each other**,
  since both encode the move.
- **ALS has zero weight**, so removing it changes nothing.

### Validation vs test (MAP@5)

Every model scored about 0.01 higher on test than on validation, and the order
of the models was the same. This comes from refitting on more recent data and
from easier summer carts (summer 0.44 vs fall 0.40 for the hybrid). There is no
sign that the validation tuning overfit.

## 5. Honest limitations

- **Where ALS adds little: here.**
  - **The numbers.** ALS is below global popularity overall (−0.014). It does
    beat popularity on the long tail (+0.019), but still loses to plain
    co-occurrence there.
  - **It isn't a fitting bug.** ALS reconstructs full training baskets well
    (in-sample precision@5 = 0.65), and its item factors find the planted
    structure (full packing ↔ unpacking, boxes ↔ tape).
  - **Fold-in from 1–2 items is noisy.** On those carts it recommends niche
    services (mean attach rate 0.32 vs 0.46 for popularity).
  - **The full table can be estimated directly.** With 45 services and 17k
    baskets, the whole 45 × 45 co-occurrence table is precise, so a low-rank
    approximation only throws information away.
  - **Tuning pushes it toward popularity.** The best ALS setting ended at the
    smallest grid values (4 factors).

  ALS earns its place with large, sparse catalogs and real user histories,
  neither of which this problem has.
- **The gains are small.** +0.004 MAP@5 is statistically real on 6,400 carts,
  but small in absolute terms. Whether it matters for a business is a question
  for an online test.
- **Top-k metrics reward common items.** Boxes appear in about half of all
  bookings, so any well-calibrated model puts them first. Personalisation shows
  up more clearly in the long-tail view and in rank diagnostics (e.g., content
  kNN moves `stairs_protector` from mean rank 24 to 20 on validation) than in
  headline MAP.
- **The data encodes my assumptions.** A model that wins here has recovered
  structure I planted, under noise I chose. The hidden segments, segment-only
  boosts and drift make this harder to game, but they can't make it real.
- **One synthetic dataset.** Everything comes from one generator seed. A
  robustness check across several seeds is not done.
- **The partial-cart protocol is idealised.** Services are hidden uniformly at
  random. Real carts fill in an order (truck first, supplies later), and the
  order changes what is easy to predict.
- **Not modelled:** prices, margins and quantities (the $0 dollies are
  reported separately but not down-weighted); returning customers' history
  (rare here, but free signal when present); and the specialty and storage
  pricing rules (documented only).
- **Settings changed after seeing validation results** (never test):
  - The ALS grid was shifted after its first best point sat at the grid edge.
  - The co-occurrence lift component was designed after raw co-occurrence turned
    out to be popularity-dominated.
  - ALS got a larger grid (27 points) than the baselines (4).

## 6. What I'd do next

1. **A learned ranker.** Score each (booking, service) pair with a
   gradient-boosted model or logistic regression, trained with a ranking
   objective (e.g., LambdaMART) on simulated partial carts from training.
   - **Features:** the move profile, service identity and group, the scores
     from the models above (profile rate, kNN rate, max and mean lift from the
     cart, ALS score), and cart size.
   - **Why:** each service gets its own learned dependence on lead time,
     access and season. The blend can't do that, and it is where the missing
     long-tail signal sits (e.g., the lead time → day-of-packing effect neither
     profile model picked up).
   - **Bonus:** calibrated probabilities make margin-aware ranking possible
     (expected revenue = P(buy) × margin) and handle the $0 items properly.
2. **A better evaluation.** Hide services in a realistic order. Add a
   revenue-weighted metric. Repeat everything across several generator seeds
   and report the spread.
3. **Two-tower retrieval only if the problem grows.** That means a catalog of
   hundreds or thousands of items, or rich text and image features for services
   and customers. With 45 services, retrieval is not a bottleneck: every model
   here scores the whole catalog for a cart in about a millisecond or less.
4. **Online test before any claim.** An A/B test on attach rate and revenue per
   booking, with guardrails on conversion. This is the only way to measure real
   impact, and this repo deliberately does not pretend to have one.
