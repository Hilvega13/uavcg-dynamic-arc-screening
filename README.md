# Machine-learning arc screening for column generation in drone-delivery routing

**What this is.** A code archive from an undergraduate research-training project
(2025-2026). Nothing here is published or under submission, and no new
algorithm is claimed: the method extends Morabit et al. (2023), and the
contribution is a controlled ablation. The repository exists so that a reader
can rerun the pipeline and decide which of the report's claims hold - including
the ones that do not. The report itself is not included.

## The problem

Multi-UAV parcel delivery. Every customer is served exactly once; a route is a
feasible tour from the depot subject to payload, battery/energy and time-window
constraints; the objective is travel cost plus a fixed cost per dispatch.

The master problem is a set-partitioning LP solved with HiGHS through SciPy.
Pricing is an ESPPRC solved by labelling with an ng-route relaxation (k = 10)
and an exact fallback. Only the LP relaxation is certified - there is no
branching, so no integer optimum is claimed.

## The method

A random forest scores each candidate arc with the probability that it lies on a
useful column; arcs below a threshold are deleted from the pricing network.
Deleting arcs never changes column validity or the master constraints, it only
shrinks the network, so optimality survives as long as the final pricing is done
on the full network with the true duals.

The claim under test is narrower than "learning helps": re-screening the arcs
while the duals move should beat screening once.

Mask rules (report settings):

- keep arcs with `p >= tau`, `tau = 0.5`;
- keep the top 3 inbound arcs of every customer (connectivity floor);
- keep every arc used by the last 40 generated columns (no holes next to the
  columns that were just produced);
- depot arcs are always kept.

Learned variants:

- **ML-S (static).** 22 features (geometry, in/out degree, time-window slack,
  ranking), computed once; the mask never changes during a run.
- **DynML-S (dynamic).** The same 22 plus 12 dual-derived features: reduced cost
  `c_ij - pi_j` (tail attribution) and its group rank/gap, `pi_j / pi_max`,
  round-trip and marginal duals, `sigma`, iteration progress, and the previous
  round's negative-column rate (34 features in total). The mask is rebuilt every
  3 pricing calls from the live duals.

Forest (report runs): 200 trees, `max_depth=5`, `min_samples_leaf=50`,
`min_samples_split=100`, `max_features=5`, `class_weight="balanced"`.
The label is `y_opt` - "the arc lies on an optimal pricing path in that round",
collected from full column-generation runs. Features are min-max normalised per
instance. Training and serving share the same feature code.

Dual anchoring: for the first `U = 20` pricing calls the duals are smoothed as
`alpha * pi_bar + (1 - alpha) * pi` with `alpha = 0.25` (R102 uses `U = 8`) to
damp early oscillation. After the window the run returns to the true duals.

Certification (LP only) requires all of: valid duals (C0), pricing under the
true duals (C1), the last pricing on the full network (C2), no truncated pricing
(C3), and `LB_ng >= 0` (C4). A run that stops inside the anchoring window, or
whose ng bound is negative, is reported as `INCOMPLETE` regardless of its LP
value.

## The arms

| Arm | Entry point | What changes |
|---|---|---|
| Pure CG | `scripts/run_redcost_baseline.py --arm baseline` | no mask |
| RedCost-S | `scripts/run_redcost_baseline.py --arm redcost` | deterministic reduced-cost arc filter, refreshed each round |
| ML-S | `scripts/run_mls.py --arm mls` | 22 static features, mask frozen |
| DynML-S | `scripts/run_mls.py --arm dynmls` | 34 features, mask rebuilt from live duals |

All arms share the master LP, the initial columns (one singleton route per
customer) and the certification machinery. Only the pricing network differs, so
comparable outputs are iteration count, label count and per-call pricing cost,
not "whose solution is better".

## Results (report)

Labels are pricing labels summed over the run; iterations are in parentheses;
the last column is DynML-S against ML-S. All five runs are CERTIFIED.

| Instance | Pure CG | ML-S (22 dims) | DynML-S (34 dims) | Change |
|---|---|---|---|---|
| R102-n25 (U=8) | 66,023 (7) | 40,011 (12) | 21,580 (10) | -46.1% |
| R105-n25 | 11,990 (6) | 8,426 (8) | 4,894 (7) | -41.9% |
| R107-n25 | 531,224 (10) | 267,659 (14) | 165,712 (9) | -38.1% |
| R108-n25 | 1,105,770 (9) | 447,853 (14) | 343,886 (9) | -23.2% |
| R201-n50 | 8,491,807 (37) | 1,352,730 (26) | 881,205 (26) | -34.9% |

Against pure column generation the mask alone cuts labels by 29.7% to 84.1%.
The two-factor interaction is the paper's actual point. On R105-n25, with
`alpha = 0`:

| Arm | Labels (iterations) |
|---|---|
| 22 features, frozen | 8,426 (8) |
| 22 features, refreshed | 8,039 (8) |
| 34 features, frozen | 10,107 (10) |
| 34 features, refreshed | 5,306 (7) |

More features alone hurt (+20.0%), refresh alone barely helps (-4.6%), the two
together give -37.0%. The same table on R201-n50 gives +9.9% (and an
`INCOMPLETE` run whose last full-network pricing still found a negative
column), +49.7%, and -9.7%.

The archived `results/v11_configB/` is one such four-arm set: R201-n50,
`alpha = 0.5`, all four arms CERTIFIED at LP `851.5050989447`:

| Arm | Labels (iterations) |
|---|---|
| SF - 22 features, frozen | 2,656,426 (34) |
| STA - 22 features, refreshed | 1,327,859 (40) |
| FRZ - 34 features, frozen | 1,898,442 (26) |
| DYN - 34 features, refreshed | 1,161,284 (26) |

## What does not hold

These are counter-examples measured in the project. They are in the README
because they are worth more than the method description:

| Claim | Measurement |
|---|---|
| A better classifier is enough | The 34-feature frozen arm costs +20.0% labels on R105; only features **and** refresh together help |
| Learning shortens wall clock | The master LP is 89.6% of wall clock, learning and prediction 0.30%. R201: 54.69 -> 50.87 s (-7.0%); R105: 0.21 -> 0.30 s (+42.9%) |
| Fewer labels means less work | On R201 total labels fell 9.7% while dominance checks rose 85.2% and negative columns rose 200.7% |
| The effect transfers across families | C101 +16.6%, RC101 +12.9%, RC104 -43.2%, RC201 -1.7%; R205 is INCOMPLETE at `LB_ng = -3.43` although its last full-network pricing found no column |
| The forest parameters transfer | 200 vs 500 trees flips the sign on R105 (+17.3%) and leaves R107 `INCOMPLETE`, while R201 improves (-41.8%) |
| The original label works | `promising` is 93.6% positive at n = 25 and needs 9,004,683 labels on R201 (6.7x the `y_opt` arm); it does not discriminate at that size |
| It scales to n = 100 | The probe runs cost 1.7-10.7 h per instance; no claim is made beyond n = 50 |

**Reporting discipline.** Split train and test by instance, never by row.
Report per instance and never pool instances into a mean. Compare iteration and
column counts only when both sides converge. A truncated LP is an upper bound
and takes no part in certification.

## Layout

```
uavcg/                 core library (21 modules, ~6,300 lines)
  instance.py          instance parsing, time windows, energy and payload
  labeling.py          ESPPRC labels (ng-route neighbourhood, dominance)
  rmp.py               restricted master problem, HiGHS solve, dual self-check
  cg.py                column-generation loop (stopping rules, certification)
  features.py          static arc features
  dynfeatures.py       dual-derived features
  learning.py          dataset construction, training, inference
  mls.py               mask construction for ML-S and DynML-S
  pricers/             exact / ngroute / greedy / redcost / rollout
scripts/               entry points (21 formal scripts) + _probe_n60_cost.py,
                       one helper imported by the self-test
data/                  benchmark instances and split files
results/v11_configB/   archived four-arm run records (8 runs, 5 diagnostics, 5 traces)
results/_probe_matrix.jsonl  raw budget-probe readings behind the frozen split
                       (the self-test re-derives the split from them)
```

## Code versus report

`CGConfig.mask_after` plus `mask_hook` install **one** mask per run, at the end
of iteration `mask_after`; the hook is called once and never again (see
`uavcg/cg.py`). Consequently:

- `run_mls.py --arm mls` is the closest shipped approximation of the static
  arm: the mask is installed after the first pricing call (the hook needs a
  warm-up trace), not before the first LP solve;
- `run_mls.py --arm dynmls` rebuilds the 34-feature mask **once**, after
  `--w` warm-up calls. It is not the report's every-3-calls refresh;
- the top-3 and protect-40 rules, the anchoring (`alpha`, `U`) and the 3-call
  period were applied by the run driver, which is **not included**;
- each archived record stores the configuration it actually used
  (`features`, `refresh`, `anchor_alpha`, `anchor_until`, `in_top_k`,
  `protect_last`, `mask_lag`) and the same LP value, so the records can be
  checked even where the driver cannot be rerun.

The scripts below reproduce the mechanism and the correctness anchors, not the
five-instance table. Rerunning that table needs the driver (or periodic refresh
added to `CGConfig`); the building blocks are in `uavcg`.

## Environment

Python 3.11. Tested with numpy 2.2.2, SciPy 1.15.1 and scikit-learn 1.6.1:

```
pip install -r requirements.txt
```

No commercial solver is needed: the LP is solved by HiGHS through SciPy.

## Reproducing the anchors

```
python scripts/selftest.py                  # about 8 s, prints "all checks passed"
python scripts/regress_bitlevel.py --fast   # about 15 s, dual anchors
python scripts/regress_bitlevel.py          # about 80 s, adds the BKS red lines
```

The self-test cross-checks master duals, reduced-cost arithmetic, the labelling
pricer and the dominance rules against brute-force enumeration.

The regression anchors are exact, compared with `==`, not a tolerance:

- R105, n = 25: LP `531.5386420627178`, 15 iterations, 277 columns
- R107, n = 25: LP `425.26852495439266`, 43 iterations, 886 columns
- R101: LP must not exceed `1637.7`; RC101: LP must not exceed `1619.8`

A reordering that really changes a result shows up in the seventh digit, so any
tolerance wide enough to hide that hides everything.

```
python scripts/run_cg.py --name R105 --n 25 --pricer exact
python scripts/run_experiment.py --instances R105,RC104,R107 --n 25 \
    --pricers ngroute,greedy,rollout-dual --ng-k 10 --json-out results/local/exp.json
```

The first command is a different setup from the anchor: it reports 12 iterations
and 218 columns, because the anchor runs ng-route with an exact fallback while
`--pricer exact` runs the exact pricer alone. Neither is wrong. In
`run_experiment.py`, `greedy` and `rollout-*` are heuristics and need an exact
fallback; `--exact-max-n` and `--fallback-max-n` are independent.

## Running the learned arms

No model and no training matrix is versioned: the calibrated forest is about
45 MB and the label matrix about 33 MB. Both are rebuilt from full column
generation:

```
python scripts/collect_arc_labels.py --names R105,RC105,C105 --n 25 --tag n25
python scripts/train_rf_arc.py --train results/arc_labels_n25.narrowed.npz \
    --label narrow --tag n25 --features static
python scripts/run_mls.py --arm mls --names R201 --tau 0.5 \
    --model models/rf_arc_n25_static_narrow.pkl
```

Use `--features all` when training for `--arm dynmls`: the dynamic hook rejects
a static-only bundle. `run_mls.py` defaults `--model` to
`models/rf_arc_n25_static_narrow.pkl`, so the path may be omitted.

Three caveats about reproduction:

- `train_rf_arc.py` as shipped grid-searches 500-tree forests
  (`max_depth` 5/10, `max_features` 5/7/sqrt). The report's runs use a 200-tree
  forest with `max_depth=5`, `max_features=5`; adjust `BASE_RF` and `GRID`
  (or supply your own calibrated bundle) to match.
- `run_mls.py` runs with stabilisation off and without anchoring; the report's
  table used anchoring, as noted in "Code versus report".
- `run_mls.py` has no `--n`: it reads the default n = 60 split
  (`data/splits/n60_cat2_v1.json`). A bundle trained at n = 25 demonstrates the
  mechanics but is not a like-for-like run; train on the split you run.

## Data

- `data/solomon100/` - the Solomon 100-customer VRPTW benchmark, 56 instances
- `data/homberger200/` - the Homberger 200-customer extension, 16 instances
- `data/solomon100/minimum.csv` - best known solutions (BKS)

Instances were fetched from two public repositories:
[ML4VRP/ML4VRP2023](https://github.com/ML4VRP/ML4VRP2023)
(`Instances/text/Customer100`, `Customer200`, via
`scripts/fetch_instances.py`) and [felipelagos/mahh](https://github.com/felipelagos/mahh)
(the remaining category-2 instances and `instances/csv/minimum.csv`, via
`scripts/fetch_solomon_cat2.py`). The files under `data/` keep their original
terms.

When citing C101, note that our BKS table gives **827.3**, not the 828.94 used in
part of the literature; state which table you quote.

`data/splits/*.json` split **by instance**, not by arc row; file names encode
the size and instance class.

## References

- LUBBECKE M E, DESROSIERS J. Selected topics in column generation.
  *Operations Research*, 2005, 53(6): 1007-1023. DOI: 10.1287/opre.1050.0234
- MORABIT M, DESAULNIERS G, LODI A. Machine-learning-based arc selection for
  constrained shortest path problems in column generation. *INFORMS Journal on
  Optimization*, 2023, 5(2): 191-210. DOI: 10.1287/ijoo.2022.0082
- MORABIT M, DESAULNIERS G, LODI A. Machine-learning-based column selection for
  column generation. *Transportation Science*, 2021, 55(4): 815-831.
  DOI: 10.1287/trsc.2021.1045
- BALDACCI R, MINGOZZI A, ROBERTI R. New route relaxation and pricing strategies
  for the vehicle routing problem. *Operations Research*, 2011, 59(5):
  1269-1283. DOI: 10.1287/opre.1110.0975
- BEN AMOR H, DESROSIERS J, CARVALHO J M V. Dual-optimal inequalities for
  stabilized column generation. *Operations Research*, 2006, 54(3): 454-463.
  DOI: 10.1287/opre.1060.0278
- PESSOA A, SADYKOV R, UCHOA E, et al. Automation and combination of
  linear-programming based stabilization techniques in column generation.
  *INFORMS Journal on Computing*, 2018, 30(2): 339-360.
  DOI: 10.1287/ijoc.2017.0784
- SOLOMON M M. Algorithms for the vehicle routing and scheduling problems with
  time window constraints. *Operations Research*, 1987, 35(2): 254-265.
  DOI: 10.1287/opre.35.2.254
- BREIMAN L. Random forests. *Machine Learning*, 2001, 45(1): 5-32.
  DOI: 10.1023/A:1010933404324
- PEDREGOSA F, VAROQUAUX G, GRAMFORT A, et al. Scikit-learn: machine learning in
  Python. *Journal of Machine Learning Research*, 2011, 12: 2825-2830
- HUANGFU Q, HALL J A J. Parallelizing the dual revised simplex method.
  *Mathematical Programming Computation*, 2018, 10(1): 119-142.
  DOI: 10.1007/s12532-017-0130-5

## Limitations

- Five R-family instances at n = 25 and one at n = 50; a single seed; the
  10-seed comparison on a random-family instance gives p around 0.091, i.e. not
  significant.
- The certificate covers the LP relaxation only, and C4 depends on the ng bound.
  R205 shows a rejected certificate whose last full-network pricing found no
  column; that is "the relaxation cannot prove it", not "a column is missing".
- HiGHS version changes can move the last digits of LP values.
- The run driver behind the report's tables is not in this repository.

## License

MIT, see [LICENSE](LICENSE). The benchmark instances under `data/` are
third-party and keep their original terms.
