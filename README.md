# Extending CoSimRank for Dynamic Similarity Search on Evolving Graphs

[![CI](https://github.com/yingtongc4/dynamic-cosimrank/actions/workflows/ci.yml/badge.svg)](https://github.com/yingtongc4/dynamic-cosimrank/actions/workflows/ci.yml)

Code accompanying the CS907 dissertation of the same title.

Chen Yin-tung — Supervisor: Dr Weiren Yu
Department of Computer Science, University of Warwick

---

## What this is

An exact incremental maintenance framework for CoSimRank on a changing graph.
It extends the rank-one edge-insertion update of D-CoSim (Yu & Wang, WWW 2018)
to edge **deletion**, to node insertion and deletion, and adds a provably sound
stopping rule, a cost model with a closed-form break-even, and a refresh policy
derived from measured drift.

The central claim the code supports is a threshold: repairing the maintained
columns beats recomputing them exactly when the number of maintained queries
exceeds `2*d_bar/(d_bar - 1)`, which is about three on any graph denser than a
tree.

## Requirements

    python -m pip install -r requirements.txt

Python 3.12.10 with NumPy 2.2.6, SciPy 1.17.0, NetworkX 3.6.1, matplotlib
3.10.8 and pytest 9.1.1. No compiled extensions, no GPU, no network access at
run time.

## Layout

    src/          the framework
      dynamic_cosimrank.py   unified API over all four graph operations
      edge_deletion.py       signed-column deletion (Theorem 1)
      node_ops.py            node insertion and deletion by reduction to edges
      per_query.py           sparse per-query mode; maintains only S[:, Q]
      d_cosim.py             the published insertion update, for comparison
      simrank.py             SimRank, for the contrasting-measure discussion
    tests/        44 pytest functions (API behaviour, invariants, streams)
    benchmarks/   verification and measurement scripts
      results/    every CSV and PNG the dissertation reports
    data/         dataset loaders and the SNAP edge lists

## Running it

Tests first — they take under two seconds:

    python -m pytest tests/ -q

Then the verification layers, in this order. The benchmarks refuse to report a
timing unless the correctness gate in `bench_perquery.py` passes, so running
them out of order will not silently produce a wrong number.

| Layer | Script | Establishes |
|---|---|---|
| 1 | `verify_correctness.py` | 1,120 operations, four sizes, two graph families |
| 1 | `verify_exact.py` | equality over the rationals, no tolerance |
| 1 | `verify_perquery.py` | per-query mode matches the full matrix |
| 2 | `verify_error_bound.py` | the bounds of Lemma 3 and Theorem 2 hold at every depth |
| 2 | `verify_ranking.py` | top-k overlap, order and NDCG against a converged reference |
| 3 | `bench_perquery.py` | the six parameter sweeps E1–E6 |

Everything under layer 3 is a timing experiment and may be run in any order.
Full reproduction is roughly seventy minutes, dominated by
`bench_case_twitter.py` (~19 min), `bench_dblp_full.py` (~17 min) and
`verify_correctness.py` (~8 min).

## Which script produces which table

Nineteen captions in the dissertation name their script directly. The rest:

| Table / figure | Script |
|---|---|
| Measured dataset statistics | `data/dataset_stats.py` |
| Correctness across sizes and families | `verify_correctness.py` |
| Exact verification over the rationals | `verify_exact.py` |
| Distance to the converged reference; residual and its ratio | `verify_error_bound.py` |
| Adaptive-K against fixed depth | `verify_error_bound.py` |
| Ranking agreement, and the com-DBLP tie | `verify_ranking.py`, `verify_ranking_ties.py` |
| The six sweeps E1–E6, and their figure | `bench_perquery.py` |
| Node-operation phase breakdown; low out-degree deletion | `bench_nodeop_phases.py`, `bench_constants.py` |
| Blocked baseline; all four implementations | `bench_blocked.py`, `bench_blocked_both.py` |
| Memory, resident and transient | `bench_memory.py` |
| Cumulative error over a stream | `bench_stream.py` |
| Updates between refreshes | `bench_drift_policy.py` |
| Drift against density | `bench_drift_density.py` |
| Extraction realisations | `bench_realizations.py` |
| Scope of an update (eight nodes), and its figure | `fig_scope8.py` |
| The three worked-example graphs | `fig_toy_graphs.py` |
| Worked Example A round trip; Worked Example B counts | `worked_example_nodeops.py`, `worked_example_inedge.py` |

The operating-envelope summary and the related-work table are written by hand
from the results above; they are not generated.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- the 44 unit tests on Python 3.11, 3.12 and 3.13
- `verify_perquery.py`, which checks the per-query mode against the full matrix
- `verify_exact.py`, which re-derives the update in exact rational arithmetic
- the same three again against the pinned versions in `requirements.txt`

None of these reads a dataset, so CI needs none of the SNAP files and finishes
in seconds.

What the unit tests do and do not catch, measured by injecting a relative error
into the deletion update: 1e-3 fails 10 of the 44, 1e-1 fails 12, and 1e-6
passes. Their tolerances predate the 1e-8 regime used for the dissertation's
own correctness sweep, so they are a regression gate rather than a precision
one. `verify_exact.py` imports nothing from `src/`: it checks that the
mathematics holds, not that the implementation matches it.


## Data

Three public SNAP datasets, `data/`:

| File | Dataset | Source |
|---|---|---|
| `facebook_combined.txt.gz` | ego-Facebook, 4,039 nodes | McAuley & Leskovec, NIPS 2012 |
| `twitter_combined.txt.gz` | ego-Twitter, 81,306 nodes, directed | McAuley & Leskovec, NIPS 2012 |
| `com-dblp.txt.gz` | com-DBLP, 317,080 nodes | Yang & Leskovec, ICDM 2012 |

Only the edge lists are used. The ego-network collections are distributed with
node features, circles and ego identifiers as well; none of those files is
included or read here. The loaders read two integers per line and build a
graph, so the only data entering any computation are anonymised node
identifiers and the edges between them.

`com-dblp.txt.gz` is read directly; an uncompressed `com-dblp.txt` is used in
preference if present, but is not required.

## Notes for a reader

- Timings vary between sessions by up to a factor of about 1.5 on this
  hardware. Ratios are more stable than absolute times, and the dissertation
  reports medians with the spread stated.
- `benchmarks/results/` is checked in so the tables can be inspected without
  re-running anything.
- `benchmark.py` and `benchmark_dblp.py` are the interim-report harnesses.
  Nothing in the final dissertation is taken from them — they were superseded
  by the gated per-experiment scripts above — and they are retained only so the
  progression from the interim submission is visible.
