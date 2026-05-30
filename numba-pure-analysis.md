# GroupIndex & Disorder Measures: Numba vs Pure -- Performance Analysis

## Context

The `skrough` package uses `numba` (`@numba.njit`) in several hot-path
functions. The question was whether removing the numba dependency (replacing
all `@njit` functions with pure-numpy equivalents) would be viable without a
significant performance penalty.

## Files Using Numba

16 `@numba.njit`-decorated functions across 7 files:

| File | Functions |
|------|-----------|
| `disorder_measures/disorder_measures.py` | `gini_impurity`, `entropy`, `conflicts_count` |
| `structs/group_index/_numba.py` | `_get_distribution` |
| `structs/group_index/_hash_numba.py` | `_hash_combine_u64`, `_hash_rows`, `_hash_split` |
| `structs/group_index/_dict_numba.py` | `_streaming_disorder` |
| `predict/helpers.py` | 1 function |
| `predict/aggregate.py` | 1 function |
| `homogeneity.py` | 3 functions |
| `rough.py` | 1 function |
| `utils.py` | 2 functions |

## Experiment 1: `get_distribution` (GroupIndex)

### Setup

Micro-benchmark of `GroupIndexPure.get_distribution` (`np.add.at`) vs
`GroupIndexNumba.get_distribution` (`@numba.njit` loop), with cold-start
compilation done before timing.

### Results

| Dataset | objs | groups | values | numba | pure | Ratio |
|---------|------|--------|--------|-------|------|-------|
| Small   | 1k   | 50     | 5      | 17us  | 34us | 2.0x  |
| Medium  | 50k  | 500    | 10     | 42us  | 286us| 6.8x  |
| Large   | 200k | 2k     | 20     | 214us | 1.3ms| 5.9x  |
| V.Large | 1M   | 5k     | 50     | 1.6ms | 7.0ms| 4.4x  |

Numba is 2-7x faster at `get_distribution`.

### But...

Profile of the **full algorithm** (`get_approx_reduct_greedy_heuristic`, 5000
objects) reveals that `get_distribution` is **only 6.5% of total runtime**:

| Operation | Time | % of total |
|-----------|------|------------|
| `get_disorder_score` (incl. disorder_fun) | 0.28s | 54% |
| `split` | 0.16s | 30% |
| `compress` (pandas C) | 0.10s | 20% |
| **`get_distribution` (np.add.at)** | **0.03s** | **6.5%** |

The bottleneck is `split` + `compress` (pandas/numpy C-level), not
`get_distribution`.

## Experiment 2: `disorder_measures` (gini_impurity, entropy, conflicts_count)

### Setup

Pure-numpy equivalents written for each measure. Benchmarked with cold-start
compilation done before timing.

### Results

| Matrix size `(groups, values)` | gini_impurity | entropy | conflicts_count |
|---|---:|---:|---:|
| (5, 3) | **13.8x** | **14.1x** | **4.8x** |
| (500, 10) | **11.7x** | **4.5x** | **8.6x** |
| (2000, 20) | **11.0x** | **2.3x** | **5.3x** |
| (10000, 50) | **6.4x** | **3.8x** | **6.1x** |

Numba consistently 2-14x faster. However absolute times are still in
microseconds -- even the worst case (entropy, 10000×50) is 5.3ms pure vs
1.4ms numba.

## Key Finding: Pure Is Not Truly Pure

When selecting `group_index_class="pure"`, only `get_distribution` uses the
pure-numpy path. The disorder measures (`gini_impurity`, `entropy`,
`conflicts_count`) called by `get_disorder_score` always use numba -- there
is no pure fallback implementation.

## Profile: Numba JIT Compilation Overhead

Even with `group_index_class="pure"`, the profile shows **0.24s spent on
numba JIT compilation** (triggered by `gini_impurity`). For the numba
GroupIndex, this is **0.20s** (with additional recompilations for varying
matrix shapes).

On a 0.5s total run for 5000 objects, JIT compilation is **40-50% of
runtime**.

## Total Cost-Benefit (5000 objects run)

| Component | Numba time | Hypothetical all-pure | Delta |
|-----------|-----------|----------------------|-------|
| JIT compilation (one-time) | 0.20s | 0s | numba loses 0.20s |
| `get_distribution` (1203 calls) | 0.004s | 0.033s | numba gains 0.03s |
| `gini_impurity` (1203 calls) | ~0.007s | ~0.14s | numba gains 0.13s |
| `split` + `compress` | 0.26s | 0.26s | tie |
| **Total** | **~0.47s** | **~0.43s** | **pure would be ~0.04s faster** |

For small/medium data the JIT compilation overhead cancels out all numba
gains. For very large datasets (where computation dominates), numba would win.

## Conclusions

1. **Numba provides real speedups** (2-14x) on the functions it accelerates.

2. **JIT compilation overhead is significant** for small runs (~0.2s
   one-time cost), often exceeding the computational benefit.

3. **Removing numba entirely would require a major refactor** -- 16
   functions in 7 files. For `disorder_measures` the pure-numpy
   implementations exist (this experiment) but are not wired in.

4. **The status quo is reasonable**: keep numba but understand it is not a
   free lunch -- compilation overhead dominates on small data, computational
   benefit dominates on large data.

5. **GroupIndex `pure` is a misnomer** -- it only avoids numba for
   `get_distribution`, but the disorder measures always use numba regardless
   of the chosen GroupIndex class.

## Generated Artifacts

- Benchmark script: `/tmp/opencode/bench_group_index.py`
- Profile script: `/tmp/opencode/profile_algorithm.py`
- Disorder measures benchmark: `/tmp/opencode/bench_disorder_measures.py`
