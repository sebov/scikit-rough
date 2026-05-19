# Presentation Structure

**Title:** Algorithms for Approximate Reducts: Iterative Attribute Selection
**Duration:** ~75 minutes
**Audience:** Students with basic rough set knowledge; guest lecture in a theory-heavy series
**Angle:** Practical engineering perspective -- algorithms, complexity, optimization, SWE practices

## Resolved Decisions

- Single notebook rendered via Quarto/revealjs; participant notebook provided separately
- Code on slides is concise visual support, not meant to be read line-by-line
- Small example table (Weather/Golf) on slides for definitions and algorithm
- Larger real dataset for profiling/benchmarks -- results shown only as charts, not raw data
- Simple pseudocode first, then annotated with hook points across multiple slides (Variant A)
- GroupIndex narrative: sort-based -> hash+compress -> bags (progressive optimization)
- DAAR: mentioned only briefly (1 slide) as proof of modular architecture
- Classifiers: stretch goal; thesis is ensembles maintain/improve quality despite fewer attrs
- Background: disorder measures directly (entropy, gini, conflicts), not gamma/POS
- Approximate reduct: disorder_score(B) <= threshold where threshold = total + epsilon * (base - total)
- Profiling: full pipeline with GroupIndex as bottleneck (not GroupIndex-internal breakdown)
- Benchmarks: line chart -- time vs. dataset size for 3 GroupIndex implementations

## Slide Outline

1. **Title slide** -- title, workshop context, speaker
2. **Agenda** -- what we will cover
3. **Background** (3 slides):
   - 3a: Decision table (U, A ∪ {d}), Weather as example
   - 3b: Indiscernibility, equivalence classes, disorder measures (entropy/gini/conflicts)
   - 3c: Approximate reduct definition -- disorder_score(B) <= threshold
4. **Greedy algorithm -- simple pseudocode** -- gain() as black box; progressive build-up across slides ("animation") -- pseudocode first, GroupIndex later
5. **Modular extension: hook architecture** -- 1 slide: annotated pseudocode + ProcessingMultiStage diagram + DAAR mention
6. **Pipeline in action** -- Weather example, at least one iteration step by step
7. **Performance profiling of full pipeline** -- chart showing GroupIndex dominates runtime
8. **GroupIndex: implementations** -- sort-based (baseline), hash+compress (current), bags (experimental); line chart: time vs. dataset size
9. **GroupIndex in full pipeline** -- end-to-end timing with each variant, overall speedup
10. **Application: classifiers** (stretch goal) -- ensembles of reducts maintain/improve quality
11. **Closing / thanks**

## Open Questions

- [ ] Which large dataset to use for benchmarks
- [ ] Use specific classifier example (to be provided later)

## Checkpoint

To resume grilling: load `grill-with-docs` skill and continue from this file. Key files:
- `presentation/structure.md` (this file)
- `presentation/CONTEXT.md` (glossary)
- `presentation/presentation.ipynb` (the notebook being built)
