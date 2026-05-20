# Presentation Structure

**Title:** Algorithms for Approximate Reducts: Iterative Attribute Selection
**Duration:** ~75 minutes
**Audience:** Students with basic rough set knowledge; guest lecture in a theory-heavy series
**Angle:** Practical engineering perspective -- algorithms, complexity, optimization, SWE practices

## Resolved Decisions

- Single notebook rendered via Quarto/revealjs; participant notebook provided separately
- Code on slides is concise visual support, not meant to be read line-by-line
- Golf dataset (14 objects, 4 attrs) for definitions and reduct result demo
  - epsilon=0.2, entropy -> reduct: [Outlook, Humidity, Wind] (9/10 times)
  - Feature importance: Outlook dominates across ensembles
- Seismic dataset (133150 objects, 738 attrs) for profiling/benchmarks/end-to-end
  - From AAIA'16 Data Mining Challenge (seismic event prediction)
  - Greedy: 100 reducts, avg length 16.38; DAAR: 100 reducts, avg length 11.89
  - Classification: greedy reducts ensemble BAC=0.844, DT=0.834, XGBoost=0.824
  - VotingClassifier with reduct-triplet feature selection: BAC=0.858 (best)
- Hook architecture moved to bonus (slide 11, if time permits)
- Narrative arc: problem -> why it matters -> bottleneck -> GroupIndex solutions -> end-to-end results
- GroupIndex approaches order: lazy -> dict/dict_numba -> hash -> pure -> numba
- Each approach gets: explanation + complexity + pseudocode + benchmark plot (progressive lines)
- Benchmark: dataset sizes [5000, 10000, 20000, 25000, 50000, 75000, 100000, 133150], candidates=30, reducts=20, n_jobs=-1
- End-to-end: real reduct computation on Seismic dataset, snakeviz lazy vs. numba (different sizes, shown separately)
- snakeviz screenshots: profile_output/greedy_heuristic_lazy.png (smaller data), greedy_heuristic_numba.png (full data)
- Background: disorder measures directly (entropy, gini, conflicts), not gamma/POS
- Approximate reduct: disorder_score(B) <= threshold where threshold = total + epsilon * (base - total)
- Profiling: cProfile + snakeviz, greedy heuristic

## Slide Outline

1. **Title slide** -- title, workshop context, speaker
2. **Agenda** -- what we will cover
3. **Background** (3 slides):
   - 3a: Decision table (U, A ∪ {d}), Weather as example
   - 3b: Indiscernibility, equivalence classes, disorder measures (entropy/gini/conflicts)
   - 3c: Approximate reduct definition -- disorder_score(B) <= threshold
4. **Result of reduct computation** -- Golf dataset, concrete output showing what the algorithm produces
5. **Why it matters** -- classification gain + feature importance (brief, motivating)
6. **What takes long** -- snakeviz heatmap/profile showing the bottleneck
7. **GroupIndex in context** -- what it is in the greedy pseudo-algorithm, its API
8. **Approach categories** -- stateless/lazy vs stateful; within stateful: forward index vs inverted index
9. **Approach deep-dives** (one slide each, progressive):
   - Explanation of how it works (with concrete example)
   - Complexity analysis
   - Pseudocode of the key operation
   - Benchmark plot: time vs. number of objects (adds one line per approach)
10. **End-to-end experiments** -- real reduct computation on real dataset; snakeviz comparison: best vs. initial approach
11. **Bonus: hook architecture** (if time) -- ProcessingMultiStage, building algorithms from hooks, DAAR mention
12. **Closing / thanks**

## Open Questions

- [ ] Benchmark design: exact subset sizes for x-axis (e.g., 500, 1000, 2500, 5000, 10000, 20000?)
- [ ] Which GroupIndex implementations to compare (lazy, numba, others?)
- [ ] snakeviz screenshots: wait for .prof files to finish, then extract
- [ ] Slide 4 (reduct result): exact format of the output table/slide from Golf example
- [ ] Slide 5 (why it matters): which classification numbers to highlight (BAC comparison chart?)
- [ ] Slide 6 (what takes long): snakeviz flame chart or icicle chart? which .prof file?
- [ ] Slide 8 (approach categories): confirm the exact categorization (lazy/stateless, numba/forward, inverted?)

## Checkpoint

To resume grilling: load `grill-with-docs` skill and continue from this file. Key files:
- `presentation/structure.md` (this file)
- `presentation/CONTEXT.md` (glossary)
- `presentation/presentation.ipynb` (the notebook being built)
