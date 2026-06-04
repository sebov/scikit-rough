# Pending Items & Resume Instructions

Items from `tmp/phd/thesis.tex` not yet added to the knowledge base. Checked off as they are
ingested.

## Resume Instructions

### What we are doing

Ingesting `tmp/phd/thesis.tex` (PhD dissertation on decision bireducts in rough set theory) into
`kb/` following the schema defined in `kb/AGENTS.md`.

### Current state (2026-06-04)

- **Concepts**: 21 files in `kb/concepts/` — Chapters 1–4 (Preliminaries, Foundations of Decision
  Reducts, Foundations of Decision Bireducts, Algorithms) fully processed. All definitions extracted.
- **Propositions**: 3 files in `kb/propositions/` — first three propositions from thesis extracted as
  standalone files. Preference: create standalone proposition files (not inline), preserve proofs,
  reference-style proofs (like thesis) are acceptable.
- **Examples**: 4 files in `kb/examples/` — golf dataset tables (reduct rules, gamma-reduct rules,
  bireduct rules, complete bireduct listing). Examples were verified against original LaTeX sources.
- **Notation**: `kb/notation.md` contains 47 symbols registered from the thesis preamble. This is the
  canonical notation registry — all new files must use these conventions.
- **Index & log**: `kb/index.md` and `kb/log.md` are up to date.

### How to resume

1. Read this file (pending.md) for the task checklist.
2. Review `kb/AGENTS.md` for schema rules (required before any operation).
3. Check `kb/notation.md` for symbol conventions before writing any math.
4. Use `kb/template.md` for file structure.
5. Source material is in `tmp/phd/thesis.tex` (6572 lines). Custom LaTeX commands decoded in
   `kb/notation.md` bottom section.
6. Reference bibliography: `tmp/phd/thesisbib.bib`. Always verify citation titles against the bib
   file.
7. Old knowledgebase at `knowledgebase_old/` can serve as reference/comparison but is not
   authoritative.

### Key user preferences

- **Language**: user communicates in Polish; all KB output in English.
- **Propositions**: standalone files in `kb/propositions/` with proofs preserved. Reference-based
  proofs OK but prefer explanatory commentary alongside citations. Inline a brief summary + link in
  the relevant concept file.
- **Citation titles**: always verify against `tmp/phd/thesisbib.bib`. Do not invent or paraphrase.
- **Examples**: small (single-table) → inline in concept `## Example`; complex (multi-table) →
  standalone in `kb/examples/`. Faithfully reproduce source data line by line. Never summarise
  counts or invent sets when condensing tables -- prefer completeness over brevity.
- **Cross-checking**: compare against `knowledgebase_old/` and original LaTeX sources. Flag
  discrepancies.
- **No content from** `trash/` or `dev/` directories.

### Next steps (priority order)

1. Continue extracting propositions from thesis (next: `prop:gamma_decision_reduct_boolean_formula`
   at L1700, then bireduct-related propositions).
2. Extract remaining examples (epsilon-bireducts, ensembles, permutations).
3. Process remaining chapters (Case Study, Feature Importance, Conclusions, Appendices).
4. Periodic lint checks as KB grows.

---

## Examples (not yet extracted from thesis include files)

- [ ] `m_decision_epsilon_reducts_decision_epsilon_bireducts_all.tex` --
  $\varepsilon$-bireducts and $M$-reducts for golf ($\varepsilon = 4/14$). Illustrates
  `concept-epsilon-decision-bireduct`.
- [ ] `ensembles_decision_epsilon_bireducts.tex` -- 3-element correct ensembles of
  $\varepsilon$-bireducts. Illustrates `concept-bireduct-ensemble`.
- [ ] `decision_bireducts_from_permutations.tex` -- results of permutation algorithm.
- [ ] `gamma_decision_bireducts_from_permutations.tex` -- gamma-bireduct permutation results.
- [ ] `decision_bireducts_cnf_dnf.tex` -- CNF/DNF Boolean formulae for decision bireducts.
- [ ] `gamma_decision_bireducts_cnf_dnf.tex` -- CNF/DNF for gamma-decision bireducts.
- [ ] `golf_dataset_diagonal.tex` -- diagonal table transformation.
- [ ] `temporal_bireducts.tex` -- temporal bireduct computation walkthrough.
- [ ] `nphard_graph_gamma.tex` / `nphard_graph_m.tex` -- NP-hardness construction figures.

## Propositions (from thesis.tex, not yet created as standalone files)

Many short propositions were inline'd into concept files. Those with substantial proofs
(> 20 lines) or multi-reference should be standalone.

- [x] `prop:indiscernibility_eqivalence_relation` (L1315) -- IND(B) is equivalence relation.
  Short. Currently inline in `concept-indiscernibility`. **→ created as
  `prop-indiscernibility-equivalence-relation`**.
- [x] `prop:decision_reduct_boolean_formula` (L1515) -- Boolean formula characterization of
  decision reducts. **→ created as `prop-decision-reduct-boolean-formula`**.
- [x] `prop:gamma_decision_reduct_inconsistent_decision_table` (L1683) -- Gamma-reduct iff
  reduct in modified consistent table. **→ created as
  `prop-gamma-decision-reduct-consistent-table`**.
- [ ] `prop:gamma_decision_reduct_boolean_formula` (L1700) -- Boolean formula for
  gamma-decision reducts. Inline in `concept-gamma-decision-reduct`.
- [ ] `prop:monotony_properties` (L2286) -- Monotonicity of functional dependency. Inline in
  `concept-decision-bireduct`.
- [ ] `prop:decision_reduct_iff_bireduct` (L2313) -- Reduct iff U-bireduct. Inline.
- [ ] `prop:decision_bireduct_iff_reduct` (L2330) -- Bireduct via subtable consistency. Inline.
- [ ] `prop:bireduct_objects_and_rules` (L2376) -- Rules interpretation of bireducts. Inline.
- [ ] `prop:gamma_monotony_properties` (L2480) -- Gamma monotonicity. Inline.
- [ ] `prop:gamma_decision_bireduct_to_reduct` (L2497) -- Gamma-bireduct with U iff reduct.
  Inline.
- [ ] `prop:gamma_decision_bireduct_pos` (L2528) -- Gamma-bireduct iff X = POS(B). Long proof.
  Inline.
- [ ] `prop:decision_bireduct_boolean_formula` (L2681) -- Boolean formula for bireducts.
  **Very long proof** (~130 lines). Needs standalone file.
- [ ] `prop:decision_table_diagonal` (L2870) -- Diagonal transformation. Inline.
- [ ] `prop:gamma_decision_bireduct_boolean_formula` (L2946) -- Boolean formula for
  gamma-bireducts. Inline.
- [ ] `prop:smallest_m_decision_epsilon_reduct_decision_epsilon_bireduct` (L3145) --
  Correspondence between M-reducts and epsilon-bireducts. Inline.
- [ ] `prop:minimal_decision_epsilon_bireduct_problem` (L3209) -- NP-hardness proof. Inline
  (referenced).
- [ ] `prop:ensemble_np` (L3468) -- NP-hardness of SCDBEP. Inline (proof present).
- [ ] `prop:decision_bireduct_ordering` (L3586) -- Ordering algorithm correctness. Long proof.
  Inline.
- [ ] `prop:gamma_decision_bireduct_ordering` (L3741) -- Gamma ordering algorithm. Inline.
- [ ] `prop:decision_bireduct_sampling` (L3844) -- Sampling algorithm correctness. Inline.
- [ ] `prop:gamma_decision_bireduct_sampling` (L4012) -- Gamma sampling algorithm. Inline.
- [ ] `prop:temporal_bireduct_computation` (L4492) -- Temporal bireduct computation. Inline.

## NP-hardness Propositions (Chapter 2, all inline currently)

- [ ] `prop:minimal_dominating_set_problem` (L1906) -- NP-hardness of MDS.
- [ ] `prop:minimal_relative_gamma_decision_epsilon_reduct_problem` (L1915) -- Long proof.
- [ ] `prop:minimal_gamma_decision_epsilon_reduct_problem` (L2035)
- [ ] `prop:alpha_dominating_set_problem` (L2067)
- [ ] `prop:minimal_relative_m_decision_epsilon_reduct_problem` (L2076) -- Long proof.
- [ ] `prop:minimal_m_decision_epsilon_reduct_problem` (L2164)
- [ ] `prop:minimal_relative_r_decision_epsilon_reduct_problem` (L2180)
- [ ] `prop:minimal_r_decision_epsilon_reduct_problem` (L2197)

## Remaining Chapters (not yet processed at all)

- [ ] Chapter 5: Case Study (ch:case_study)
- [ ] Chapter 6: Feature Importance and Ranks (ch:feature_importance_and_ranks)
- [ ] Chapter 7: Conclusions (ch:conclusions)
- [ ] Appendices (appendix_feature_importance_profiling, appendix_notebook_examples)
