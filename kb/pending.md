# Pending Items

Items from `tmp/phd/thesis.tex` not yet added to the knowledge base. Checked off as they are
ingested.

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
- [ ] `prop:decision_reduct_boolean_formula` (L1515) -- Boolean formula characterization of
  decision reducts. Inline in `concept-decision-reduct`.
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
