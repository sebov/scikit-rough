# Ingestion Tracking

Items from `tmp/phd/thesis.tex` ingested into the knowledge base. Checked off as completed.

**Status: COMPLETE.** All definitions, propositions, and examples from the thesis have been extracted.

## Context & Instructions

### What we did

Ingested `tmp/phd/thesis.tex` (PhD dissertation on decision bireducts in rough set theory) into
`kb/` following the schema defined in `kb/AGENTS.md`. Completed 2026-06-07.

### Final state (2026-06-07)

- **Concepts**: 21 files in `kb/concepts/`.
- **Propositions**: 30 files in `kb/propositions/` -- all propositions from thesis extracted.
- **Examples**: 13 files in `kb/examples/` -- all examples from thesis and include files extracted.
- **Source Summaries**: 1 file in `kb/sources/`.
- **Notation**: `kb/notation.md` contains 47 symbols.
- **Total**: 65 wiki pages.
- **Notation**: `kb/notation.md` contains 47 symbols registered from the thesis preamble. This is the
  canonical notation registry -- all new files must use these conventions.
- **Index & log**: `kb/index.md` and `kb/log.md` are up to date.

### How to resume

1. Read this file (ingestion.md) for the task checklist.
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
  proofs OK (when thesis cites external source) but prefer explanatory commentary alongside
  citations. Inline a brief summary + link in the relevant concept file.
- **Proofs**: preserve thesis proofs faithfully in terms of **completeness**, not literal wording.
  Key requirement: no gaps, no skipped cases, no hand-waving. All branches must be checked, all
  non-trivial steps justified. "It is obvious" or "it follows directly" are acceptable when the
  step genuinely follows from a definition or prior result without additional reasoning -- but
  never when the step requires a non-trivial argument. When thesis proof is detailed and
  step-by-step, preserve that level of detail. When thesis proof cites external source (e.g.,
  "See Skowron & Rauszer 1992"), keep the citation but add explanatory commentary about the
  construction's intuition. If thesis proof contains errors, flag them and correct in KB
  (correctness > faithfulness).
- **Citation titles**: always verify against `tmp/phd/thesisbib.bib`. Do not invent or paraphrase.
- **Examples**: small (single-table) → inline in concept `## Example`; complex (multi-table) →
  standalone in `kb/examples/`. Faithfully reproduce source data line by line. Never summarise
  counts or invent sets when condensing tables -- prefer completeness over brevity.
- **Cross-checking**: compare against `knowledgebase_old/` and original LaTeX sources. Flag
  discrepancies.
- **No content from** `trash/` or `dev/` directories.

### Next steps

All items from thesis.tex have been extracted. For future ingestions from new source files,
use this file as a template: add new sections for the new source, track progress with checkboxes,
and update the final state summary.

---

## Examples (not yet extracted from thesis include files)

- [x] `m_decision_epsilon_reducts_decision_epsilon_bireducts_all.tex` --
  $\varepsilon$-bireducts and $M$-reducts for golf ($\varepsilon = 4/14$). Illustrates
  `concept-epsilon-decision-bireduct`. **→ created as `ex-golf-epsilon-bireducts-m-reducts`.**
- [x] `ensembles_decision_epsilon_bireducts.tex` -- 3-element correct ensembles of
  $\varepsilon$-bireducts. Illustrates `concept-bireduct-ensemble`. **→ created as
  `ex-golf-epsilon-bireduct-ensembles`.**
- [x] `decision_bireducts_from_permutations.tex` -- results of permutation algorithm. **→ created
  as `ex-golf-permutation-bireducts`.**
- [x] `gamma_decision_bireducts_from_permutations.tex` -- gamma-bireduct permutation results. **→
  created as `ex-golf-permutation-gamma-bireducts`.**
- [x] `decision_bireducts_cnf_dnf.tex` -- CNF/DNF Boolean formulae for decision bireducts. **→
  created as `ex-golf-bireduct-cnf-dnf`.**
- [x] `gamma_decision_bireducts_cnf_dnf.tex` -- CNF/DNF for gamma-decision bireducts. **→ created
  as `ex-golf-gamma-bireduct-cnf-dnf`.**
- [x] `golf_dataset_diagonal.tex` -- diagonal table transformation. **→ created as
  `ex-golf-diagonal-table`.**
- [x] `temporal_bireducts.tex` -- temporal bireduct computation walkthrough. **→ created as
  `ex-temporal-bireduct-walkthrough`.**
- [x] `nphard_graph_gamma.tex` / `nphard_graph_m.tex` -- NP-hardness construction figures. **→
  created as `ex-nphard-construction-tables`.**

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
- [x] `prop:gamma_decision_reduct_boolean_formula` (L1700) -- Boolean formula for
  gamma-decision reducts. **→ created as `prop-gamma-decision-reduct-boolean-formula`**.
- [x] `prop:monotony_properties` (L2286) -- Monotonicity of functional dependency. **→ created as
  `prop-monotony-properties`**.
- [x] `prop:decision_reduct_iff_bireduct` (L2313) -- Reduct iff U-bireduct. Inline. **→ created as
  `prop-decision-reduct-iff-bireduct`**.
- [x] `prop:decision_bireduct_iff_reduct` (L2330) -- Bireduct via subtable consistency. Inline.
  **→ created as `prop-decision-bireduct-iff-reduct`**.
- [x] `prop:bireduct_objects_and_rules` (L2376) -- Rules interpretation of bireducts. Inline.
  **→ created as `prop-bireduct-objects-and-rules`**.
- [x] `prop:gamma_monotony_properties` (L2480) -- Gamma monotonicity. Inline. **→ created as
  `prop-gamma-monotony-properties`**.
- [x] `prop:gamma_decision_bireduct_to_reduct` (L2497) -- Gamma-bireduct with U iff reduct.
  Inline. **→ created as `prop-gamma-decision-bireduct-to-reduct`**.
- [x] `prop:gamma_decision_bireduct_pos` (L2528) -- Gamma-bireduct iff X = POS(B). Long proof.
  Inline. **→ created as `prop-gamma-decision-bireduct-pos`**.
- [x] `prop:decision_bireduct_boolean_formula` (L2681) -- Boolean formula for bireducts.
  **Very long proof** (~130 lines). Needs standalone file. **→ created as
  `prop-decision-bireduct-boolean-formula`**.
- [x] `prop:decision_table_diagonal` (L2870) -- Diagonal transformation. Inline. **→ created as
  `prop-decision-table-diagonal`**.
- [x] `prop:gamma_decision_bireduct_boolean_formula` (L2946) -- Boolean formula for
  gamma-bireducts. Inline. **→ created as `prop-gamma-decision-bireduct-boolean-formula`**.
- [x] `prop:smallest_m_decision_epsilon_reduct_decision_epsilon_bireduct` (L3145) --
  Correspondence between M-reducts and epsilon-bireducts. Inline. **→ created as
  `prop-m-reduct-epsilon-bireduct-correspondence`**.
- [x] `prop:minimal_decision_epsilon_bireduct_problem` (L3209) -- NP-hardness proof. Inline
  (referenced). **→ created as `prop-minimal-epsilon-bireduct-np-hard`**.
- [x] `prop:ensemble_np` (L3468) -- NP-hardness of SCDBEP. Inline (proof present). **→ created as
  `prop-ensemble-np-hard`**.
- [x] `prop:decision_bireduct_ordering` (L3586) -- Ordering algorithm correctness. Long proof.
  **→ created as `prop-decision-bireduct-ordering`**.
- [x] `prop:gamma_decision_bireduct_ordering` (L3741) -- Gamma ordering algorithm. Inline.
  **→ created as `prop-gamma-decision-bireduct-ordering`**.
- [x] `prop:decision_bireduct_sampling` (L3844) -- Sampling algorithm correctness. Inline.
  **→ created as `prop-decision-bireduct-sampling`**.
- [x] `prop:gamma_decision_bireduct_sampling` (L4012) -- Gamma sampling algorithm. Inline.
  **→ created as `prop-gamma-decision-bireduct-sampling`**.
- [x] `prop:temporal_bireduct_computation` (L4492) -- Temporal bireduct computation. Inline.
  **→ created as `prop-temporal-bireduct-computation`**.

## NP-hardness Propositions (Chapter 2, all inline currently)

- [x] `prop:minimal_dominating_set_problem` (L1906) -- NP-hardness of MDS. **→ created as
  `prop-minimal-dominating-set-np-hard`**.
- [x] `prop:minimal_relative_gamma_decision_epsilon_reduct_problem` (L1915) -- Long proof. **→
  created as `prop-relative-gamma-epsilon-reduct-np-hard`**.
- [x] `prop:minimal_gamma_decision_epsilon_reduct_problem` (L2035) **→ created as
  `prop-gamma-epsilon-reduct-np-hard`**.
- [x] `prop:alpha_dominating_set_problem` (L2067) **→ created as `prop-alpha-dominating-set-np-hard`**.
- [x] `prop:minimal_relative_m_decision_epsilon_reduct_problem` (L2076) -- Long proof. **→ created
  as `prop-relative-m-epsilon-reduct-np-hard`**.
- [x] `prop:minimal_m_decision_epsilon_reduct_problem` (L2164) **→ created as
  `prop-m-epsilon-reduct-np-hard`**.
- [x] `prop:minimal_relative_r_decision_epsilon_reduct_problem` (L2180) **→ created as
  `prop-relative-r-epsilon-reduct-np-hard`**.
- [x] `prop:minimal_r_decision_epsilon_reduct_problem` (L2197) **→ created as
  `prop-r-epsilon-reduct-np-hard`**.

## Remaining Chapters (intentionally skipped)

- [x] Chapter 5: Case Study (ch:case_study) -- experimental results, no new definitions or propositions.
- [x] Chapter 6: Feature Importance and Ranks (ch:feature_importance_and_ranks) -- experimental rankings, no new formal content.
- [x] Chapter 7: Conclusions (ch:conclusions) -- summary, no new definitions or propositions.
- [x] Appendices (appendix_feature_importance_profiling, appendix_notebook_examples) -- profiling + notebook examples, no new theory.

All definitions (`\label{def:}`) and propositions (`\label{prop:}`) from thesis.tex end at line
4511. Chapters 5-7 and appendices contain only experimental results, case studies, and concluding
remarks.

**Ingestion of `tmp/phd/thesis.tex` is complete.**

## Proof Gaps & Open Issues

No unresolved proof gaps. The previously flagged gap in
`prop-temporal-bireduct-computation` (backward non-extendability) was investigated and found to be
valid -- see `prop-temporal-bireduct-computation.md` for the tightened argument.

## Session Reflections (2026-06-06)

### What worked well

- **Standalone proposition files + inline summaries in concept files.** For each extracted
  proposition, a summary + link was added to the relevant concept file (when applicable). This
  keeps concept files under the 150-line limit while making propositions easily discoverable.
- **Proof Strategy sections.** For complex proofs (e.g., NP-hardness reductions), adding an
  explicit `## Proof Strategy` section before the `## Proof` made the reasoning structure
  immediately clear.
- **Per-batch verification.** Checking all proofs for correctness/completeness in each batch
  caught one arithmetic error (`prop-ensemble-np-hard`: $n+(n-1) \to 1+(n-1)$) and one precision
  issue ("dominating set" $\to$ "minimal dominating set").
- **Immediate flagging.** When a gap was found (`prop-temporal-bireduct-computation`), flagging
  it in-line with a `> **Proof gap (flagged).**` blockquote preserves both the thesis proof and
  the critique. The gap was later investigated (2026-06-07) and confirmed false -- the proof is
  correct; the flag was removed and the proof tightened.

### Patterns worth repeating

- **Three‑pass verification**: (1) check statement matches thesis label, (2) verify each logical
  step has a justification, (3) stress-test edge cases (e.g., empty sets, boundary indices).
- **Cost vs construction size** (NP-hardness proofs): always verify that the cost function used
  in the reduction is the intended one, not the raw construction size.
- **"Minimal" vs "any"** in dominating set reductions: step (⇒) works for any dominating set;
  minimality is only needed in step (⇐). Mixing these up creates false proof requirements.
- **Gamma-analogy proofs**: when thesis says "proof is analogous", explicitly verify that every
  referenced lemma has a gamma counterpart (e.g., monotonicity, dependency definition).

### Caveats for future sessions

- **`prop:gamma_decision_bireduct_ordering`** and **`prop:gamma_decision_bireduct_sampling`**:
  the gamma versions of ordering/sampling propositions have very brief thesis proofs (referencing
  the standard case). While the analogies are valid, the KB files would benefit from more
  explicit step-by-step expansion.
- **NP-hardness propositions (L1906-L2197)**: DONE (2026-06-07). All 8 extracted as standalone
  files forming a complete reduction chain: MDS → relative γ → γ → α-MDS → relative M → M →
  relative R → R. Concept files updated with inline summaries.
- **Examples (epsilon-bireducts, ensembles, permutations)**: DONE (2026-06-07). All 9 examples
  extracted from `tmp/phd/include/` as standalone files.
