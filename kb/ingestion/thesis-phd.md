# Ingestion: PhD Thesis on Decision Bireducts

Source: `tmp/phd/thesis.tex` (6572 lines with includes in `tmp/phd/include/`).
Source-summary: `src-thesis-phd` (see `kb/sources/thesis-phd.md`).

**Status: COMPLETE.** All definitions, propositions, and examples from the thesis have been extracted.

---

## Final State (2026-06-07)

- **Concepts**: 21 files in `kb/concepts/`.
- **Propositions**: 30 files in `kb/propositions/` -- all propositions from thesis extracted.
- **Examples**: 13 files in `kb/examples/` -- all examples from thesis and include files extracted.
- **Notation**: `kb/notation.md` contains 47 symbols registered from the thesis preamble.
- **Total**: 64 wiki pages extracted from this source.

---

## Context & Instructions

### What We Did

Ingested the PhD thesis on decision bireducts in rough set theory into `kb/` following the
schema defined in `kb/AGENTS.md`. Completed 2026-06-07.

### How to Resume

1. Review `kb/AGENTS.md` for schema rules (required before any operation).
2. Check `kb/notation.md` for symbol conventions before writing any math.
3. Use `kb/template.md` for file structure.
4. Source provenance: all wiki pages from this source reference `src-thesis-phd`.
5. Old knowledgebase at `knowledgebase_old/` can serve as reference/comparison but is not
   authoritative.
6. General guidelines for proof handling, examples, and verification: see `kb/ingestion_guidelines.md`.

### Key User Preferences

> These are decisions specific to the thesis ingestion. General guidelines for proof handling,
> examples, and verification are in `kb/ingestion_guidelines.md`.

- **Language**: user communicates in Polish; all KB output in English.
- **Propositions**: standalone files in `kb/propositions/` with proofs preserved. Reference-based
  proofs OK (when thesis cites external source) but prefer explanatory commentary alongside
  citations. Inline a brief summary + link in the relevant concept file.
- **Proofs**: preserve thesis proofs faithfully in terms of **completeness**, not literal wording.
  Key requirement: no gaps, no skipped cases, no hand-waving.
- **Citation titles**: always verify against `tmp/phd/thesisbib.bib`. Do not invent or paraphrase.
- **Examples**: small (single-table) -> inline in concept `## Example`; complex (multi-table) ->
  standalone in `kb/examples/`. Faithfully reproduce source data line by line.
- **Cross-checking**: compare against `knowledgebase_old/` and original LaTeX sources. Flag
  discrepancies.
- **No content from** `trash/` or `dev/` directories.

---

## Extraction Checklist

### Examples (from thesis include files)

- [x] `m_decision_epsilon_reducts_decision_epsilon_bireducts_all.tex` -> `ex-golf-epsilon-bireducts-m-reducts`
- [x] `ensembles_decision_epsilon_bireducts.tex` -> `ex-golf-epsilon-bireduct-ensembles`
- [x] `decision_bireducts_from_permutations.tex` -> `ex-golf-permutation-bireducts`
- [x] `gamma_decision_bireducts_from_permutations.tex` -> `ex-golf-permutation-gamma-bireducts`
- [x] `decision_bireducts_cnf_dnf.tex` -> `ex-golf-bireduct-cnf-dnf`
- [x] `gamma_decision_bireducts_cnf_dnf.tex` -> `ex-golf-gamma-bireduct-cnf-dnf`
- [x] `golf_dataset_diagonal.tex` -> `ex-golf-diagonal-table`
- [x] `temporal_bireducts.tex` -> `ex-temporal-bireduct-walkthrough`
- [x] `nphard_graph_gamma.tex` / `nphard_graph_m.tex` -> `ex-nphard-construction-tables`

### Propositions (from thesis.tex)

- [x] `prop:indiscernibility_eqivalence_relation` (L1315) -> `prop-indiscernibility-equivalence-relation`
- [x] `prop:decision_reduct_boolean_formula` (L1515) -> `prop-decision-reduct-boolean-formula`
- [x] `prop:gamma_decision_reduct_inconsistent_decision_table` (L1683) -> `prop-gamma-decision-reduct-consistent-table`
- [x] `prop:gamma_decision_reduct_boolean_formula` (L1700) -> `prop-gamma-decision-reduct-boolean-formula`
- [x] `prop:monotony_properties` (L2286) -> `prop-monotony-properties`
- [x] `prop:decision_reduct_iff_bireduct` (L2313) -> `prop-decision-reduct-iff-bireduct`
- [x] `prop:decision_bireduct_iff_reduct` (L2330) -> `prop-decision-bireduct-iff-reduct`
- [x] `prop:bireduct_objects_and_rules` (L2376) -> `prop-bireduct-objects-and-rules`
- [x] `prop:gamma_monotony_properties` (L2480) -> `prop-gamma-monotony-properties`
- [x] `prop:gamma_decision_bireduct_to_reduct` (L2497) -> `prop-gamma-decision-bireduct-to-reduct`
- [x] `prop:gamma_decision_bireduct_pos` (L2528) -> `prop-gamma-decision-bireduct-pos`
- [x] `prop:decision_bireduct_boolean_formula` (L2681) -> `prop-decision-bireduct-boolean-formula`
- [x] `prop:decision_table_diagonal` (L2870) -> `prop-decision-table-diagonal`
- [x] `prop:gamma_decision_bireduct_boolean_formula` (L2946) -> `prop-gamma-decision-bireduct-boolean-formula`
- [x] `prop:smallest_m_decision_epsilon_reduct_decision_epsilon_bireduct` (L3145) -> `prop-m-reduct-epsilon-bireduct-correspondence`
- [x] `prop:minimal_decision_epsilon_bireduct_problem` (L3209) -> `prop-minimal-epsilon-bireduct-np-hard`
- [x] `prop:ensemble_np` (L3468) -> `prop-ensemble-np-hard`
- [x] `prop:decision_bireduct_ordering` (L3586) -> `prop-decision-bireduct-ordering`
- [x] `prop:gamma_decision_bireduct_ordering` (L3741) -> `prop-gamma-decision-bireduct-ordering`
- [x] `prop:decision_bireduct_sampling` (L3844) -> `prop-decision-bireduct-sampling`
- [x] `prop:gamma_decision_bireduct_sampling` (L4012) -> `prop-gamma-decision-bireduct-sampling`
- [x] `prop:temporal_bireduct_computation` (L4492) -> `prop-temporal-bireduct-computation`

### NP-Hardness Propositions (Chapter 2)

- [x] `prop:minimal_dominating_set_problem` (L1906) -> `prop-minimal-dominating-set-np-hard`
- [x] `prop:minimal_relative_gamma_decision_epsilon_reduct_problem` (L1915) -> `prop-relative-gamma-epsilon-reduct-np-hard`
- [x] `prop:minimal_gamma_decision_epsilon_reduct_problem` (L2035) -> `prop-gamma-epsilon-reduct-np-hard`
- [x] `prop:alpha_dominating_set_problem` (L2067) -> `prop-alpha-dominating-set-np-hard`
- [x] `prop:minimal_relative_m_decision_epsilon_reduct_problem` (L2076) -> `prop-relative-m-epsilon-reduct-np-hard`
- [x] `prop:minimal_m_decision_epsilon_reduct_problem` (L2164) -> `prop-m-epsilon-reduct-np-hard`
- [x] `prop:minimal_relative_r_decision_epsilon_reduct_problem` (L2180) -> `prop-relative-r-epsilon-reduct-np-hard`
- [x] `prop:minimal_r_decision_epsilon_reduct_problem` (L2197) -> `prop-r-epsilon-reduct-np-hard`

### Remaining Chapters (intentionally skipped)

- [x] Chapter 5: Case Study -- experimental results, no new definitions or propositions.
- [x] Chapter 6: Feature Importance and Ranks -- experimental rankings, no new formal content.
- [x] Chapter 7: Conclusions -- summary, no new definitions or propositions.
- [x] Appendices -- profiling + notebook examples, no new theory.

---

## Proof Gaps & Open Issues

No unresolved proof gaps. The previously flagged gap in
`prop-temporal-bireduct-computation` (backward non-extendability) was investigated and found to be
valid -- see `prop-temporal-bireduct-computation.md` for the tightened argument.

---

## Session Reflections (2026-06-06)

### Caveats for Future Sessions

- **`prop:gamma_decision_bireduct_ordering`** and **`prop:gamma_decision_bireduct_sampling`**:
  the gamma versions of ordering/sampling propositions have very brief thesis proofs (referencing
  the standard case). While the analogies are valid, the KB files would benefit from more
  explicit step-by-step expansion.
- **NP-hardness propositions**: all 8 extracted as standalone files forming a complete reduction
  chain: MDS -> relative γ -> γ -> α-MDS -> relative M -> M -> relative R -> R.
- **Examples**: all 9 extracted from `tmp/phd/include/` as standalone files.
