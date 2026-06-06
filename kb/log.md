# Knowledge Base Log

An append-only chronological journal of all operations performed on the knowledge base. Each entry
has a date and type prefix for parseability.

---

## [2026-06-04] init | Knowledge base initialized

Schema layer (AGENTS.md) created. Directory structure established. Template, index, log, and
notation files initialized. Executor prompt written. Knowledge base is ready for first ingest.

## [2026-06-04] notation | Notation table populated from thesis.tex

Source: tmp/phd/thesis.tex. Extracted all custom LaTeX commands from preamble (lines 150-359) and
mapped them to KB conventions. Verified against knowledgebase_old/notation_and_symbols.md for
backward compatibility. Minor divergences: gamma special value changed from $*$ to $\circledast$;
script fonts normalized from \mathpzc to \mathcal or plain letters. Symbols registered: 47 total.
Source-to-KB translation table added in ### Source-to-KB Translation Notes section.

## [2026-06-04] ingest | thesis.tex -- decision table

Created: concept-decision-table.
Updated: notation.md, index.md (first concept entry).
Source: tmp/phd/thesis.tex, def:decision_table.
Status: complete.

## [2026-06-04] ingest | thesis.tex -- Preliminaries: classification + evaluation

Created: concept-classification (def:classification_task + def:classification_model),
concept-classification-evaluation (def:evaluation_metric through def:balanced_accuracy,
9 definitions grouped into one file).
Source: tmp/phd/thesis.tex, Sections "Classification Task" and "Evaluation Metrics for
Classification Models" (ch:preliminaries).
Notes: Evaluation metrics grouped because individual definitions are short (< 15 lines each)
and tightly coupled. Old knowledgebase_old had only a 14-line evaluation.md stub.
Status: complete.

## [2026-06-04] ingest | thesis.tex -- Ch2: Foundations of Decision Reducts + Ch3: Decision Bireducts + Ch4: Algorithms

Created (Ch2 -- RST core):
  concept-indiscernibility (def:indiscernibility_relation + def:discernibility_relation),
  concept-approximations (def:approximations),
  concept-consistency (def:consistent_decision_table),
  concept-formulae (def:formulae),
  concept-decision-rule (def:decision_rule),
  concept-decision-reduct (def:decision_reduct + def:discernibility_based_decision_reduct
    + def:minimal_decision_reduct + prop:decision_reduct_boolean_formula, in concepts/reducts.md),
  concept-discernibility-measure (def:disc_measure),
  concept-positive-region (def:positive_region, including gamma function),
  concept-gamma-decision-reduct (def:gamma_decision_reduct + prop:gamma_decision_reduct_*),
  concept-approximate-decision-reduct (def:relative_approximate_decision_reduct
    + def:approximate_decision_reduct),
  concept-majority-function (def:m_measure),
  concept-relative-gain-function (def:r_measure),
  concept-np-hardness-foundations (def:graph, def:dominating_set, def:alpha_dominating_set).

Created (Ch3 -- Bireducts):
  concept-decision-bireduct (def:decision_bireduct + monotonicity + relationship to reducts
    + rules interpretation + Boolean formula + diagonal table),
  concept-gamma-decision-bireduct (def:gamma_decision_bireduct + POS equivalence),
  concept-epsilon-decision-bireduct (def:epsilon_decision_bireduct
    + def:gamma_epsilon_decision_bireduct + NP-hardness),
  concept-bireduct-ensemble (def:bireduct_ensemble_correct + def:bireduct_ensemble_simpler
    + def:simplest_correct_decision_bireduct_ensemble_problem).

Created (Ch4 -- Algorithms):
  concept-temporal-bireduct (def:temporal_bireduct).

Updated: index.md (all 21 concept entries).
Total files created this batch: 17 (21 total in kb/concepts/).
Compared with knowledgebase_old: old KB had 8 definition files + 4 proposition files covering
similar ground. New KB provides significantly more detail (formulae, rules, measures, bireduct
variants, ensembles, NP foundations, temporal bireducts). All alternative formulations from
thesis captured inline.

Status: complete.

## [2026-06-04] ingest | thesis.tex -- examples (golf dataset tables)

Created:
  ex-golf-reduct-rules (decision rules from 2 reducts, tab:decision_reducts_rules),
  ex-golf-bireduct-rules (rules from sample bireducts and gamma-bireducts,
    tab:decision_bireducts_rules + tab:gamma_decision_bireducts_rules),
  ex-golf-all-bireducts (complete list of all bireducts and gamma-bireducts,
    tab:decision_bireducts_gamma_decision_bireducts_all).
Updated: index.md (Examples section populated).

AGENTS.md guidance on examples: small/single-table examples go inline in concept ## Example
sections; complex/multi-table examples go in kb/examples/ as standalone files with
type: example. The golf dataset itself is already inline in concept-decision-table.
Status: complete.

## [2026-06-04] ingest | thesis.tex -- examples (gamma-reduct rules, all-bireducts fix)

Created: ex-golf-gamma-reduct-rules (gamma-modified tables + rules from 2 gamma-reducts).
Fixed: ex-golf-all-bireducts (corrected 8 wrong gamma-bireduct sets and 7 wrong counts;
  original was summarised with invented values instead of faithfully reproducing the source).
Updated: index.md.

## [2026-06-04] ingest | thesis.tex -- first standalone proposition + memory file

Created: prop-indiscernibility-equivalence-relation (prop:indiscernibility_eqivalence_relation,
  with full proof structured as reflexivity/symmetry/transitivity + consequences),
  kb/pending.md (tracks remaining unprocessed examples, propositions, and chapters).
Updated: concept-indiscernibility (inline proof replaced with link to standalone proposition),
  index.md (Propositions section populated).

## [2026-06-04] ingest | thesis.tex -- propositions 2 and 3

Created:
  prop-decision-reduct-boolean-formula
    (prop:decision_reduct_boolean_formula; proof ref: Skowron & Rauszer 1992),
  prop-gamma-decision-reduct-consistent-table
    (prop:gamma_decision_reduct_inconsistent_decision_table; proof ref: Slezak 2018).
Both proofs follow the thesis style -- referencing external sources with explanatory
commentary on the construction's intuition.
Updated: index.md, pending.md.
Fixed: prop-gamma-decision-reduct-consistent-table citation title corrected against thesisbib.bib.

## [2026-06-04] meta | Project conventions consolidated in pending.md

Reverted executor_prompt.md edit -- project-specific conventions are per-ingest, not universal
executor rules. All conventions now live in pending.md under "Key user preferences" and
"Resume Instructions". pending.md is the single entry point for resuming this ingest.

## [2026-06-05] ingest | thesis.tex -- proposition: gamma-decision-reduct-boolean-formula

Created: prop-gamma-decision-reduct-boolean-formula
  (prop:gamma_decision_reduct_boolean_formula; Boolean formula $\tau^\gamma$ for $\gamma$-decision
  reducts; proof ref: Skowron & Rauszer 1992).
Updated: concept-gamma-decision-reduct (inline proposition replaced with link to standalone file),
  index.md (Propositions section: 4 entries).
Status: complete.

## [2026-06-05] ingest | thesis.tex -- proposition: monotony-properties

Created: prop-monotony-properties
  (prop:monotony_properties; two monotonicity properties of inexact functional dependency with
  detailed step-by-step proof preserving thesis style).
Updated: concept-decision-bireduct (inline proposition replaced with link to standalone file),
  index.md (Propositions section: 5 entries).
Status: complete.

## [2026-06-06] ingest | thesis.tex -- propositions: bireduct-reduct bridge and rules

Created:
  prop-decision-reduct-iff-bireduct
    (prop:decision_reduct_iff_bireduct; reduct iff $(U, B)$ is a bireduct; short proof from
    definitions),
  prop-decision-bireduct-iff-reduct
    (prop:decision_bireduct_iff_reduct; bireduct characterized via subtable consistency and
    reduct; detailed two-direction proof),
  prop-bireduct-objects-and-rules
    (prop:bireduct_objects_and_rules; three structural properties linking bireduct objects to
    decision rule supports; three-part proof by contradiction).
Updated: concept-decision-bireduct (inline propositions replaced with summaries + links,
  see_also expanded), index.md (Propositions section: 8 entries).
Status: complete.

## [2026-06-06] ingest | thesis.tex -- propositions: gamma-bireduct properties

Created:
  prop-gamma-monotony-properties
    (prop:gamma_monotony_properties; gamma functional dependency monotonicity; proof expanded from
    thesis "analogous to prop:monotony_properties" reference),
  prop-gamma-decision-bireduct-to-reduct
    (prop:gamma_decision_bireduct_to_reduct; reduct iff $(U, B)$ is a $\gamma$-bireduct; two-direction
    proof from definitions),
  prop-gamma-decision-bireduct-pos
    (prop:gamma_decision_bireduct_pos; $\gamma$-bireduct iff $X = POS(B)$ + irreducibility;
    long two-direction proof with consequences: uniqueness, reduction to standard reducts via
    $\mathbb{A}_A^\gamma$, rule interpretation).
Updated: concept-gamma-decision-bireduct (inline propositions replaced with summaries + links,
  monotonicity section added, see_also expanded), index.md (Propositions section: 11 entries).
Status: complete.

## [2026-06-06] ingest | thesis.tex -- propositions: Boolean formulae and diagonal transformation

Created:
  prop-decision-bireduct-boolean-formula
    (prop:decision_bireduct_boolean_formula; bireducts as prime implicants of $\tau_{bi}$;
    long two-step proof: Step 1 functional dependency $\iff$ implicant, Step 2 bireduct $\iff$
    prime implicant, with detailed case analysis),
  prop-decision-table-diagonal
    (prop:decision_table_diagonal; bireducts as reducts on diagonal-augmented table;
    two-direction proof showing discernibility and irreducibility preservation),
  prop-gamma-decision-bireduct-boolean-formula
    (prop:gamma_decision_bireduct_boolean_formula; $\gamma$-bireducts as prime implicants of
    $\tau_{bi}^{\gamma}$; proof "fully analogous" to prop:decision_bireduct_boolean_formula,
    verified correct; includes formula transformation revealing positive region connection).
Updated: concept-decision-bireduct (inline Boolean formula and diagonal proposition replaced
  with summaries + links, see_also expanded),
  concept-gamma-decision-bireduct (inline Boolean formula proposition replaced with summary +
  link, see_also expanded),
  index.md (Propositions section: 14 entries).
Status: complete.

## [2026-06-06] ingest | thesis.tex -- propositions: epsilon-bireducts and ensembles complexity

Created:
  prop-m-reduct-epsilon-bireduct-correspondence
    (prop:smallest_m_decision_epsilon_reduct_decision_epsilon_bireduct; bidirectional correspondence
    between smallest M-reducts and ε-bireducts; two-direction proof with construction details),
  prop-minimal-epsilon-bireduct-np-hard
    (prop:minimal_decision_epsilon_bireduct_problem; NP-hardness of MDεBP for any ε ∈ [0,1);
    reduction from minimal M-reduct problem using correspondence proposition),
  prop-ensemble-np-hard
    (prop:ensemble_np; SCDBEP is NP-hard; polynomial reduction from Minimal Dominating Set with
    graph-to-table encoding, construction of simpler ensemble from dominating set, dummy classifiers).
Updated: concept-epsilon-decision-bireduct (inline propositions replaced with summaries + links,
  see_also expanded),
  concept-bireduct-ensemble (inline SCDBEP proposition replaced with summary + link, see_also
  expanded),
  index.md (Propositions section: 17 entries, Source Summaries section: 1 entry).
Status: complete.

## [2026-06-06] ingest | thesis.tex -- Algorithms: ordering and sampling propositions (Round 5)

Created:
  prop-decision-bireduct-ordering
    (prop:decision_bireduct_ordering; ordering algorithm always outputs a decision bireduct;
    two-part proof: irreducibility/maximality by contradiction + achievability via 4-segment
    permutation),
  prop-gamma-decision-bireduct-ordering
    (prop:gamma_decision_bireduct_ordering; gamma analog; proof analogous to standard ordering
    case using gamma-dependency and gamma-monotony),
  prop-decision-bireduct-sampling
    (prop:decision_bireduct_sampling; sampling algorithm correctness; consistent sub-table
    construction + reduct property transfer; reverse direction by setting A^◇ = B),
  prop-gamma-decision-bireduct-sampling
    (prop:gamma_decision_bireduct_sampling; gamma analog; simpler proof using gamma-modified
    decision table and positive region).
Updated: index.md (Propositions section: 21 entries),
  pending.md (checked off 4 propositions, updated current state and next steps).
Status: complete.

## [2026-06-06] ingest | thesis.tex -- temporal bireduct computation proposition

Created:
  prop-temporal-bireduct-computation
    (prop:temporal_bireduct_computation; streaming buffer algorithm produces temporal bireducts;
    two-part proof: saved pairs satisfy forward non-extendability + reachability by
    setting A' = B and tracing buffer evolution).
Flagged: backward non-extendability proof has a gap (conflicting partner may be removed in
  subsequent reset before save time). Documented in pending.md ## Proof Gaps.
Updated: index.md (Propositions section: 22 entries),
  pending.md (checked off 1 proposition, added ## Proof Gaps section, added ## Session
  Reflections).
Status: complete.

## [2026-06-06] ingest | Erickson -- NP-hardness methodology

Created: src-erickson-np-hardness-methodology (three-step reduction template, certificate
  perspective, asymmetry of (⇐), optimization vs. decision, common pitfalls; illustrated with
  CircuitSat→3Sat and Minimal Dominating Set→SCDBEP examples).
Updated: index.md.
Status: complete.
