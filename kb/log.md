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
