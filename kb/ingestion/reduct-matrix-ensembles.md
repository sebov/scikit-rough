# Ingestion: Complexity of Searching for the Simplest Reduct Matrix Ensembles

Source: `tmp/pub/main.tex` (1573 lines, LaTeX, LNCS format).
Authors: Dominik Ślęzak, Sebastian Stawicki (University of Warsaw).
Source-status: **WORK-IN-PROGRESS** -- paper is incomplete (missing introduction, optimization
problem proof, conclusions; contains TODOs and one proof in Polish).
Source-summary: not yet created (will be `src-reduct-matrix-ensembles` in `kb/sources/`).

**Status: IN PROGRESS.** Iterative ingestion -- extracting formal content (definitions,
propositions, proofs) incrementally with user verification.

---

## Context & Instructions

### What This Source Is

A conference paper extending the ensemble NP-hardness result from the PhD thesis
(`prop-ensemble-np-hard`). The key difference: the thesis used a simplicity order based only on
sorted attribute cardinalities (ignoring objects), while this paper uses **description length** --
a measure that accounts for both attributes and objects in each bireduct:

$$BireductDescLen(X, B) = |X/B| \cdot (|B| + 1)$$

$$EnsembleDescLen(\mathcal{B}) = \sum_{(X_i, B_i) \in \mathcal{B}} BireductDescLen(X_i, B_i)$$

This is a more comprehensive notion of ensemble simplicity.

### Main Results (Target)

1. **Decision problem (CDBEkP)**: NP-complete -- reduction from Set Cover (Section 4, mostly
   complete in source).
2. **Optimization problem (SCDBEP)**: NP-hard -- (Section 5, **completely empty** in source).

### Relationship to Existing KB

- Continuation of `prop-ensemble-np-hard` (thesis result, different measure).
- Uses the same bireduct/ensemble foundations already in `kb/concepts/`.
- The reduction base changes: thesis used Dominating Set, this paper uses **Set Cover**.
- Several foundational definitions are repeated from the thesis (indiscernibility, consistency,
  decision bireduct) -- these do NOT need new KB files, just notation alignment.

### How to Resume

1. Review `kb/AGENTS.md` for schema rules.
2. Check `kb/notation.md` for symbol conventions.
3. Use `kb/template.md` for file structure.
4. Work through the extraction checklist below **incrementally** -- user wants to verify each
   proposition before moving on.
5. General guidelines: `kb/ingestion_guidelines.md`.
6. **Next up**: `prop:number_of_equiv_classes_xb_equals_ub` (line 140 in checklist) -- for bireduct
   $(X, B)$: $|X/B| = |U/B|$. Proof is complete in source.

### Key User Preferences

- **Iterative**: do NOT extract everything at once. Work proposition-by-positiion, verify with
  user.
- **Formal content priority**: definitions and propositions are the main focus. Narrative,
  related work, introduction are not a priority (source is WIP).
- **Verification**: user wants to verify proofs -- be ready to discuss and validate each step.
- **Proof graph**: eventually sketch the dependency graph between propositions showing how the
  final NP-completeness/hardness result is built from intermediate lemmas.
- **Missing results**: at least one intermediate result is known to be missing. Do not worry
  about it now -- will be addressed later.
- **Language**: user communicates in Polish; all KB output in English.

---

## Notation Mapping (Source -> KB)

Source uses the same custom LaTeX commands as the thesis. Mapping is identical to what is already
documented in `kb/notation.md` under "Source-to-KB Translation Notes". Key mappings:

| Source Command      | KB Symbol                       | Notes                   |
| :------------------ | :------------------------------ | :---------------------- |
| `\birobjects`       | $X$                             | Bireduct object subset  |
| `\ensembleb`        | $\mathcal{B}$                   | Ensemble                |
| `\ensemblec`        | $\mathcal{C}$                   | Alternate ensemble      |
| `\funcdep{B}{X}{d}` | $B \Rrightarrow_X d$            | Functional dependency   |
| `\coverage{A}{B}`   | $cov_{\mathbb{A}, \mathcal{B}}$ | Coverage count function |

### New Symbols Needed (not yet in `kb/notation.md`)

| Symbol                         | Name                        | Description                                                    |
| :----------------------------- | :-------------------------- | :------------------------------------------------------------- |
| $BireductDescLen(X, B)$        | Bireduct description length | Total descriptors in rules induced by bireduct $(X, B)$        |
| $EnsembleDescLen(\mathcal{B})$ | Ensemble description length | Sum of bireduct description lengths                            |
| $W$                            | Set cover universe          | Finite universe of objects in set cover problem                |
| $\mathcal{S}$                  | Set cover family            | Family of subsets of $W$                                       |
| $\mathcal{C}ov$                | Set cover solution          | Subfamily of $\mathcal{S}$ whose union equals $W$              |
| $\omega$                       | Set cover element           | Generic element of $W$                                         |
| $\mathbb{A}_{\mathcal{S}}$     | Transformed decision table  | Decision table constructed from set cover instance             |
| $u_*$                          | Special object              | Additional object in transformed table (decision = 1)          |
| $u_\omega$                     | Object from universe        | Object in transformed table corresponding to $\omega \in W$    |
| $\mathcal{K}$                  | K-multiset                  | Bireducts $(\{u_*\}, \emptyset)$ with multiplicity $K$         |
| $\mathcal{L}$                  | L-multiset                  | Bireducts $(U_{\mathcal{S}}, \emptyset)$ with multiplicity $L$ |
| $\mathcal{M}$                  | M-multiset                  | Bireducts with single-attribute subsets                        |
| $\mathcal{R}epl$               | Replacement multiset        | Collection replacing a multi-attribute bireduct                |

---

## Extraction Checklist

### Definitions

Source repeats some definitions already in KB (indiscernibility, consistency, decision bireduct).
These do NOT need new files -- just confirm notation alignment.

- [ ] `def:reduct_matrix` -- **NEW concept**: reduct matrix $\mathbb{M}_B$ as projection of
  $\mathbb{A}$ onto decision rules induced by $B$. Not in KB yet.
- [x] `def:indiscernibility_relation` -- already in `concept-indiscernibility`.
- [x] `def:consistent_decision_table` -- already in `concept-consistency`.
- [x] `def:decision_bireduct` -- already in `concept-decision-bireduct`.
- [x] `def:bireduct_desc_len` -- **DONE**: added to `concept-bireduct-ensemble` as new section "Description Length".
- [x] `def:ensemble` -- **DONE**: added to `concept-bireduct-ensemble`.
- [x] `def:ensemble_coverage` -- **DONE**: added to `concept-bireduct-ensemble`.
- [x] `def:bireduct_ensemble_correct` -- **DONE**: updated to use $cov_{\mathbb{A},\mathcal{B}}$.
- [x] `def:bireduct_ensemble_desc_len` -- **DONE**: added to `concept-bireduct-ensemble`.
- [ ] `def:correct_ensemble_of_size_k_problem` -- **NEW concept**: CDBEkP (decision problem).
- [ ] `def:bireduct_ensemble_simpler` -- **UPDATE**: current `concept-bireduct-ensemble` uses
  the old $\prec$ based on sorted cardinalities. This paper defines $\prec$ via description
  length. Need to handle as alternative formulation or update.
- [ ] `def:simplest_correct_ensemble_problem` -- **UPDATE**: SCDBEP with new measure. Same
  consideration as above.
- [ ] `def:simple_bireducts_ensemble` -- **NEW concept**: ensemble with only 0/1-attribute
  bireducts (used in the proof).

### Propositions

Already in KB (skip or cross-reference):
- [x] `prop:bireduct_objects_and_rules` -> already `prop-bireduct-objects-and-rules`.

New propositions to extract (in order of appearance / dependency):

#### Foundational Lemmas (Section 2)

- [ ] `prop:bireduct_attrs_subset_form_bireduct` -- Every attribute subset of a bireduct's
  attributes can form a bireduct with some object subset. **NOTE**: proof in source is in Polish
  and incomplete/sketchy. Needs verification and possibly completion.
- [x] `prop:number_of_equiv_classes_b_leq_bprim` -- Monotonicity: $B \subseteq B'$ implies
  $|U/B| \leq |U/B'|$. Proof is complete. **DONE**: `prop-equiv-classes-monotonicity`.
- [ ] `prop:number_of_equiv_classes_xb_equals_ub` -- For bireduct $(X, B)$: $|X/B| = |U/B|$.
  Proof is complete.
- [ ] `prop:bireduct_desc_len_equals_xb_bplus1` -- $BireductDescLen(X, B) = |X/B| \cdot (|B|+1)$.
  Proof is complete.
- [ ] `prop:bireduct_equiv_classes_geq_bplus1` -- $|X/B| \geq |B|+1$ for any bireduct. Proof by
  induction, complete.
- [ ] `prop:bireduct_desc_len_geq_bplus1_squared` -- $BireductDescLen(X, B) \geq (|B|+1)^2$.
  Direct corollary, complete.

#### Ensemble Properties (Section 3)

- [ ] `prop:correct_ensemble_iff_dectab_consistent` -- Correct ensemble exists iff table is
  consistent. Proof is complete.

#### Decision Problem Proof (Section 4)

- [ ] `prop:correct_ensemble_of_size_k_problem` -- CDBEkP is NP-complete. **NOTE**: source has
  TODO ("NP-complete or NP-hard -- depends on whether we can show NP membership"). Proof
  references `sec:decision_problem_proof`.
- [ ] Proposition: transformed table $\mathbb{A}_{\mathcal{S}}$ is consistent. (Unnumbered in
  source, proof is complete but short.)
- [ ] **Note** `note:solution_bireduct_properties` -- Structural properties of bireducts in the
  transformed table. Important intermediate result.
- [ ] **Note** `note:bireduct_replacement_correctness` -- Replacement preserves correctness.
  Long but complete.
- [ ] `prop:bireducts_with_0and1_attrs_desc_size` -- Description lengths for 0-attr (1) and
  1-attr (4) bireducts in transformed table. Complete.
- [ ] **Note** `note:bireduct_replacement_simpler` -- Replacement does not increase description
  length. Uses quadratic argument.
- [ ] `prop:ensemble_with_0and1_attrs_decomposition` -- K/L/M decomposition of simple-bireducts
  ensemble. Complete.
- [ ] `prop:correct_ensemble_klm_then_m_nonempty` -- Correct simple-bireducts ensemble must
  have at least one 1-attribute bireduct. Complete.
- [ ] Proposition: from any correct K/L/M ensemble, extract a correct L'/M ensemble (remove K).
  **NOTE**: unlabeled in source (no `\label{}`). Complete proof with two cases.

#### Missing / Incomplete

- [ ] Section 4 is incomplete after the K/L/M decomposition -- there are TODOs:
  - "pokazać, że mając ensemble of bireducts to mamy set cover" (ensemble -> set cover)
  - "pokazać, że mając set cover, możemy zbudować ensemble of bireducts" (set cover -> ensemble)
  - These are the two directions of the NP-completeness reduction.
- [ ] Section 5 (optimization problem, SCDBEP NP-hard) is **completely empty**.
- [ ] NP membership for CDBEkP has a TODO ("Uzasadnić dlaczego to jest NP").

---

## Proof Dependency Graph (Draft)

This sketches how the final results are built from intermediate propositions. To be refined as
ingestion progresses.

### Decision Problem (CDBEkP is NP-complete)

```
FOUNDATIONAL LEMMAS:
  prop:bireduct_attrs_subset_form_bireduct
  prop:number_of_equiv_classes_b_leq_bprim
  prop:number_of_equiv_classes_xb_equals_ub
    |
    v
  prop:bireduct_desc_len_equals_xb_bplus1
  prop:bireduct_equiv_classes_geq_bplus1
    |
    v
  prop:bireduct_desc_len_geq_bplus1_squared

ENSEMBLE PROPERTIES:
  prop:correct_ensemble_iff_dectab_consistent

DECISION PROOF (Set Cover -> CDBEkP):
  [Set Cover instance] --reduction--> [transformed table A_S]
    |
    v
  [A_S is consistent] + prop:correct_ensemble_iff_dectab_consistent
    |
    v
  note:solution_bireduct_properties (structure of bireducts in A_S)
    |
    v
  note:bireduct_replacement_correctness (splitting multi-attr bireducts preserves correctness)
  note:bireduct_replacement_simpler (splitting does not increase desc length)
    |                                          |
    v                                          v
  prop:bireducts_with_0and1_attrs_desc_size    prop:bireduct_desc_len_geq_bplus1_squared
    |
    v
  [simple-bireducts ensemble]
    |
    v
  prop:ensemble_with_0and1_attrs_decomposition (K/L/M structure)
    |
    v
  prop:correct_ensemble_klm_then_m_nonempty
    |
    v
  [remove K -> L'/M ensemble] (unlabeled prop)
    |
    v
  [TODO: L'/M ensemble -> set cover solution]
  [TODO: set cover solution -> L'/M ensemble]
    |
    v
  [CDBEkP is NP-complete]
```

### Optimization Problem (SCDBEP is NP-hard)

```
  [EMPTY IN SOURCE -- Section 5]
```

---

## Open Issues

1. **Missing proof directions**: the two directions of the Set Cover <-> CDBEkP reduction are
   TODO in the source. These are critical for the NP-completeness result.
2. **Missing Section 5**: the entire optimization problem proof is absent.
3. **Polish proof**: `prop:bireduct_attrs_subset_form_bireduct` has a proof sketch in Polish
   that is incomplete. Needs to be verified and potentially completed from scratch.
4. **NP membership**: the source has a TODO for justifying why CDBEkP is in NP.
5. **Unlabeled proposition**: the proposition about removing K from the ensemble (after
   `prop:correct_ensemble_klm_then_m_nonempty`) has no `\label{}` in the source.
6. **Simplicity order conflict**: the existing `concept-bireduct-ensemble` defines $\prec$ using
   sorted attribute cardinalities. This paper uses description length. Need to decide: update
   existing concept, add as alternative formulation, or create separate concept.
7. **At least one intermediate result is known to be missing** (user is aware).

---

## Session Log

### 2026-06-10 -- Foundational Lemmas (Part 1)

- Extracted `prop-equiv-classes-monotonicity` from source (lines 357-402 of `tmp/pub/main.tex`):
  monotonicity of equivalence class count under attribute subset inclusion. Proof verified as
  complete and correct.
- Extended `concept-indiscernibility.md` with new section "Indiscernibility on a Subset of Objects":
  added $IND_V(B)$ and $[u]_B^V$ notation. This generalization is needed for bireduct context where
  $IND_X(B)$ is used.
- Updated `notation.md` with new symbols: $IND_V(B)$ and $[u]_B^V$.
- User preference confirmed: iterative extraction, verify proofs, discuss notation before
  extracting.

**Status**: 1 of 6 foundational lemmas extracted. Next: `prop:number_of_equiv_classes_xb_equals_ub`
(for bireduct $(X, B)$: $|X/B| = |U/B|$).

### 2026-06-08 -- Initial Setup

- Created ingestion tracking file.
- Read full source (1573 lines of LaTeX).
- Identified all definitions, propositions, notes, and their dependencies.
- Mapped source notation to KB conventions.
- Drafted proof dependency graph.
- No extraction to KB yet -- waiting for user to begin iterative work.
