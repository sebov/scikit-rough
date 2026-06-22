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
6. **Next up**: `prop:bireducts_with_0and1_attrs_desc_size` (description lengths: 0-attr = 1,
   1-attr = 4) and `def:simple_bireducts_ensemble`.

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
- [x] `def:bireduct_ensemble_desc_len` -- **DONE**: already in `concept-bireduct-ensemble` sections "Description Length" and "Ensemble Description Length".
- [x] `def:correct_ensemble_of_size_k_problem` -- **DONE**: added `concept-cdbe-kp`.
- [x] `def:bireduct_ensemble_simpler` -- **DONE**: added to `concept-bireduct-ensemble` as "Simpler Ensemble" ($\prec$ via description length). Distinct from attribute-based $\prec_A$.
- [x] `def:simplest_correct_ensemble_problem` -- **DONE**: added to `concept-bireduct-ensemble` as "Simplest Correct Ensemble Problem (SCDBEP)". Old thesis version renamed to ASCDBEP ($\prec_A$).
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
- [x] `prop:number_of_equiv_classes_xb_equals_ub` -- For bireduct $(X, B)$: $|X/B| = |U/B|$.
  Proof is complete. **DONE**: `prop-equiv-classes-bireduct`.
- [x] `prop:bireduct_desc_len_equals_xb_bplus1` -- $BireductDescLen(X, B) = |X/B| \cdot (|B|+1)$.
  Proof is complete. **DONE**: `prop-bireduct-desc-len-formula`.
- [x] `prop:bireduct_equiv_classes_geq_bplus1` -- $|X/B| \geq |B|+1$ for any bireduct. Proof by
  induction, complete. **DONE**: `prop-bireduct-equiv-classes-geq-bplus1`.
- [x] `prop:bireduct_desc_len_geq_bplus1_squared` -- $BireductDescLen(X, B) \geq (|B|+1)^2$.
  Direct corollary, complete. **DONE**: `prop-bireduct-desc-len-geq-bplus1-squared`.

#### Ensemble Properties (Section 3)

- [x] `prop:correct_ensemble_iff_dectab_consistent` -- Correct ensemble exists iff table is
  consistent. Proof is complete.

#### Decision Problem Proof (Section 4)

- [x] `prop:correct_ensemble_of_size_k_problem` -- **DONE**: created `prop-cdbe-kp-np-complete.md`. Definition moved to `concept-bireduct-ensemble` (replacing standalone `concept-cdbe-kp`). Auxiliary lemmas to go in `cdbe-kp-np-complete/` subdirectory.
- [x] `prop-set-cover-problem` -- **DONE**: Set Cover decision problem definition (Karp, 1972). In `cdbe-kp-np-complete/`.
- [x] `prop-set-cover-construction` -- **DONE**: polynomial-time construction of $\mathbb{A}_{\mathcal{S}}$ from $(W, \mathcal{S})$ with example. In `cdbe-kp-np-complete/`.
- [x] Proposition: transformed table $\mathbb{A}_{\mathcal{S}}$ is consistent. (Unnumbered in
  source, proof is complete but short.) **DONE**: `prop-transformed-table-consistent`.
- [x] **Note** `note:solution_bireduct_properties` -- Structural properties of bireducts in the
  transformed table. Important intermediate result. **DONE**: promoted to `prop-solution-bireduct-properties`.
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

### 2026-06-21 — Issues Found in Proof Review

8. **Proof gap: "at least two indiscernibility classes"** -- `prop-solution-bireduct-properties`
   (KB and TeX). The proof claims $IND(B)$ has $\geq 2$ classes when $|B| \geq 1$, but does not
   address the degenerate case where all $S_i \in B$ are empty ($S_i = \emptyset$). In that case
   all objects have value $0$ on every attribute in $B$, yielding one class. Such $B$ would not be
   a bireduct (reducible to $\emptyset$), but the proof must state this justification. **Affects
   both TeX `prop:solution_bireduct_properties` and KB `prop-solution-bireduct-properties`.**

9. **~~Forward reference to non-existent lemmas~~** -- **FIXED**. Removed "description length
   formulas" from `prop-set-cover-construction.md` Remarks; only existing lemmas (consistency,
   bireduct characterization) are referenced.

10. **~~Consistency proof: informal link to definition~~** -- **FIXED**. `prop-transformed-table-consistent.md`
    now explicitly references the equivalent pairwise formulation from `concept-consistency.md`.

11. **Missing Set Cover link in formula** -- `prop-solution-bireduct-properties.md` states
    $X = \{u_*\} \cup \{u_\omega \mid \exists a_{S_i} \in B : a_{S_i}(u_\omega) = 1\}$ without
    noting the equivalent characterization: $u_\omega \in X \iff \omega \in \bigcup_{a_{S_i} \in B} S_i$.
    The Set Cover interpretation should be explicit since it is the crux of the reduction.

---

## Session Log

### 2026-06-21 (session 2) -- Solution Bireduct Properties & Proof Review

- Extracted `prop-solution-bireduct-properties` from source (promoted from `\note` to `\proposition`):
  structural characterization of bireducts in $\mathbb{A}_{\mathcal{S}}$. For $|B| \geq 1$,
  $X = \{u_*\} \cup \{u_\omega \mid \exists a_{S_i} \in B : a_{S_i}(u_\omega) = 1\}$;
  for $B = \emptyset$, $X = \{u_*\}$ or $X = U_{\mathcal{S}}$. Proof by indiscernibility class
  analysis. Referenced 5 times in later lemmas.
- Extracted `prop-transformed-table-consistent` from source (lines 867-882):
  $\mathbb{A}_{\mathcal{S}}$ is always consistent because $u_*$ (decision $1$, all attrs $0$) is
  distinguishable from every $u_\omega \in U_{\mathcal{S}}$ (decision $0$, at least one attr $1$).
- Updated `prop-cdbe-kp-np-complete.md` with `requires` links to new lemmas.
- Updated `index.md` with entries for new propositions.
- Proof review of all 4 auxiliary files -- found issues (see Open Issues section below).

**Status**: Four foundational lemmas of the reduction chain established. Next:
  1. Fix Issue 1 (proof gap in `solution-bireduct-properties`) -- TeX first, then KB
  2. `prop:bireducts_with_0and1_attrs_desc_size` (description lengths: 0-attr = 1, 1-attr = 4)
  3. `def:simple_bireducts_ensemble`

### 2026-06-21 (session 1) -- Set Cover Reduction Base (Section 4)

- Extracted `prop-set-cover-problem` from source (lines 789-800):
  formal definition of the Set Cover decision problem -- universe $W$, family $\mathcal{S}$,
  subfamily $\mathcal{C}$ of size $\leq l$ covering $W$ (Karp, 1972).
- Extracted `prop-set-cover-construction` from source (lines 802-865):
  polynomial-time construction of decision table $\mathbb{A}_{\mathcal{S}}$ from $(W, \mathcal{S})$;
  includes objects $U_{\mathcal{S}} \cup \{u_*\}$, binary attributes $A_{\mathcal{S}} = \{a_{S_i}\}$,
  decision $d_{\mathcal{S}}$, proof of polynomial time, and worked example.
- Created local notation file `cdbe-kp-np-complete/notation.md` -- Set Cover symbols are local to
  this proof, not in global `kb/notation.md`.
- Extracted `prop-solution-bireduct-properties` from source (promoted from `\note` to `\proposition`):
  structural characterization of bireducts in $\mathbb{A}_{\mathcal{S}}$. For $|B| \geq 1$,
  $X = \{u_*\} \cup \{u_\omega \mid \exists a_{S_i} \in B : a_{S_i}(u_\omega) = 1\}$;
  for $B = \emptyset$, $X = \{u_*\}$ or $X = U_{\mathcal{S}}$. Proof by indiscernibility class
  analysis. Referenced 5 times in later lemmas.
- Extracted `prop-transformed-table-consistent` from source (lines 867-882):
  $\mathbb{A}_{\mathcal{S}}$ is always consistent because $u_*$ (decision $1$, all attrs $0$) is
  distinguishable from every $u_\omega \in U_{\mathcal{S}}$ (decision $0$, at least one attr $1$).
- Updated `prop-cdbe-kp-np-complete.md` with `requires` links to new lemmas.
- Updated `index.md` with entries for new propositions.

**Status**: Four foundational lemmas of the reduction chain established. Next:
  1. `note:solution_bireduct_properties` (structural properties of bireducts in A_S, source lines 884+)
  2. `def:simple_bireducts_ensemble` (definition of simple-bireducts ensemble)

### 2026-06-17 -- Staging Unverified Propositions

- Moved 3 propositions to `kb/staging/` (pending verification):
  - `prop-bireduct-attrs-subset-form-bireduct`: proof incomplete in source (Polish, cuts off mid-argument), general case unverified
  - `prop-bireduct-equiv-classes-geq-bplus1`: induction proof depends on unverified `prop-bireduct-attrs-subset-form-bireduct`
  - `prop-bireduct-desc-len-geq-bplus1-squared`: depends on unverified `prop-bireduct-equiv-classes-geq-bplus1`
- Updated `index.md`: removed from Propositions, added to new Staging section.
- Note: `prop-bireduct-desc-len-formula` has only `see_also` (not `requires`) to the unverified chain, so it stays in the verified KB.

### 2026-06-11 -- Foundational Lemmas (Part 5)

- Extracted `prop-bireduct-desc-len-geq-bplus1-squared` from source (lines 544-559 of `tmp/pub/main.tex`):
  $BireductDescLen(X, B) \geq (|B| + 1)^2$. Direct corollary from `prop-bireduct-desc-len-formula`
  and `prop-bireduct-equiv-classes-geq-bplus1`. Proof is trivial (one equation).
- Updated `index.md` with new entry.

**Status**: All 6 foundational lemmas from Section 2 and first Section 3 proposition extracted.
Next: `prop:correct_ensemble_of_size_k_problem` (CDBEkP is NP-complete, Section 4).

### 2026-06-20 -- Ensemble Simplicity & CDBEkP (Sections 3-4 initial)

- Resolved $\prec$ conflict: thesis $\prec$ (sorted cardinalities) renamed to $\prec_A$
  ("attribute-simpler"), thesis SCDBEP renamed to ASCDBEP. Updated 6 files.
- Extracted `def:bireduct_ensemble_simpler` ($\prec$ via $EnsembleDescLen$) and
  `def:simplest_correct_ensemble_problem` (SCDBEP) into `concept-bireduct-ensemble`.
- Extracted `def:correct_ensemble_of_size_k_problem` (CDBEkP) -- definition moved into
  `concept-bireduct-ensemble`; standalone `concept-cdbe-kp.md` removed for consistency with
  SCDBEP/ASCDBEP.
- Created `prop-cdbe-kp-np-complete.md` (statement + proof strategy). Created subdirectory
  `propositions/cdbe-kp-np-complete/` for auxiliary lemmas of the Set Cover reduction.
- Updated `index.md` accordingly.
- `def:simple_bireducts_ensemble` still deferred (Section 4 auxiliary definition).

### 2026-06-20 -- Ensemble Properties (Section 3)

- Extracted `prop-correct-ensemble-iff-dectab-consistent` from source (lines 519-547 of
  `tmp/pub/main.tex`): a correct ensemble of decision bireducts exists iff the decision table is
  consistent. Proof verified as complete and correct. Two directions: ($\Rightarrow$) by
  contradiction using coverage counts and the fact that inconsistent pairs cannot co-occur in any
  bireduct; ($\Leftarrow$) by constructing a singleton ensemble $\{(U, B)\}$ from any decision
  reduct $B$.
- Extracted `concept-cdbe-kp` from source (lines 710-715 of `tmp/pub/main.tex`): the CDBEkP
  decision problem -- does a correct ensemble with $EnsembleDescLen \leq k$ exist? NP-complete.
- Updated `index.md` with both entries.
- **Deferred**: `def:bireduct_ensemble_simpler` and `def:simplest_correct_ensemble_problem` --
  conflict with existing $\prec$ definition in `concept-bireduct-ensemble` (thesis uses sorted
  cardinalities, this paper uses description length). Needs user decision on how to reconcile.
- **Deferred**: `def:simple_bireducts_ensemble` -- auxiliary definition from Section 4, out of scope
  for current section.

**Status**: Section 3 definitions fully extracted. $\prec$ conflict resolved: thesis version
renamed to $\prec_A$/ASCDBEP, paper version uses $\prec$/SCDBEP. Next: Section 4
propositions (CDBEkP NP-completeness proof).

### 2026-06-11 -- Foundational Lemmas (Part 4)

- Extracted `prop-bireduct-equiv-classes-geq-bplus1` from source (lines 493-535 of `tmp/pub/main.tex`):
  $|X/B| \geq |B| + 1$ for any bireduct $(X, B)$. Proof by induction on $|B|$ is complete and correct.
- Key dependencies: uses `prop-bireduct-attrs-subset-form-bireduct` (which still needs verification
  from Polish proof), `prop-equiv-classes-bireduct`, `prop-equiv-classes-monotonicity`, and
  `prop-bireduct-objects-and-rules`.
- Updated `index.md` with new entry.

**Status**: 5 of 6 foundational lemmas extracted. All foundational lemmas from Section 2 are now
complete. Next: `prop:correct_ensemble_iff_dectab_consistent` (first proposition in Section 3,
"Ensemble Properties").

### 2026-06-11 -- Foundational Lemmas (Part 3)

- Extracted `prop-bireduct-desc-len-formula` from source (lines 460-489 of `tmp/pub/main.tex`):
  $BireductDescLen(X, B) = |X/B| \cdot (|B| + 1)$. Proof is a direct consequence of the rule
  structure: each rule has $|B| + 1$ descriptors, and there are $|X/B|$ rules.
- Added remark connecting to `prop-equiv-classes-bireduct` for the equivalent form with $|U/B|$.
- Updated `index.md` with new entry.

**Status**: 3 of 6 foundational lemmas extracted. Next: `prop:bireduct_equiv_classes_geq_bplus1`
($|X/B| \geq |B|+1$ for any bireduct, proof by induction).

### 2026-06-10 -- Foundational Lemmas (Part 2)

- Extracted `prop-equiv-classes-bireduct` from source (lines 405-452 of `tmp/pub/main.tex`): for
  bireduct $(X, B)$, $|X/B| = |U/B|$. Proof verified as complete and correct. Key step: surjectivity
  uses maximality of $X$ in the bireduct definition.
- Updated `prop-equiv-classes-monotonicity.md` to link to the new proposition in Consequences.
- Updated `index.md` with new entry.

**Status**: 2 of 6 foundational lemmas extracted. Next: `prop:bireduct_desc_len_equals_xb_bplus1`
(description length formula).

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
