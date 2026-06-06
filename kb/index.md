# Knowledge Base Index

A curated catalog of all wiki pages. Organized by category. Each entry lists the page `id`, a
link, and a one-line summary. Updated on every ingest.

---

## Concepts

<!-- Format: - `id`: [Title](path) -- one-line summary -->

- `concept-decision-table`: [Decision Table](concepts/decision-table.md) -- The standard tabular
  data representation pairing a universe of objects with conditional and decision attributes.
- `concept-classification`: [Classification](concepts/classification.md) -- Classification task
  types and classification models that assign decision values to objects.
- `concept-classification-evaluation`: [Classification Evaluation](concepts/classification-evaluation.md)
  -- Evaluation metrics for assessing classifier performance (precision, recall, F1, accuracy,
  balanced accuracy, and related concepts).
- `concept-indiscernibility`: [Indiscernibility](concepts/indiscernibility.md) -- The fundamental
  equivalence relation of rough set theory, with its complement (discernibility), quotient sets,
  and equivalence classes.
- `concept-approximations`: [Approximations](concepts/approximations.md) -- Lower and upper
  approximations as the mechanism for handling imperfect knowledge about concepts.
- `concept-consistency`: [Consistency](concepts/consistency.md) -- Consistent and inconsistent
  decision tables; equivalent formulations of the consistency condition.
- `concept-formulae`: [Formulae](concepts/formulae.md) -- Propositional language of descriptors,
  selectors, and formulae with their meaning in a decision table.
- `concept-decision-rule`: [Decision Rule](concepts/decision-rule.md) -- If-then expressions
  connecting conditional attributes to the decision, with support and confidence.
- `concept-decision-reduct`: [Decision Reduct](concepts/reducts.md) -- Irreducible subsets of
  attributes preserving discernibility; includes classical, discernibility-based, and minimal
  variants, with Boolean formula characterization.
- `concept-discernibility-measure`: [Discernibility Measure](concepts/discernibility-measure.md) --
  Count of discerned object pairs with different decisions.
- `concept-positive-region`: [Positive Region](concepts/positive-region.md) -- Objects uniquely
  classifiable to decision classes; the dependency degree $\gamma$.
- `concept-gamma-decision-reduct`: [Gamma-Decision Reduct](concepts/gamma-decision-reduct.md) --
  Extension of decision reducts to inconsistent tables via preservation of the positive region,
  with consistent-table construction and Boolean formula.
- `concept-approximate-decision-reduct`: [Approximate Decision Reduct](concepts/approximate-decision-reduct.md)
  -- Reducts with an $\varepsilon$ threshold and evaluation function $F$; relative and absolute
  variants.
- `concept-majority-function`: [Majority Function](concepts/majority-function.md) -- $M(B)$:
  accuracy of a classifier pointing at the most frequent decision in each $B$-induced equivalence
  class.
- `concept-relative-gain-function`: [Relative Gain Function](concepts/relative-gain-function.md) --
  $R(B)$: Bayesian measure normalizing decision class frequencies against overall occurrence.
- `concept-np-hardness-foundations`: [NP-Hardness Foundations](concepts/np-hardness-foundations.md)
  -- Graph, dominating set, and $\alpha$-dominating set definitions used across complexity proofs.
- `concept-decision-bireduct`: [Decision Bireduct](concepts/decision-bireduct.md) -- Joint selection
  of attribute subset $B$ and object subset $X$; inexact functional dependency $B \Rrightarrow_X d$;
  Boolean and diagonal-table characterizations.
- `concept-gamma-decision-bireduct`: [Gamma-Decision Bireduct](concepts/gamma-decision-bireduct.md)
  -- Bireduct variant requiring discernibility against all of $U$; equivalence with the positive
  region.
- `concept-epsilon-decision-bireduct`: [Epsilon-Decision Bireduct](concepts/epsilon-decision-bireduct.md)
  -- Bireduct with coverage constraint $\lvert X \rvert \geq (1 - \varepsilon)\lvert U \rvert$;
  NP-hardness of the minimal variant.
- `concept-bireduct-ensemble`: [Bireduct Ensemble](concepts/bireduct-ensemble.md) -- Correct
  ensembles (majority voting), simplicity order $\prec$, and NP-hardness of the simplest correct
  ensemble problem.
- `concept-temporal-bireduct`: [Temporal Bireduct](concepts/temporal-bireduct.md) -- Bireduct with
  a continuous object range, designed for data stream processing.

## Propositions

<!-- Format: - `id`: [Title](path) -- one-line summary -->

- `prop-indiscernibility-equivalence-relation`: [Indiscernibility is an Equivalence Relation](propositions/indiscernibility-equivalence-relation.md)
  -- Proof that $IND(B)$ satisfies reflexivity, symmetry, and transitivity, with consequences for
  the quotient set partition.
- `prop-decision-reduct-boolean-formula`: [Decision Reduct Boolean Formula Characterisation](propositions/decision-reduct-boolean-formula.md)
  -- Decision reducts correspond to prime implicants of the discernibility Boolean formula $\tau$
  (Skowron & Rauszer, 1992).
- `prop-gamma-decision-reduct-consistent-table`: [Gamma-Decision Reduct via Consistent Table Construction](propositions/gamma-decision-reduct-consistent-table.md)
  -- A $\gamma$-decision reduct is exactly a standard decision reduct in the consistently modified
  table $\mathbb{A}_A^\gamma$ (Slezak, 2018).
- `prop-gamma-decision-reduct-boolean-formula`: [Gamma-Decision Reduct Boolean Formula Characterisation](propositions/gamma-decision-reduct-boolean-formula.md)
  -- $\gamma$-decision reducts correspond to prime implicants of a Boolean formula $\tau^\gamma$
  that restricts discernibility to pairs where the first element is in $POS(A)$.
- `prop-monotony-properties`: [Monotonicity Properties of Inexact Functional Dependency](propositions/monotony-properties.md)
  -- The dependency $B \Rrightarrow_X d$ is monotone with respect to attribute addition and object
  removal, justifying the irreducibility and maximality conditions in bireduct definition.
- `prop-decision-reduct-iff-bireduct`: [Decision Reduct iff Universe Bireduct](propositions/decision-reduct-iff-bireduct.md)
  -- $B$ is a decision reduct iff $(U, B)$ is a decision bireduct, embedding classical reducts
  into the bireduct framework.
- `prop-decision-bireduct-iff-reduct`: [Decision Bireduct via Subtable Reduct](propositions/decision-bireduct-iff-reduct.md)
  -- $(X, B)$ is a decision bireduct iff $\mathbb{A}_X^B$ is consistent, maximal in $X$, and $B$
  is a decision reduct for the subtable.
- `prop-bireduct-objects-and-rules`: [Bireduct Objects and Rules](propositions/bireduct-objects-and-rules.md)
  -- Each bireduct $(X, B)$ induces deterministic rules whose supports sum to exactly $X$, with
  three structural properties linking equivalence classes, decisions, and rule supports.
- `prop-gamma-monotony-properties`: [Monotonicity Properties of Gamma Functional Dependency](propositions/gamma-monotony-properties.md)
  -- The gamma dependency $B \Rrightarrow^{\gamma}_X d$ is monotone with respect to attribute
  addition and object removal, justifying irreducibility and maximality in $\gamma$-bireducts.
- `prop-gamma-decision-bireduct-to-reduct`: [Decision Reduct iff Universe Gamma-Bireduct](propositions/gamma-decision-bireduct-to-reduct.md)
  -- $B$ is a decision reduct iff $(U, B)$ is a $\gamma$-decision bireduct, embedding classical
  reducts into the $\gamma$-bireduct framework.
- `prop-gamma-decision-bireduct-pos`: [Gamma-Decision Bireduct Equals Positive Region](propositions/gamma-decision-bireduct-pos.md)
  -- $(X, B)$ is a $\gamma$-decision bireduct iff $X = POS(B)$ and $B$ is irreducible w.r.t.
  preserving the positive region; uniqueness and rule interpretation follow.

## Examples

<!-- Format: - `id`: [Title](path) -- one-line summary -->

- `ex-golf-reduct-rules`: [Golf Dataset -- Decision Reduct Rules](examples/golf-reduct-rules.md) --
  Complete decision rules generated from both decision reducts of the golf dataset.
- `ex-golf-gamma-reduct-rules`: [Golf Dataset -- Gamma-Decision Reduct Rules](examples/golf-gamma-reduct-rules.md)
  -- Gamma-modified decision tables and rules from $\gamma$-decision reducts for restricted
  attribute subsets $\{O, T, H\}$ and $\{T, H, W\}$.
- `ex-golf-bireduct-rules`: [Golf Dataset -- Bireduct Rules](examples/golf-bireduct-rules.md) --
  Decision rules from sample decision bireducts and $\gamma$-decision bireducts, with comparison of
  coverage.
- `ex-golf-all-bireducts`: [Golf Dataset -- Complete Bireduct Listing](examples/golf-all-bireducts.md)
  -- Full enumeration of all decision bireducts and $\gamma$-decision bireducts for the golf
  dataset.

## Source Summaries

*No entries yet.*

## Query Results

*No entries yet.*
