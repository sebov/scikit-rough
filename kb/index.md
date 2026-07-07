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
- `concept-decision-reduct`: [Decision Reduct](concepts/decision-reduct.md) -- Irreducible subsets of
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
  ensembles (majority voting); two simplicity orders $\prec_A$ (attribute-based) and $\prec$
  (description-length); NP-hardness of both ASCDBEP and SCDBEP, NP-completeness of CDBEkP.
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
- `prop-decision-bireduct-boolean-formula`: [Decision Bireduct Boolean Formula Characterisation](propositions/decision-bireduct-boolean-formula.md)
  -- Decision bireducts correspond to prime implicants of a Boolean formula $\tau_{bi}$ with
  propositional variables for both objects and attributes.
- `prop-decision-table-diagonal`: [Diagonal Table Transformation for Bireducts](propositions/decision-table-diagonal.md)
  -- Decision bireducts can be computed as standard reducts on a table augmented with diagonal
  attributes that uniquely identify each object.
- `prop-gamma-decision-bireduct-boolean-formula`: [Gamma-Decision Bireduct Boolean Formula Characterisation](propositions/gamma-decision-bireduct-boolean-formula.md)
  -- $\gamma$-decision bireducts correspond to prime implicants of a more restrictive Boolean
  formula $\tau_{bi}^{\gamma}$; connection to positive region via formula transformation.
- `prop-m-reduct-epsilon-bireduct-correspondence`: [Correspondence Between M-Reducts and Epsilon-Bireducts](propositions/m-reduct-epsilon-bireduct-correspondence.md)
  -- Bidirectional correspondence between smallest $M$-reducts and $\varepsilon$-bireducts; enables
  complexity result transfer.
- `prop-minimal-epsilon-bireduct-np-hard`: [NP-Hardness of Minimal Epsilon-Bireduct Problem](propositions/minimal-epsilon-bireduct-np-hard.md)
  -- For any $\varepsilon \in [0, 1)$, finding an $\varepsilon$-decision bireduct with minimum
  attributes is NP-hard (reduction from minimal $M$-reduct problem).
- `prop-ensemble-np-hard`: [NP-Hardness of Attribute-Simplest Correct Ensemble Problem](propositions/ensemble-np-hard.md)
  -- ASCDBEP is NP-hard via polynomial reduction from Minimal Dominating Set; graph encoding into
  decision table with dummy classifiers.
- `prop-decision-bireduct-ordering`: [Correctness of the Decision Bireduct Ordering Algorithm](propositions/decision-bireduct-ordering.md)
  -- The ordering algorithm always outputs a decision bireduct; every decision bireduct is
  achievable via a specific permutation.
- `prop-gamma-decision-bireduct-ordering`: [Correctness of the Gamma Decision Bireduct Ordering Algorithm](propositions/gamma-decision-bireduct-ordering.md)
  -- Analogous to the standard ordering case, using $\gamma$-dependency and $\gamma$-monotony.
- `prop-decision-bireduct-sampling`: [Correctness of the Decision Bireduct Sampling Algorithm](propositions/decision-bireduct-sampling.md)
  -- The sampling algorithm produces a decision bireduct via attribute sampling, representative
  selection, and classical reduct computation.
- `prop-gamma-decision-bireduct-sampling`: [Correctness of the Gamma Decision Bireduct Sampling Algorithm](propositions/gamma-decision-bireduct-sampling.md)
  -- Simpler $\gamma$-sampling case where the object set is always $POS_{\mathbb{A}}(B)$.
- `prop-temporal-bireduct-computation`: [Temporal Bireduct Computation via Streaming Buffer](propositions/temporal-bireduct-computation.md)
  -- Sequential streaming algorithm produces temporal bireducts; forward/backward
  non-extendability and buffer-based proof that every temporal bireduct is achievable.
- `prop-minimal-dominating-set-np-hard`: [NP-Hardness of the Minimal Dominating Set Problem](propositions/minimal-dominating-set-np-hard.md)
  -- The classical NP-hardness of MDS (Garey & Johnson, 1979); base of the polynomial reduction
  chain leading to approximate reducts.
- `prop-relative-gamma-epsilon-reduct-np-hard`: [NP-Hardness of Minimal Relative Gamma-Decision Epsilon-Reduct](propositions/relative-gamma-epsilon-reduct-np-hard.md)
  -- Reduction from Minimal Dominating Set via graph-to-table construction with auxiliary objects;
  the $\gamma$-superreduct condition forces those objects into the positive region.
- `prop-gamma-epsilon-reduct-np-hard`: [NP-Hardness of Minimal Gamma-Decision Epsilon-Reduct](propositions/gamma-epsilon-reduct-np-hard.md)
  -- Absolute variant; follows from the relative case because the constructed table is consistent.
- `prop-alpha-dominating-set-np-hard`: [NP-Hardness of the Minimal Alpha-Dominating Set Problem](propositions/alpha-dominating-set-np-hard.md)
  -- Generalization of MDS with relaxed coverage ratio; NP-hard for any $\alpha \in (0, 1]$
  (Slezak, 2000).
- `prop-relative-m-epsilon-reduct-np-hard`: [NP-Hardness of Minimal Relative M-Decision Epsilon-Reduct](propositions/relative-m-epsilon-reduct-np-hard.md)
  -- Reduction from Minimal $\alpha$-Dominating Set; construction links $M(B)$ to dominating set
  coverage via the formula $\alpha(\varepsilon) = 1 - \varepsilon/(1 - m(\varepsilon)^{-1})$.
- `prop-m-epsilon-reduct-np-hard`: [NP-Hardness of Minimal M-Decision Epsilon-Reduct](propositions/m-epsilon-reduct-np-hard.md)
  -- Absolute variant; follows from the relative case using consistency of the constructed table.
- `prop-relative-r-epsilon-reduct-np-hard`: [NP-Hardness of Minimal Relative R-Decision Epsilon-Reduct](propositions/relative-r-epsilon-reduct-np-hard.md)
  -- Same construction as the $M$ case; follows from the observation that $R(B) = M(B)$ on the
  constructed table.
- `prop-r-epsilon-reduct-np-hard`: [NP-Hardness of Minimal R-Decision Epsilon-Reduct](propositions/r-epsilon-reduct-np-hard.md)
  -- Absolute variant; follows from the same observations as the $M$ absolute case, applied to $R$.
- `prop-equiv-classes-monotonicity`: [Monotonicity of Equivalence Class Count](propositions/equiv-classes-monotonicity.md)
  -- $|U/B| \leq |U/B'|$ for $B \subseteq B'$; equality iff partitions coincide. Foundational lemma
     for bireduct description length.
- `prop-equiv-classes-bireduct`: [Equivalence Class Count for Bireducts](propositions/equiv-classes-bireduct.md)
  -- For bireduct $(X, B)$: $|X/B| = |U/B|$; connects local bireduct structure to global partition.
- `prop-bireduct-desc-len-formula`: [Bireduct Description Length Formula](propositions/bireduct-desc-len-formula.md)
  -- $BireductDescLen(X, B) = |X/B| \cdot (|B| + 1)$; direct consequence of rule structure.
- `prop-bireduct-desc-len-geq-bplus1-squared`: [Bireduct Description Length Lower Bound](propositions/bireduct-desc-len-geq-bplus1-squared.md)
  -- $BireductDescLen(X, B) \geq (|B| + 1)^2$; immediate corollary of description-length formula and
  equivalence-classes bound.
- `prop-correct-ensemble-iff-dectab-consistent`: [Correct Ensemble Exists iff Decision Table is
  Consistent](propositions/correct-ensemble-iff-dectab-consistent.md)
  -- A correct ensemble of decision bireducts exists iff the table is consistent; fundamental link
  between ensemble existence and classical consistency.
- `prop-bireduct-equiv-classes-geq-bplus1`: [Bireduct Equivalence Classes Lower Bound](propositions/bireduct-equiv-classes-geq-bplus1.md)
  -- For any decision bireduct $(X, B)$: $|X/B| \geq |B| + 1$; graph-theoretic proof via forest
  construction over equivalence classes.
- `prop-cdbe-kp-np-complete`: [NP-Completeness of CDBEkP](propositions/cdbe-kp-np-complete.md)
  -- CDBEkP is NP-complete via reduction from Set Cover; proof with auxiliary lemmas in
  [cdbe-kp-np-complete/](propositions/cdbe-kp-np-complete/).
- `prop-set-cover-problem`: [Set Cover Problem](propositions/cdbe-kp-np-complete/set-cover-problem.md)
  -- Formal definition of the decision version of the Set Cover problem (Karp, 1972).
- `prop-set-cover-construction`: [Construction of Decision Table from a Set Cover Instance](propositions/cdbe-kp-np-complete/set-cover-construction.md)
  -- Polynomial-time transformation from a Set Cover instance to a decision table
  $\mathbb{A}_{\mathcal{S}}$.
- `prop-transformed-table-consistent`: [Consistency of the Transformed Decision Table](propositions/cdbe-kp-np-complete/transformed-table-consistent.md)
  -- $\mathbb{A}_{\mathcal{S}}$ is always consistent; proof via object structure.
- `prop-solution-bireduct-properties`: [Structural Properties of Bireducts in the Transformed Table](propositions/cdbe-kp-np-complete/solution-bireduct-properties.md)
  -- For $|B| \geq 1$, $X = \{u_*\} \cup \{u_\omega \mid \exists a_{S_i} \in B : a_{S_i}(u_\omega) = 1\}$; for $B = \emptyset$, $X = \{u_*\}$ or $X = U_{\mathcal{S}}$.
- `prop-bireducts-0-and-1-attrs-desc-size`: [Description Lengths of Bireducts with 0 or 1 Attributes](propositions/cdbe-kp-np-complete/bireducts-0-and-1-attrs-desc-size.md)
  -- $BireductDescLen(X, \emptyset) = 1$, $BireductDescLen(X, \{b\}) = 4$ in $\mathbb{A}_{\mathcal{S}}$.

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
- `ex-golf-epsilon-bireducts-m-reducts`: [Golf Dataset -- Epsilon-Bireducts and M-Reducts](examples/golf-epsilon-bireducts-m-reducts.md)
  -- Table of $M$-decision $\varepsilon$-reducts and $\varepsilon$-bireduct objects for every
  attribute subset ($\varepsilon = 4/14$).
- `ex-golf-epsilon-bireduct-ensembles`: [Golf Dataset -- Epsilon-Bireduct Ensembles](examples/golf-epsilon-bireduct-ensembles.md)
  -- Seven example correct 3-element $\varepsilon$-bireduct ensembles with per-object coverage
  counts.
- `ex-golf-permutation-bireducts`: [Golf Dataset -- Decision Bireducts from the Ordering Algorithm](examples/golf-permutation-bireducts.md)
  -- Fifteen permutations and the decision bireducts produced by the ordering algorithm for each.
- `ex-golf-permutation-gamma-bireducts`: [Golf Dataset -- Gamma-Decision Bireducts from the Ordering Algorithm](examples/golf-permutation-gamma-bireducts.md)
  -- Same permutations applied to the $\gamma$-decision bireduct ordering algorithm.
- `ex-golf-bireduct-cnf-dnf`: [Golf Dataset -- Bireduct Boolean Formula (CNF/DNF)](examples/golf-bireduct-cnf-dnf.md)
  -- $\tau_{bi}$ in CNF (45 clauses) and DNF (69 terms); prime implicants correspond to decision
  bireducts.
- `ex-golf-gamma-bireduct-cnf-dnf`: [Golf Dataset -- Gamma-Bireduct Boolean Formula (CNF/DNF)](examples/golf-gamma-bireduct-cnf-dnf.md)
  -- $\tau_{bi}^{\gamma}$ in CNF (82 clauses) and DNF (12 terms); prime implicants correspond to
  $\gamma$-decision bireducts.
- `ex-golf-diagonal-table`: [Golf Dataset -- Diagonal Table Transformation](examples/golf-diagonal-table.md)
  -- Decision table augmented with 14 binary diagonal attributes; bireducts become standard reducts
  on this table.
- `ex-temporal-bireduct-walkthrough`: [Temporal Bireduct Computation Walkthrough](examples/temporal-bireduct-walkthrough.md)
  -- Step-by-step trace of the streaming buffer algorithm on the golf dataset for $A' = A$ and
  $A' = \{T,H,W\}$.
- `ex-nphard-construction-tables`: [NP-Hardness Construction -- Example Decision Tables](examples/nphard-construction-tables.md)
  -- Concrete decision tables illustrating the graph-to-table transformations from the $\gamma$ and
  $M$ NP-hardness reductions.

## Source Summaries

- `src-thesis-phd`: [PhD Thesis: Decision Bireducts in Rough Set Theory](sources/thesis-phd.md)
  -- Primary source for the KB: doctoral dissertation covering bireduct theory, variants, complexity,
  and algorithms. 64 wiki pages extracted from this source.
- `src-erickson-np-hardness-methodology`: [How to Prove NP-Hardness: Methodology](sources/erickson-np-hardness-methodology.md)
  -- Three-step template for polynomial-time reductions, certificate perspective, and common
  pitfalls, drawn from Erickson's *Algorithms* and illustrated with concrete examples from this KB.
- `src-llm-wiki`: [LLM Wiki Pattern (Karpathy)](sources/llm-wiki.md)
  -- Design document describing the three-layer architecture (raw sources, wiki, schema) and
  operations (ingest, query, lint) for LLM-maintained knowledge bases. Reference for this KB's design.

## Query Results

*No entries yet.*
