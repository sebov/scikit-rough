---
tags: [core, notation]
related: [definitions/decision_table.md, definitions/indiscernibility.md, definitions/approximations.md, definitions/positive_region.md, definitions/reducts.md, definitions/approximate_reducts.md, definitions/consistency.md, definitions/decision_rules.md]
---
# Notation and Symbols

This document defines the mathematical symbols and notation used throughout the project.

- $U$: The universe of objects (e.g., a set of objects, records in a decision table).
- $A$: The set of conditional attributes, where each $a \in A$ is a function $a : U \rightarrow
  V_a$.
- $d$: The decision attribute, where $d : U \rightarrow V_d$ and $d \notin A$.
- $\mathbb{A} = (U, A \cup \{d\})$: A decision table.
- $IND(B)$: The indiscernibility relation determined by attribute subset $B$.
- $DIS(B)$: The discernibility relation determined by attribute subset $B$.
- $\tau$: Boolean formula used for computing decision reducts via prime implicants.
- $POS_B(d)$: The positive region of decision $d$ with respect to attribute subset $B$.
- $\gamma(B)$: The degree of dependency between attribute subset $B$ and the decision $d$.
- $*$: A special decision value introduced during the construction of a consistent decision table (not in $V_d$).
- $M(B)$: The majority function -- accuracy of a rule-based classifier pointing at the most frequent decision within each $B$-induced equivalence class.
- $R(B)$: The relative gain function -- a Bayesian measure of decision information induced by attribute subset $B$.
- $disc_\mathbb{A}(B)$: The discernibility measure of attribute subset $B$ in decision table $\mathbb{A}$.
