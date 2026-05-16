---
tags: [core, notation]
related: [definitions/decision_table.md, definitions/indiscernibility.md, definitions/approximations.md, definitions/reducts.md, definitions/consistency.md, definitions/decision_rules.md]
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
- $disc_\mathbb{A}(B)$: The discernibility measure of attribute subset $B$ in decision table $\mathbb{A}$.
