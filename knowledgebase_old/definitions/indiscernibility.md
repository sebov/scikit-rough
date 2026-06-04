---
tags: [rst, core]
related: [definitions/decision_table.md, notation_and_symbols.md, definitions/approximations.md]
---
# Indiscernibility and Discernibility Relations

## Indiscernibility Relation

Let a decision table be given. An attribute subset $B \subseteq A \cup \{d\}$ determines a binary
relation $IND(B)$ on $U$:

$$u \; IND(B) \; u' \Longleftrightarrow \forall a \in B, a(u) = a(u')$$

In such a case, we say that $u$ and $u'$ are indiscernible by attributes of $B$.

### Proposition: Equivalence Relation

Let a decision table and an attribute subset $B \subseteq A \cup \{d\}$ be given. The
indiscernibility relation $IND(B)$ is an equivalence relation.

**Proof**: The fact that $IND(B)$ satisfies all the properties of an equivalence relation
(reflexivity, symmetry, and transitivity) follows straightforwardly from the definition,
specifically from the properties of equality of values on all attributes from $B$.

## Discernibility Relation

Let a decision table be given. An attribute subset $B \subseteq A \cup \{d\}$ determines a binary
relation $DIS(B)$ on $U$ defined as the complement of the relation $IND(B)$:

$$
  u \; DIS(B) \; u' \Longleftrightarrow \neg (u \; IND(B) \; u')
  \Longleftrightarrow
  \exists a \in B, a(u) \neq a(u')
$$

In such a case, we say that an attribute subset $B$ discerns objects $u, u' \in U$ (or that $u, u'
\in U$ are discerned by $B$).
