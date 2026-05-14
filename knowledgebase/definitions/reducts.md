---
tags: [rst, core, reduction]
related: [definitions/decision_table.md, definitions/indiscernibility.md, definitions/consistency.md]
---
# Decision Reducts

## Decision Reduct

Let a consistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$
is a **decision reduct** for $\mathbb{A}$ if and only if it is an irreducible subset of attributes
such that:

$$IND(B) \subseteq IND(\{d\})$$

Equivalently, $B \subseteq A$ is a decision reduct if and only if it is an irreducible subset of
attributes such that each pair $u, u' \in U$ satisfying the inequality $d(u) \neq d(u')$ is
discerned by $B$.

A decision reduct is called **minimal** if its cardinality $|B|$ is the minimum among all decision
reducts of $\mathbb{A}$.

A decision reduct $B \subseteq A$ determines the decision values in $\mathbb{A}$. This means that
the universe $U$ can be covered by a set of decision rules where the predecessors are conjunctions
of descriptors $a_i = v_{a_i}$ for $a_i \in B$ and the successors are of the form $d = v_d$. This
property is closely related to the concept of **functional dependency** from relational database
theory, which can be equivalently formulated using if-then rules and the discernibility relation.

### Computation via Boolean Formulae

A general method for computing decision reducts for consistent tables can be formulated using
Boolean expressions.

**Proposition**: Let a consistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider
the following Boolean formula $\Phi$ with propositional variables $\overline{a}$ for $a \in A$:

$$
    \Phi =
        \bigwedge_{\substack{u_i, u_j \in U \\ i < j, \; d(u_i) \neq d(u_j)}}
        \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
        \overline{a}
$$

An arbitrary subset $B \subseteq A$ is a decision reduct if and only if the Boolean formula
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\Phi$.

It is worth noting that for inconsistent decision tables, decision reducts defined as above do not
exist. However, with different criteria, the process of reduction can still be considered. For
instance, one can focus on reducing attributes while preserving discernibility.

## Discernibility-based Decision Reduct

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$ is a
**discernibility-based decision reduct** for $\mathbb{A}$ if and only if it is an irreducible subset
of attributes that discerns the same pairs of objects with different decision values as $A$:

$$\forall u, u' \in U, (u \; DIS(A) \; u' \wedge d(u) \neq d(u')) \implies (u \; DIS(B) \; u')$$

## Discernibility Measure

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. The **discernibility measure**
$disc_\mathbb{A} : 2^A \rightarrow \mathbb{N}$ is defined for $B \subseteq A$ as follows:

$$disc_\mathbb{A}(B) = |\{(u, u') \in U \times U : u \; DIS(B) \; u' \wedge d(u) \neq d(u')\}|$$

## Example

For the example golf decision table, there are two decision reducts:
$\{ \text{Outlook}, \text{Temperature}, \text{Wind} \}$ and
$\{ \text{Outlook}, \text{Humidity}, \text{Wind} \}$.
