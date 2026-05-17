---
tags: [rst, core, reduction]
related: [definitions/decision_table.md, definitions/indiscernibility.md, definitions/consistency.md, definitions/positive_region.md, definitions/decision_rules.md, propositions/decision_reduct_boolean_formula.md, propositions/gamma_decision_reduct_characterization.md]
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
exist. However, a number of extensions have been proposed, based on such notions as positive regions,
generalized decision functions, or rough membership functions. One such extension is the
$\gamma$-decision reduct defined below.

## Discernibility-based Decision Reduct

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$ is a
**discernibility-based decision reduct** for $\mathbb{A}$ if and only if it is an irreducible subset
of attributes that discerns the same pairs of objects with different decision values as $A$:

$$\forall u, u' \in U, (u \; DIS(A) \; u' \wedge d(u) \neq d(u')) \implies (u \; DIS(B) \; u')$$

## $\gamma$-Decision Reduct

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$ is a
$\gamma$**-decision reduct** for $\mathbb{A}$ if and only if it is an irreducible subset of
attributes such that $\gamma(B) = \gamma(A)$, or equivalently $POS_B(d) = POS_A(d)$.

For consistent decision tables $\gamma(A) = 1$, and the $\gamma$-decision reduct coincides with the
standard decision reduct.

### Construction of a Consistent Decision Table

A $\gamma$-decision reduct can be characterized via a transformation to a consistent decision table.
Given a subset $B \subseteq A$, define a modified decision attribute
$d_B^\gamma : U \rightarrow V_d \cup \{*\}$ where $* \notin V_d$ is a special value:

$$
  d_B^\gamma(u) =
  \begin{cases}
    *      & \text{if } u \notin POS_B(d), \\
    d(u)   & \text{otherwise}.
  \end{cases}
$$

The resulting decision table $\mathbb{A}_B^\gamma = (U, A \cup \{d_B^\gamma\})$ is consistent. In
particular, for $B = A$, a subset $B \subseteq A$ is a $\gamma$-decision reduct for $\mathbb{A}$ if
and only if it is a decision reduct for the modified consistent table $\mathbb{A}_A^\gamma$.

### Interpretation via Decision Rules

The positive region $POS_B(d)$ can be interpreted in terms of decision rules generated from
combinations of attribute values in $B$. Rules generated from objects in $POS_B(d)$ are
deterministic -- their confidence equals $1$. The role of $\gamma$-decision reducts is to use
possibly small subsets of attributes to cover data with deterministic rules as thoroughly as would be
possible using the full set of attributes.

For an example with modified decision tables and decision rules generated from $\gamma$-decision
reducts, see the golf dataset example in `decision_table.md`.

## Discernibility Measure

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. The **discernibility measure**
$disc_\mathbb{A} : 2^A \rightarrow \mathbb{N}$ is defined for $B \subseteq A$ as follows:

$$disc_\mathbb{A}(B) = |\{(u, u') \in U \times U : u \; DIS(B) \; u' \wedge d(u) \neq d(u')\}|$$

## Example

For the example golf decision table, there are two decision reducts:
$\{ \text{Outlook}, \text{Temperature}, \text{Wind} \}$ and
$\{ \text{Outlook}, \text{Humidity}, \text{Wind} \}$.
