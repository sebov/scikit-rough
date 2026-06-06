---
id: prop-gamma-decision-bireduct-boolean-formula
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, boolean-reasoning, positive-region]
requires:
  [concept-gamma-decision-bireduct,
   concept-formulae]
see_also:
  [prop-decision-bireduct-boolean-formula,
   prop-gamma-decision-reduct-boolean-formula,
   prop-gamma-decision-bireduct-pos]
source: tmp/phd/thesis.tex
---

# Gamma-Decision Bireduct Boolean Formula Characterisation

$\gamma$-decision bireducts correspond to prime implicants of a Boolean formula $\tau_{bi}^{\gamma}$
that is more restrictive than the standard bireduct formula -- if no attribute is selected that
discerns two objects, then neither can be contained in a $\gamma$-decision bireduct.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the following Boolean formula with
propositional variables $\overline{i}$ (for $i = 1, \ldots, \lvert U \rvert$) and $\overline{a}$
(for $a \in A$):

$$
\tau_{bi}^{\gamma} = \bigwedge_{u_i \in U}
  \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
  \left(
    \overline{i} \lor
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}} \overline{a}
  \right)
$$

A pair $(X, B)$, where $X \subseteq U$ and $B \subseteq A$, is a $\gamma$-decision bireduct if and
only if the Boolean formula

$$
P^{\gamma} = \bigwedge_{u_i \in U \setminus X} \overline{i} \land \bigwedge_{a \in B} \overline{a}
$$

is a prime implicant for $\tau_{bi}^{\gamma}$.

## Proof

The proof is fully analogous to that presented for
[prop-decision-bireduct-boolean-formula](decision-bireduct-boolean-formula.md).

### Step 1: Gamma Functional Dependency Equivalence

First, we show that:

$$
B \Rrightarrow^{\gamma}_X d \;\Longleftrightarrow\; P^{\gamma} \text{ is an implicant for } \tau_{bi}^{\gamma}
$$

**($\Rightarrow$)** Suppose that $P^{\gamma}$ is not an implicant for $\tau_{bi}^{\gamma}$. Hence,
there exists a valuation of propositional variables for which $P^{\gamma}$ is true but
$\tau_{bi}^{\gamma}$ is false. Thus, there is at least one clause of the form

$$
f_k = \left(\overline{i_k} \lor \bigvee_{\substack{a \in A \\ a(u_{i_k}) \neq a(u_{j_k})}} \overline{a}\right)
$$

where $u_{i_k} \in U$, $u_{j_k} \in U$, and $d(u_{i_k}) \neq d(u_{j_k})$, which is false for the
considered valuation. As $f_k$ is a disjunction, all its elements must be assigned false. Since
$P^{\gamma}$ is true and $\overline{i_k}$ is false, $\overline{i_k}$ is not part of $P^{\gamma}$.
Since $P^{\gamma}$ contains variables corresponding to all objects in $U \setminus X$, we know that
$u_{i_k} \in X$. We also know that $P^{\gamma}$ cannot contain variables corresponding to
attributes for which $a(u_{i_k}) \neq a(u_{j_k})$, i.e., for all $a \in A$ such that
$a(u_{i_k}) \neq a(u_{j_k})$, we know that $a \notin B$. This means that there exists
$u_{i_k} \in X$ and $u_{j_k} \in U$ such that $d(u_{i_k}) \neq d(u_{j_k})$ which are not discerned
by $B$. Therefore, $B \Rrightarrow^{\gamma}_X d$ does not hold.

**($\Leftarrow$)** Suppose that $B \Rrightarrow^{\gamma}_X d$ does not hold. This means that there
exists $u_{i_k} \in X$ and $u_{j_k} \in U$ such that $d(u_{i_k}) \neq d(u_{j_k})$ which is not
discerned by $B$. Consider the corresponding clause $f_k$. Assign false to the variables in $f_k$
and true to all others. For such valuation, $P^{\gamma}$ is true because it does not share any
elements with $f_k$ (the object $u_{i_k} \in X$, so $\overline{i_k}$ is not in $P^{\gamma}$; the
attributes do not discern the pair, so their variables are not in $P^{\gamma}$). On the other hand,
$\tau_{bi}^{\gamma}$ is false. Thus, $P^{\gamma}$ is not an implicant for $\tau_{bi}^{\gamma}$.

### Step 2: Gamma Bireduct Equivalence

Using Step 1, one can show analogously to
[prop-decision-bireduct-boolean-formula](decision-bireduct-boolean-formula.md) that:

$$
(X, B) \text{ is a } \gamma\text{-decision bireduct } \;\Longleftrightarrow\; P^{\gamma} \text{ is a prime implicant for } \tau_{bi}^{\gamma}
$$

The argument follows the same structure: if $P^{\gamma}$ is not a prime implicant, then either it
is not an implicant (violating gamma-determination) or it is not minimal (violating attribute
irreducibility or object maximality). Conversely, if $(X, B)$ is not a $\gamma$-decision bireduct,
then one of the three conditions fails, which translates to $P^{\gamma}$ not being a prime
implicant.

## Remarks

The formula $\tau_{bi}^{\gamma}$ can be transformed to reveal the connection to the positive region:

$$
\tau_{bi}^{\gamma} = \bigwedge_{u_i \in U}
  \left(
    \overline{i} \lor
    \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}} \overline{a}
  \right)
$$

This means that, as we are interested in prime implicants, either an object belongs to the positive
region $POS(A)$ and then it belongs to a $\gamma$-decision bireduct, or in other case, it is
definitively excluded from a $\gamma$-decision bireduct. This shows that $\gamma$-decision bireducts
are more restrictive than standard decision bireducts.
