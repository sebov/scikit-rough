---
id: src-erickson-np-hardness-methodology
type: source-summary
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [complexity, methodology]
requires: [concept-np-hardness-foundations]
see_also:
  [prop-ensemble-np-hard,
   prop-minimal-epsilon-bireduct-np-hard]
source: "Erickson, J. Algorithms (unpublished textbook), Chapter 12: NP-Hardness"
---

# How to Prove NP-Hardness: Methodology

A distillation of the general pattern for polynomial-time reductions, drawn from Jeff Erickson's
*Algorithms* and applied concretely in the knowledge base.

## The Three-Step Template

To reduce problem $X$ to problem $Y$ in polynomial time:

1. **Transformation.** Describe a polynomial-time algorithm to transform an arbitrary instance
   $x$ of $X$ into a special instance $y$ of $Y$.
2. **(⇒) Feasibility.** Prove that if $x$ is a "good" instance of $X$, then $y$ is a "good"
   instance of $Y$.
3. **(⇐) Optimality/Correctness.** Prove that if $y$ is a "good" instance of $Y$, then $x$ is a
   "good" instance of $X$.

What "good" means depends on the problem type:

| Problem type | "Good" means |
| :----------- | :----------- |
| Decision (NP-complete) | Answer is **YES** (e.g., formula is satisfiable, graph has dominating set of size $\le k$) |
| Optimization (NP-hard) | Solution is **optimal** (e.g., smallest dominating set, attribute-simplest ensemble) |

## The Asymmetry: Why Step 3 Is Easier Than It Looks

> The reduction algorithm only "works one way" -- from $X$ to $Y$ -- but the correctness proof
> needs to "work both ways". The proofs are not symmetric. The "if" proof (step 2) needs to handle
> arbitrary instances of $X$, but the "only if" (step 3) only needs to handle the **special
> instances** of $Y$ produced by the reduction algorithm.

This means in step 3 you can exploit the specific structure of your $y = \text{transform}(x)$ --
constraints, symmetries, special values -- that are not present in arbitrary instances of $Y$.

## Certificate Perspective

Erickson reframes the three steps in terms of certificates (proofs that an instance is "good"):

1. Transform instance $x \in X$ into instance $y \in Y$ (polynomial time).
2. Transform a certificate for $x$ into a certificate for $y$.
3. Transform a certificate for $y$ into a certificate for $x$.

Only step 1 must run in polynomial time. Steps 2 and 3 are typically simpler.

## Concrete Example: CircuitSat to 3Sat

| Step | Action |
| :--- | :----- |
| 1 | Encode circuit $K$ as 3CNF formula $\Phi_3$: label each wire with a variable, express each gate as a sub-formula, expand to 3CNF |
| 2 | Given satisfying input for $K$: trace values through circuit, assign corresponding variables in $\Phi_3$, assign arbitrary values to remaining variables |
| 3 | Given satisfying assignment for $\Phi_3$: transfer wire variable values back to the corresponding wires in $K$ |

## Concrete Example: Minimal Dominating Set to ASCDBEP (optimization)

| Step | Action |
| :--- | :----- |
| 1 | Transform graph $\mathbb{G}$ into decision table $\mathbb{A}_{\mathbb{G}}$ with $|V|$ attributes (one per vertex), $|V|+1$ objects, and a special object $u_*$ with opposite decision |
| 2 | From dominating set $B$ of size $n$: construct ensemble with $n$ single-attribute bireducts and $n-1$ dummy bireducts, giving $OPT_Y \le n = OPT_X$ |
| 3 | From any correct ensemble: show it must have $\ge n$ ones (by contradiction using minimality of $B$), giving $OPT_Y \ge n = OPT_X$ |

Together: $OPT_X = OPT_Y$ -- the cost function is the identity $n \mapsto n$.

**Key insight for optimization:** In step 2 you do **not** need to prove the constructed solution is
optimal -- only that it is **feasible** (correct) and has the same cost $n$ as the source solution.
The inequality $OPT_Y \le OPT_X$ follows because the optimum cannot be worse than any known
feasible solution. Optimality in the other direction ($OPT_X \le OPT_Y$) is handled by step 3.

## Common Pitfalls

1. **Forgetting step 2 in optimization.** With only step 3, you have $OPT_X \le OPT_Y$ but no
   guarantee that $Y$ can actually achieve cost $OPT_X$. The reduction could be "lossy" --
   mapping an optimal $X$ to a suboptimal $Y$.

2. **Confusing construction size with cost.** In the ASCDBEP proof, the ensemble has $2n-1$
   elements but its cost under $\prec_A$ is $n$ (the number of ones). The $n-1$ zeros matter for
   correctness (you need at least that many) but excess zeros only make the ensemble more
   complex.

3. **Using minimality outside step 3.** The minimality of the dominating set is only needed in
   step 3 (to prove $m \not< n$). Step 2 works for **any** dominating set.

4. **Forgetting to verify that constructed objects belong to the target problem.** In ASCDBEP,
   each element of the ensemble must be verified to be a genuine decision bireduct (determination
   + irreducibility + maximality), not just a pair satisfying the functional dependency.
