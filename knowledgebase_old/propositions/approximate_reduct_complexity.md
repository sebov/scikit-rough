---
tags: [rst, core, proposition, complexity]
related: [notation_and_symbols.md, definitions/approximate_reducts.md, definitions/positive_region.md, definitions/decision_table.md]
---

# Complexity of Approximate Reducts

The search for minimal $\varepsilon$-$F$-reducts is NP-hard for functions such as $\gamma$, $M$,
and $R$. The proofs use a polynomial reduction from the Minimal Dominating Set Problem.

## Proposition 1: Minimal Relative $\varepsilon$-$\gamma$-Decision Reduct

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal relative
$\varepsilon$-$\gamma$-decision reduct for an input decision table is **NP-hard**.

### Proof Idea

The reduction transforms a graph $G = (V, E)$ into a decision table
$\mathbb{A}^\varepsilon_G = (U^\varepsilon_G, A^\varepsilon_G \cup \{d^\varepsilon_G\})$ as follows:

- For each vertex $v_j \in V$, create a binary conditional attribute $a_j$.
- For each vertex $v_i \in V$, create an object $u_i$ with $a_j(u_i) = 1$ if $i = j$ or
  $(v_i, v_j) \in E$, and $0$ otherwise.
- Add $t(\varepsilon) = \lfloor \frac{|V|\varepsilon}{1-\varepsilon} + 1 \rfloor$ additional
  objects with all attributes set to $0$.
- Set $d^\varepsilon_G(u_i) = 0$ for vertex-objects and $1$ for the extra objects.

The threshold $t(\varepsilon)$ ensures that a relative $\varepsilon$-$\gamma$-superreduct $B$ must
satisfy $|POS_B(d)| > |V|$, which forces it to discern all vertex-objects from at least one
extra object. This is equivalent to requiring that the set $W_B = \{ v_j \in V : a_j \in B \}$
is a dominating set for $G$. Minimality of $B$ corresponds to minimality of $W_B$, establishing
NP-hardness.

### Example

Consider the graph $G = (V, E)$ with $V = \{1, \dots, 8\}$ and edges forming a cube-like structure.
The reduction produces the following decision table $\mathbb{A}^\varepsilon_G$ (with $t(\varepsilon)$
extra objects at the bottom):

|   $U^\varepsilon_G$    |  $a_1$   |  $a_2$   |  $a_3$   |  $a_4$   |  $a_5$   |  $a_6$   |  $a_7$   |  $a_8$   | $d^\varepsilon_G$ |
| :--------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :---------------: |
|         $u_1$          |    1     |    1     |    0     |    1     |    1     |    0     |    0     |    0     |         0         |
|         $u_2$          |    1     |    1     |    1     |    0     |    0     |    1     |    0     |    0     |         0         |
|         $u_3$          |    0     |    1     |    1     |    1     |    0     |    0     |    1     |    0     |         0         |
|         $u_4$          |    1     |    0     |    1     |    1     |    0     |    0     |    0     |    1     |         0         |
|         $u_5$          |    1     |    0     |    0     |    0     |    1     |    1     |    0     |    1     |         0         |
|         $u_6$          |    0     |    1     |    0     |    0     |    1     |    1     |    1     |    0     |         0         |
|         $u_7$          |    0     |    0     |    1     |    0     |    0     |    1     |    1     |    1     |         0         |
|         $u_8$          |    0     |    0     |    0     |    1     |    1     |    0     |    1     |    1     |         0         |
|       $u_{8+1}$        |    0     |    0     |    0     |    0     |    0     |    0     |    0     |    0     |         1         |
|       $u_{8+2}$        |    0     |    0     |    0     |    0     |    0     |    0     |    0     |    0     |         1         |
|        $\cdots$        | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |     $\cdots$      |
| $u_{8+t(\varepsilon)}$ |    0     |    0     |    0     |    0     |    0     |    0     |    0     |    0     |         1         |

Each vertex-object $u_i$ has $1$ on attributes corresponding to itself and its neighbors in $G$,
and $0$ elsewhere. For instance, $\{a_1, a_2, a_4, a_5\}$ is a dominating set in the graph,
corresponding to the attribute subset $\{a_1, a_2, a_4, a_5\}$ which is a relative
$\varepsilon$-$\gamma$-superreduct in the table.

## Proposition 2: Minimal $\varepsilon$-$\gamma$-Decision Reduct

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal
$\varepsilon$-$\gamma$-decision reduct (absolute version) for a decision table is **NP-hard**.

### Proof Idea

The same construction yields a consistent decision table. Since $\gamma(A^\varepsilon_G) = 1$, the
relative and absolute versions coincide, making the proof identical to Proposition 1.

## $\alpha$-Dominating Set

The reductions for $M$ and $R$ rely on a generalized notion of dominating set.

Let $\alpha \in (0, 1]$ and a non-directed graph $G = (V, E)$ be given. A subset $W \subseteq V$ is
an **$\alpha$-dominating set** if and only if:

$$
  \frac{|Cov_G(W)|}{|V|} \geq \alpha
$$

where $Cov_G(W)$ is the set of vertices that belong to $W$ or are adjacent to a member of $W$.

For any $\alpha \in (0, 1]$, the problem of finding a minimal $\alpha$-dominating set is **NP-hard**.

## Proposition 3: Minimal Relative $\varepsilon$-$M$-Decision Reduct

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal relative
$\varepsilon$-$M$-decision reduct for a decision table is **NP-hard**.

### Proof Idea

The reduction transforms an $\alpha(\varepsilon)$-dominating set problem into a problem of finding a
minimal relative $\varepsilon$-$M$-reduct. For a given graph $G = (V, E)$, construct a decision
table $\mathbb{A}^\varepsilon_{G,M}$ with $|V|$ conditional attributes (one per vertex) and
$m(\varepsilon) = \lfloor (1-\varepsilon)^{-1} + 1 \rfloor$ objects per vertex, yielding
$|U| = m(\varepsilon)|V|$ objects total.

Each vertex $v_i$ generates a chunk of $m(\varepsilon)$ objects. Within a chunk, decision values
cycle through $1, 2, \dots, m(\varepsilon)$. Attribute $a_j$ takes value $i$ for object $u_i$ if
$\text{chunk}(i) = j$ (same vertex) or $(v_{\text{chunk}(i)}, v_j) \in E$ (adjacent vertices);
otherwise it takes the value $(\text{chunk}(i)-1)m(\varepsilon) + 1$ (the first index of the chunk).

For this table, it can be shown that:

$$
  M(B) = \frac{|Cov_G(W_B)| \cdot m(\varepsilon) + |V \setminus Cov_G(W_B)|}{|U|}
$$

where $W_B = \{ v_j \in V : a_j \in B \}$. Setting
$\alpha(\varepsilon) = 1 - \frac{\varepsilon}{1 - m(\varepsilon)^{-1}} \in (0, 1]$, the inequality
$M(B) \geq (1-\varepsilon)M(A)$ is equivalent to
$|Cov_G(W_B)|/|V| \geq \alpha(\varepsilon)$, i.e., $W_B$ is an $\alpha(\varepsilon)$-dominating
set. Minimality of $B$ corresponds to minimality of $W_B$, establishing NP-hardness.

### Example ($\varepsilon = 0.6$)

For $\varepsilon = 0.6$, we have $m(\varepsilon) = \lfloor 2.5 + 1 \rfloor = 3$ objects per vertex.
The table for the cube graph (8 vertices) has 24 objects, with decision values cycling $1,2,3$
within each chunk. Below are the first two chunks (vertices 1 and 2):

| $U^\varepsilon_{G,M}$ |  $a_1$   |  $a_2$   |  $a_3$   |  $a_4$   |  $a_5$   |  $a_6$   |  $a_7$   |  $a_8$   |   $d$    |
| :-------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
|         $u_1$         |    1     |    1     |    1     |    1     |    1     |    1     |    1     |    1     |    1     |
|         $u_2$         |    2     |    2     |    1     |    2     |    2     |    1     |    1     |    1     |    2     |
|         $u_3$         |    3     |    3     |    1     |    3     |    3     |    1     |    1     |    1     |    3     |
|         $u_4$         |    4     |    4     |    4     |    4     |    4     |    4     |    4     |    4     |    1     |
|         $u_5$         |    5     |    5     |    5     |    4     |    4     |    5     |    4     |    4     |    2     |
|         $u_6$         |    6     |    6     |    6     |    4     |    4     |    6     |    4     |    4     |    3     |
|       $\cdots$        | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |

The $M$ function's preference for majority decisions encodes the $\alpha$-domination criterion
through the chunk structure.

## Proposition 4: Minimal $\varepsilon$-$M$-Decision Reduct

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal $\varepsilon$-$M$-decision
reduct (absolute version) for a decision table is **NP-hard**.

### Proof Idea

The construction in Proposition 3 yields a consistent decision table. For consistent tables
$M(B) \geq (1-\varepsilon)M(A)$ and $M(B) \geq 1-\varepsilon$ are equivalent, so the same proof
applies.

## Proposition 5: Minimal Relative $\varepsilon$-$R$-Decision Reduct

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal relative
$\varepsilon$-$R$-decision reduct for a decision table is **NP-hard**.

### Proof Idea

The same construction as in Proposition 3 applies. For the constructed table
$\mathbb{A}^\varepsilon_{G,M}$, it holds that $R(B) = M(B)$ for any $B \subseteq A$. Therefore the
entire proof for $M$ carries over to $R$ unchanged.

## Proposition 6: Minimal $\varepsilon$-$R$-Decision Reduct

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal $\varepsilon$-$R$-decision
reduct (absolute version) for a decision table is **NP-hard**.

### Proof Idea

Same observation as in Proposition 4, applied to $R$ instead of $M$.
