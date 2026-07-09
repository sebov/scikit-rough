# Notation and Symbols

Centralized registry of all mathematical symbols used across the knowledge base. This file is the
single source of truth for notation conventions. Executor agents must check this file before
creating new content and update it when introducing new symbols.

---

| Symbol                           | Name                          | Description                                                                                          | First Used In                          |
| :------------------------------- | :---------------------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------- |
| $U$                              | Universe                      | The universe of objects                                                                              | concept-decision-table                 |
| $A$                              | Conditional attributes        | Set of conditional attributes                                                                        | concept-decision-table                 |
| $d$                              | Decision attribute            | The distinguished decision attribute, $d \notin A$                                                   | concept-decision-table                 |
| $V_a$                            | Value set of $a$              | Codomain of attribute $a$, $a: U \to V_a$                                                            | concept-decision-table                 |
| $\mathbb{A}$                     | Decision table                | A pair $\mathbb{A} = (U, A \cup \{d\})$                                                              | concept-decision-table                 |
| $\mathbb{A}_X^B$                 | Sub-table                     | Decision table restricted to objects $X \subseteq U$ and attributes $B \subseteq A$                  | concept-decision-table                 |
| $\mathbb{A}_B^\gamma$            | Gamma decision table          | Gamma-modified consistent decision table for subset $B$                                              | concept-gamma-decision-reduct          |
| $\circledast$                    | Gamma special value           | Special decision value for gamma-modified tables, $\circledast \notin V_d$                           | concept-gamma-decision-reduct          |
| $IND(B)$                         | Indiscernibility relation     | Equivalence relation on $U$ induced by attribute subset $B$                                          | concept-indiscernibility               |
| $IND_V(B)$                       | Indiscernibility on subset    | Equivalence relation on $V \subseteq U$ induced by attribute subset $B$                              | concept-indiscernibility               |
| $DIS(B)$                         | Discernibility relation       | Complement of $IND(B)$                                                                               | concept-discernibility                 |
| $U/B$                            | Quotient set                  | Set of all equivalence classes of $IND(B)$                                                           | concept-indiscernibility               |
| $[u]_B$                          | Equivalence class             | Equivalence class of object $u$ under $IND(B)$                                                       | concept-indiscernibility               |
| $[u]_B^V$                        | Equivalence class on subset   | Equivalence class of object $u \in V$ under $IND_V(B)$                                               | concept-indiscernibility               |
| $E$                              | Equivalence class             | Generic notation for an equivalence class in $U/B$                                                   | concept-indiscernibility               |
| $\underline{X}_B$                | Lower approximation           | Lower approximation of $X \subseteq U$ induced by $B$                                                | concept-approximations                 |
| $\overline{X}_B$                 | Upper approximation           | Upper approximation of $X \subseteq U$ induced by $B$                                                | concept-approximations                 |
| $POS(B)$                         | Positive region               | Subset of $U$ uniquely classifiable to decision classes using $B$                                    | concept-positive-region                |
| $POS_{\mathbb{A}}(B)$            | Positive region (indexed)     | Positive region with explicit decision table subscript                                               | concept-positive-region                |
| $\gamma(B)$                      | Degree of dependency          | $\gamma(B) = \lvert POS(B) \rvert / \lvert U \rvert$                                                 | concept-positive-region                |
| $\tau$                           | Boolean formula               | Discernibility Boolean formula for reduct computation                                                | concept-decision-reduct                |
| $\lambda$                        | Boolean special variable      | Auxiliary propositional variable in Boolean proofs                                                   | concept-decision-bireduct              |
| $X^{\langle k \rangle}$          | Decision class                | Objects with decision value $k$                                                                      | concept-decision-table                 |
| $X^{\langle d = v \rangle}$      | Decision class (value)        | Objects with decision value $v \in V_d$                                                              | concept-decision-table                 |
| $\mathscr{F}(B, V)$              | Formulae set                  | Set of propositional formulae over attributes $B$ and values $V$                                     | concept-formulae                       |
| $\varphi \Rightarrow \psi$       | Decision rule                 | A decision rule with predecessor $\varphi$ and successor $\psi$                                      | concept-decision-rule                  |
| $\|\varphi\|_{\mathbb{A}}$       | Meaning of $\varphi$          | Set of objects in $\mathbb{A}$ satisfying formula $\varphi$                                          | concept-formulae                       |
| $\varepsilon$                    | Epsilon threshold             | Approximation threshold, $\varepsilon \in [0, 1)$                                                    | concept-approximate-decision-reduct    |
| $M(B)$                           | Majority function             | Accuracy of rule-based classifier using most frequent decision in each $B$-induced equivalence class | concept-majority-function              |
| $R(B)$                           | Relative gain function        | Bayesian measure of decision information induced by $B$                                              | concept-relative-gain-function         |
| $disc_{\mathbb{A}}(B)$           | Discernibility measure        | Number of object pairs with different decisions discerned by $B$                                     | concept-discernibility-measure         |
| $disc\_ratio(B)$                 | Discernibility ratio          | Normalized discernibility measure, $disc\_ratio: 2^A \to [0, 1]$                                     | concept-discernibility-measure         |
| $X$                              | Bireduct objects              | Subset of objects covered by a decision bireduct                                                     | concept-decision-bireduct              |
| $Y$                              | Bireduct objects (alternate)  | Another subset of objects for comparison                                                             | concept-decision-bireduct              |
| $(X, B)$                         | Decision bireduct             | A decision bireduct pair of object subset $X$ and attribute subset $B$                               | concept-decision-bireduct              |
| $B \Rrightarrow_{X} d$           | Inexact functional dependency | Attribute subset $B$ determines decision $d$ within objects $X$                                      | concept-decision-bireduct              |
| $B \Rrightarrow^{\gamma}_{X} d$  | Gamma functional dependency   | $B$ gamma-determines $d$ within $X$ (discernibility against whole $U$)                               | concept-gamma-decision-bireduct        |
| $\mathcal{B}$                    | Bireduct ensemble             | An ensemble of decision bireducts $\{(X_1, B_1), \ldots, (X_m, B_m)\}$                               | concept-bireduct-ensemble              |
| $\mathcal{C}$                    | Bireduct ensemble (alternate) | Another ensemble for comparison                                                                      | concept-bireduct-ensemble              |
| $\prec$                          | Simpler ensemble order        | Linear order over ensembles by sorted attribute cardinalities                                        | concept-bireduct-ensemble-optimization |
| $\mathbb{A}^{\boxbslash}$        | Diagonal decision table       | Decision table with diagonal auxiliary attributes for bireduct computation                           | concept-decision-bireduct              |
| $\mathbb{A}_{\diamond}$          | Sampled decision table        | Decision table obtained by sampling objects                                                          | concept-decision-bireduct              |
| $A^{\diamond}$                   | Sampled attributes            | Subset of attributes used in the sampling algorithm                                                   | prop-decision-bireduct-sampling        |
| $U^{\diamond}$                   | Sampled objects               | Set of representative objects in the sampling algorithm                                               | prop-decision-bireduct-sampling        |
| $\mathbb{G}$                     | Graph                         | A graph $(\mathbb{V}, \mathbb{E})$ used in NP-hardness proofs                                        | concept-np-hardness                    |
| $\mathbb{V}$                     | Vertices                      | Set of vertices in a graph                                                                           | concept-np-hardness                    |
| $\mathbb{E}$                     | Edges                         | Set of edges in a graph                                                                              | concept-np-hardness                    |
| $Cov_{\mathbb{G}}(W)$            | Covered vertices              | Vertices in $W$ or adjacent to $W$ in $\mathbb{G}$                                                   | concept-np-hardness                    |
| $first$                          | Temporal first index          | Start index of a temporal bireduct interval                                                          | concept-temporal-bireduct              |
| $last$                           | Temporal last index           | End index of a temporal bireduct interval                                                            | concept-temporal-bireduct              |
| $\mathcal{M}$                    | Classification model          | A trained classification model                                                                       | concept-classification-model           |
| $\hat{y}_M$                      | Predicted decision            | Decision value predicted by model $M$ for a given object                                             | concept-classification-model           |
| $top\_k$                         | Top-k                         | Top-k elements according to some ranking                                                             | concept-feature-importance             |
| $\mathbb{A}^{\circlearrowright}$ | Shuffled decision table       | Decision table with shuffled auxiliary attributes                                                    | concept-decision-bireduct              |
| $BirDesc(X, B)$                 | Bireduct description length   | Total descriptors in rules induced by bireduct $(X, B)$: $|X/B| \cdot (|B|+1)$                      | concept-bireduct-ensemble              |
| $cov_{\mathbb{A},\mathcal{B}}$  | Coverage count function       | Number of bireducts in ensemble $\mathcal{B}$ covering object $u$                                   | concept-bireduct-ensemble              |
| $EnsDesc(\mathcal{B})$          | Ensemble description length   | Sum of bireduct description lengths: $\sum_{(X_i,B_i) \in \mathcal{B}} BirDesc(X_i, B_i)$           | concept-bireduct-ensemble              |

---

## Source-to-KB Translation Notes

These notes document how the notation in `tmp/phd/thesis.tex` maps to the KB conventions
established above. The KB conventions are the authoritative ones.

### Direct Matches (no change)

The following thesis symbols map directly to KB symbols with identical semantics:
$U$, $A$, $d$, $V_a$, $\mathbb{A}$, $IND(B)$, $DIS(B)$, $U/B$, $[u]_B$,
$\underline{X}_B$, $\overline{X}_B$, $\gamma(B)$, $\tau$, $M(B)$, $R(B)$,
$disc_{\mathbb{A}}(B)$, $\varepsilon$, $\|\varphi\|_{\mathbb{A}}$.

### Translation Table

Thesis custom commands and their KB equivalents:

| Thesis Command              | Renders As                            | KB Symbol                             | Notes                                                                     |
| :-------------------------- | :------------------------------------ | :------------------------------------ | :------------------------------------------------------------------------ |
| `\dectabdef`                | $\mathbb{A} = (U, A \cup \{d\})$      | $\mathbb{A}$                          | KB uses $\mathbb{A}$ for the table; the full definition is in prose       |
| `\inattrs{a}`               | $a$                                   | $a$                                   | KB uses plain math mode; no wrapper needed                                |
| `\attrs{a}`                 | $\{a\}$                               | $\{a\}$                               | Same                                                                      |
| `\inobjs{u}`                | $u$                                   | $u$                                   | Same                                                                      |
| `\objs{u}`                  | $\{u\}$                               | $\{u\}$                               | Same                                                                      |
| `\objsidx{1,2}`             | $\llbracket 1, 2 \rrbracket$          | indexed notation                      | Use $u_1, u_2$ in KB prose; double brackets are thesis-specific shorthand |
| `\attrval{a}{u}`            | $a(u)$                                | $a(u)$                                | Same                                                                      |
| `\birobjects`               | $\mathpzc{X}$                         | $X$                                   | KB uses plain $X$ for bireduct objects (standard KaTeX-compatible)        |
| `\birobjectsY`              | $\mathpzc{Y}$                         | $Y$                                   | Same                                                                      |
| `\bireduct{X}{B}`           | $(X, B)$                              | $(X, B)$                              | Same                                                                      |
| `\decclass{k}`              | $X^{\langle k \rangle}$               | $X^{\langle k \rangle}$               | Same                                                                      |
| `\formulae{B}{V}`           | $\mathscr{F}(B, V)$                   | $\mathscr{F}(B, V)$                   | Same                                                                      |
| `\decrule{\varphi}{\psi}`   | $\varphi \Rightarrow \psi$            | $\varphi \Rightarrow \psi$            | Same                                                                      |
| `\pos{B}`                   | $POS(B)$                              | $POS(B)$                              | Same                                                                      |
| `\posdectab{\mathbb{A}}{B}` | $POS_{\mathbb{A}}(B)$                 | $POS_{\mathbb{A}}(B)$                 | Same                                                                      |
| `\boolformulasymbol`        | $\tau$                                | $\tau$                                | Same                                                                      |
| `\boolformulaspecial`       | $\lambda$                             | $\lambda$                             | Same                                                                      |
| `\funcdep{B}{X}{d}`         | $B \Rrightarrow_{X} d$                | $B \Rrightarrow_{X} d$                | Same                                                                      |
| `\funcdepgamma{B}{X}{d}`    | $B \Rrightarrow^{\gamma}_{X} d$       | $B \Rrightarrow^{\gamma}_{X} d$       | Same                                                                      |
| `\gammasymbol`              | $\gamma$                              | $\gamma$                              | Same                                                                      |
| `\eps`                      | $\varepsilon$                         | $\varepsilon$                         | Same                                                                      |
| `\dectabgammaspecial`       | $\circledast$                         | $\circledast$                         | KB uses $\circledast$ (not $*$ as in old KB)                              |
| `\dectabxb{X}{B}`           | $\mathbb{A}_X^B$                      | $\mathbb{A}_X^B$                      | Same                                                                      |
| `\dectabgamma{B}`           | $\mathbb{A}_B^\gamma$                 | $\mathbb{A}_B^\gamma$                 | Same                                                                      |
| `\dectabdiag`               | $\mathbb{A}^{\boxbslash}$             | $\mathbb{A}^{\boxbslash}$             | Same                                                                      |
| `\dectabsample`             | $\mathbb{A}_{\diamond}$               | $\mathbb{A}_{\diamond}$               | Same                                                                      |
| `\dectabshuffled`           | $\mathbb{A}^{\circlearrowright}$      | $\mathbb{A}^{\circlearrowright}$      | Same                                                                      |
| `\ensembleb`                | $\mathpzc{B}$                         | $\mathcal{B}$                         | KB uses $\mathcal{B}$ (standard KaTeX-compatible)                         |
| `\ensemblec`                | $\mathpzc{C}$                         | $\mathcal{C}$                         | Same                                                                      |
| `\model`                    | $\mathpzc{M}$                         | $\mathcal{M}$                         | KB uses $\mathcal{M}$                                                     |
| `\equivclass`               | $E$                                   | $E$                                   | Same                                                                      |
| `\mmeasure`                 | $M$                                   | $M$                                   | Same                                                                      |
| `\rmeasure`                 | $R$                                   | $R$                                   | Same                                                                      |
| `\discratiomeasure`         | $disc\_ratio$                         | $disc\_ratio$                         | Same                                                                      |
| `\temporalfirst`            | $first$                               | $first$                               | Same                                                                      |
| `\temporallast`             | $last$                                | $last$                                | Same                                                                      |
| `\graphg`                   | $\mathbb{G}$                          | $\mathbb{G}$                          | Same                                                                      |
| `\graphv`                   | $\mathbb{V}$                          | $\mathbb{V}$                          | Same                                                                      |
| `\graphe`                   | $\mathbb{E}$                          | $\mathbb{E}$                          | Same                                                                      |
| `\domgraph{W}`              | $Cov_{\mathbb{G}}(W)$                 | $Cov_{\mathbb{G}}(W)$                 | Same                                                                      |
| `\optparam{p}`              | $\langle\!\langle p \rangle\!\rangle$ | $\langle\!\langle p \rangle\!\rangle$ | Same                                                                      |
| `\pred{y}{M}`               | $\hat{y}_M$                           | $\hat{y}_M$                           | Same                                                                      |

### Minor Divergences

- **Gamma special value**: old KB used $*$; thesis uses $\circledast$. KB adopts $\circledast$ as
  the primary convention.
- **Script fonts**: thesis uses `\mathpzc` (Zapf Chancery) for bireduct objects and ensembles.
  KB uses plain letters ($X$, $Y$) or standard `\mathcal` ($\mathcal{B}$, $\mathcal{M}$) for
  KaTeX/MathJax compatibility.
- **Positive region subscript**: old KB used $POS_B(d)$ with decision attribute in subscript.
  Thesis uses $POS(B)$ or $POS_{\mathbb{A}}(B)$. KB follows thesis convention (table as subscript
  when disambiguation is needed).
