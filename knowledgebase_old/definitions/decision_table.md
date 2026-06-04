---
tags: [rst, core]
related: [notation_and_symbols.md, definitions/consistency.md, definitions/indiscernibility.md, definitions/decision_rules.md, definitions/positive_region.md, definitions/reducts.md]
---
# Decision Table

A decision table $\mathbb{A} = (U, A \cup \{d\})$ is a pair of non-empty sets, where:

- $U$ is a universe of objects.
- $A \cup \{d\}$ is a set consisting of attributes such that every $a \in A \cup \{d\}$ is a
  function $a : U \rightarrow V_a$, where $V_a$ denotes $a$'s codomain and is called the value set
  of $a$.
- The distinguished attribute $d$, such that $d \notin A$, is called a decision attribute.
- The elements of $A$ are called conditional attributes.

## Example

Consider a decision table where $U = \{1, 2, \dots, 14\}$, $A = \{\text{Outlook, Temperature,
Humidity, Wind}\}$, and $d = \text{Play}$.

| ID  | Outlook  | Temperature | Humidity | Wind   | Play |
| --- | -------- | ----------- | -------- | ------ | ---- |
| 1   | sunny    | hot         | high     | weak   | no   |
| 2   | sunny    | hot         | high     | strong | no   |
| 3   | overcast | hot         | high     | weak   | yes  |
| 4   | rain     | mild        | high     | weak   | yes  |
| 5   | rain     | cool        | normal   | weak   | yes  |
| 6   | rain     | cool        | normal   | strong | no   |
| 7   | overcast | cool        | normal   | strong | yes  |
| 8   | sunny    | mild        | high     | weak   | no   |
| 9   | sunny    | cool        | normal   | weak   | yes  |
| 10  | rain     | mild        | normal   | weak   | yes  |
| 11  | sunny    | mild        | normal   | strong | yes  |
| 12  | overcast | mild        | high     | strong | yes  |
| 13  | overcast | hot         | normal   | weak   | yes  |
| 14  | rain     | mild        | high     | strong | no   |

## Decision Reducts and Rules

For this dataset there are two decision reducts:
$\{ \text{Outlook}, \text{Temperature}, \text{Wind} \}$ and
$\{ \text{Outlook}, \text{Humidity}, \text{Wind} \}$.

### Decision Rules from $\{ \text{Outlook}, \text{Temperature}, \text{Wind} \}$

|  No. | Rule                                                                                             |   Support   |
| ---: | :----------------------------------------------------------------------------------------------- | :---------: |
|    1 | $(O=\text{overcast}) \wedge (T=\text{cool}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$ |   $\{7\}$   |
|    2 | $(O=\text{overcast}) \wedge (T=\text{hot}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$    | $\{3, 13\}$ |
|    3 | $(O=\text{overcast}) \wedge (T=\text{mild}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$ |  $\{12\}$   |
|    4 | $(O=\text{rain}) \wedge (T=\text{cool}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$      |   $\{6\}$   |
|    5 | $(O=\text{rain}) \wedge (T=\text{cool}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$       |   $\{5\}$   |
|    6 | $(O=\text{rain}) \wedge (T=\text{mild}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$      |  $\{14\}$   |
|    7 | $(O=\text{rain}) \wedge (T=\text{mild}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$       | $\{4, 10\}$ |
|    8 | $(O=\text{sunny}) \wedge (T=\text{cool}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$      |   $\{9\}$   |
|    9 | $(O=\text{sunny}) \wedge (T=\text{hot}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$      |   $\{2\}$   |
|   10 | $(O=\text{sunny}) \wedge (T=\text{hot}) \wedge (W=\text{weak}) \Rightarrow (d=\text{no})$        |   $\{1\}$   |
|   11 | $(O=\text{sunny}) \wedge (T=\text{mild}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$    |  $\{11\}$   |
|   12 | $(O=\text{sunny}) \wedge (T=\text{mild}) \wedge (W=\text{weak}) \Rightarrow (d=\text{no})$       |   $\{8\}$   |

### Decision Rules from $\{ \text{Outlook}, \text{Humidity}, \text{Wind} \}$

|  No. | Rule                                                                                               |   Support   |
| ---: | :------------------------------------------------------------------------------------------------- | :---------: |
|    1 | $(O=\text{overcast}) \wedge (H=\text{high}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$   |  $\{12\}$   |
|    2 | $(O=\text{overcast}) \wedge (H=\text{high}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$     |   $\{3\}$   |
|    3 | $(O=\text{overcast}) \wedge (H=\text{normal}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$ |   $\{7\}$   |
|    4 | $(O=\text{overcast}) \wedge (H=\text{normal}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$   |  $\{13\}$   |
|    5 | $(O=\text{rain}) \wedge (H=\text{high}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$        |  $\{14\}$   |
|    6 | $(O=\text{rain}) \wedge (H=\text{high}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$         |   $\{4\}$   |
|    7 | $(O=\text{rain}) \wedge (H=\text{normal}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$      |   $\{6\}$   |
|    8 | $(O=\text{rain}) \wedge (H=\text{normal}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$       | $\{5, 10\}$ |
|    9 | $(O=\text{sunny}) \wedge (H=\text{high}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$       |   $\{2\}$   |
|   10 | $(O=\text{sunny}) \wedge (H=\text{high}) \wedge (W=\text{weak}) \Rightarrow (d=\text{no})$         | $\{1, 8\}$  |
|   11 | $(O=\text{sunny}) \wedge (H=\text{normal}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$    |  $\{11\}$   |
|   12 | $(O=\text{sunny}) \wedge (H=\text{normal}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$      |   $\{9\}$   |

All rules have confidence = 1 because the golf dataset is consistent.

## $\gamma$-Decision Reduct Example

For an inconsistent decision table (or a table restricted to a subset of attributes that makes it
inconsistent), the $\gamma$-decision reduct can be illustrated using the construction of the
modified decision table $\mathbb{A}_B^\gamma$. Since the full golf table is consistent, the case
$B = A$ is not interesting ($\gamma(A) = 1$). Instead, consider smaller attribute subsets.

**Table 1**: Modified decision tables $\mathbb{A}_B^\gamma$ for $B = \{\text{Outlook},
\text{Temperature}, \text{Humidity}\}$ and $B = \{\text{Temperature}, \text{Humidity},
\text{Wind}\}$.

For $B = \{\text{Outlook}, \text{Temperature}, \text{Humidity}\}$:

| ID | Outlook  | Temp. | Humidity | $d_B^\gamma$ |
| -- | -------- | ----- | -------- | :----------: |
| 1  | sunny    | hot   | high     | no           |
| 2  | sunny    | hot   | high     | no           |
| 3  | overcast | hot   | high     | yes          |
| 4  | rain     | mild  | high     | $*$          |
| 5  | rain     | cool  | normal   | $*$          |
| 6  | rain     | cool  | normal   | $*$          |
| 7  | overcast | cool  | normal   | yes          |
| 8  | sunny    | mild  | high     | no           |
| 9  | sunny    | cool  | normal   | yes          |
| 10 | rain     | mild  | normal   | yes          |
| 11 | sunny    | mild  | normal   | yes          |
| 12 | overcast | mild  | high     | yes          |
| 13 | overcast | hot   | normal   | yes          |
| 14 | rain     | mild  | high     | $*$          |

For $B = \{\text{Temperature}, \text{Humidity}, \text{Wind}\}$:

| ID | Temp. | Humidity | Wind   | $d_B^\gamma$ |
| -- | ----- | -------- | ------ | :----------: |
| 1  | hot   | high     | weak   | $*$          |
| 2  | hot   | high     | strong | no           |
| 3  | hot   | high     | weak   | $*$          |
| 4  | mild  | high     | weak   | $*$          |
| 5  | cool  | normal   | weak   | yes          |
| 6  | cool  | normal   | strong | $*$          |
| 7  | cool  | normal   | strong | $*$          |
| 8  | mild  | high     | weak   | $*$          |
| 9  | cool  | normal   | weak   | yes          |
| 10 | mild  | normal   | weak   | yes          |
| 11 | mild  | normal   | strong | yes          |
| 12 | mild  | high     | strong | $*$          |
| 13 | hot   | normal   | weak   | yes          |
| 14 | mild  | high     | strong | $*$          |

**Table 2**: Decision rules generated from the $\gamma$-decision reducts for the original decision
table limited to the two attribute subsets. In both cases the $\gamma$-decision reduct consists of
all attributes of the limited table (no further reduction is possible).

|                                                       No. | Rule                                                                                             |  Support   |
| --------------------------------------------------------: | :----------------------------------------------------------------------------------------------- | :--------: |
| $\{\text{Outlook}, \text{Temperature}, \text{Humidity}\}$ |                                                                                                  |            |
|                                                         1 | $(O=\text{overcast}) \wedge (T=\text{cool}) \wedge (H=\text{normal}) \Rightarrow (d=\text{yes})$ |  $\{7\}$   |
|                                                         2 | $(O=\text{overcast}) \wedge (T=\text{hot}) \wedge (H=\text{high}) \Rightarrow (d=\text{yes})$    |  $\{3\}$   |
|                                                         3 | $(O=\text{overcast}) \wedge (T=\text{hot}) \wedge (H=\text{normal}) \Rightarrow (d=\text{yes})$  |  $\{13\}$  |
|                                                         4 | $(O=\text{overcast}) \wedge (T=\text{mild}) \wedge (H=\text{high}) \Rightarrow (d=\text{yes})$   |  $\{12\}$  |
|                                                         5 | $(O=\text{rain}) \wedge (T=\text{mild}) \wedge (H=\text{normal}) \Rightarrow (d=\text{yes})$     |  $\{10\}$  |
|                                                         6 | $(O=\text{sunny}) \wedge (T=\text{cool}) \wedge (H=\text{normal}) \Rightarrow (d=\text{yes})$    |  $\{9\}$   |
|                                                         7 | $(O=\text{sunny}) \wedge (T=\text{hot}) \wedge (H=\text{high}) \Rightarrow (d=\text{no})$        | $\{1, 2\}$ |
|                                                         8 | $(O=\text{sunny}) \wedge (T=\text{mild}) \wedge (H=\text{high}) \Rightarrow (d=\text{no})$       |  $\{8\}$   |
|                                                         9 | $(O=\text{sunny}) \wedge (T=\text{mild}) \wedge (H=\text{normal}) \Rightarrow (d=\text{yes})$    |  $\{11\}$  |
|    $\{\text{Temperature}, \text{Humidity}, \text{Wind}\}$ |                                                                                                  |            |
|                                                         1 | $(T=\text{cool}) \wedge (H=\text{normal}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$     | $\{5, 9\}$ |
|                                                         2 | $(T=\text{hot}) \wedge (H=\text{high}) \wedge (W=\text{strong}) \Rightarrow (d=\text{no})$       |  $\{2\}$   |
|                                                         3 | $(T=\text{hot}) \wedge (H=\text{normal}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$      |  $\{13\}$  |
|                                                         4 | $(T=\text{mild}) \wedge (H=\text{normal}) \wedge (W=\text{strong}) \Rightarrow (d=\text{yes})$   |  $\{11\}$  |
|                                                         5 | $(T=\text{mild}) \wedge (H=\text{normal}) \wedge (W=\text{weak}) \Rightarrow (d=\text{yes})$     |  $\{10\}$  |

All rules have confidence $= 1$ because they are generated from objects belonging to the positive
region.
