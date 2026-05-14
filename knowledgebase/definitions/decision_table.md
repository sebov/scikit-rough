---
tags: [rst, core]
related: [notation_and_symbols.md]
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
| 7   | overcast | cool        | normal   | weak   | yes  |
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
