---
tags: [rst, core]
related: [notation_and_symbols.md]
---
# Decision Table

A decision table $\mathbb{A} = (U, A \cup \{d\})$ is a pair of non-empty sets, where:
- $U$ is a universe of objects.
- $A \cup \{d\}$ is a set consisting of attributes such that every $a \in A \cup \{d\}$ is a function $a : U \rightarrow V_a$, where $V_a$ denotes $a$'s codomain and is called the value set of $a$.
- The distinguished attribute $d$, such that $d \notin A$, is called a decision attribute.
- The elements of $A$ are called conditional attributes.

## Example

Consider a decision table where $U = \{1, 2, \dots, 14\}$, $A = \{\text{Outlook, Temperature, Humidity, Wind}\}$, and $d = \text{Play}$.

| ID | Outlook | Temperature | Humidity | Wind | Play |
|---|---|---|---|---|---|
| 1 | sunny | hot | high | weak | no |
| 2 | sunny | hot | high | strong | no |
| 3 | overcast | hot | high | weak | yes |
| 4 | rain | mild | high | weak | yes |
| 5 | rain | cool | normal | weak | yes |
| 6 | rain | cool | normal | strong | no |
| 7 | overcast | cool | normal | weak | yes |
| 8 | sunny | mild | high | weak | no |
| 9 | sunny | cool | normal | weak | yes |
| 10 | rain | mild | normal | weak | yes |
| 11 | sunny | mild | normal | strong | yes |
| 12 | overcast | mild | high | strong | yes |
| 13 | overcast | hot | normal | weak | yes |
| 14 | rain | mild | high | strong | no |
