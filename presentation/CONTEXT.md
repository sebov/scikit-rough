# Presentation: Approximate Reducts & GroupIndex

Workshop presentation on practical algorithms for approximate decision reducts,
with a focus on the GroupIndex data structure and its implementations.

## Language

**Decision table**:
A tuple (U, A ∪ {d}) where U is a universe of objects, A is a set of conditional
attributes, and d is the decision attribute.
_Avoid_: data table, dataset

**Equivalence class / group**:
A subset of objects that are pairwise indiscernible with respect to a given set of
attributes B ⊆ A. Represented in code as objects sharing the same group index value.
_Avoid_: cluster, bucket, partition (use only when referring to the set of all groups)

**Group index**:
An array mapping each object (by position) to an integer group identifier. Together with
n_groups, it compactly represents the partition of U induced by a set of attributes.
_Avoid_: partition vector, cluster labels

**GroupIndex** (capitalized):
The Python class (`skrough.structs.group_index.GroupIndex`) that holds the group index
and provides operations on it: split, compress, get_disorder_score.
_Avoid_: GI

**Disorder measure**:
A function that quantifies the impurity of decision values within groups. Lower values
mean purer groups (better alignment between attribute-induced groups and the decision).
Examples: entropy, gini impurity, conflicts count.
_Avoid_: cost function, loss, impurity (ambiguous -- impurity may mean Gini specifically)

**Disorder score**:
The weighted average of a disorder measure over all groups in a GroupIndex, weighted by
group size. This is what the algorithm minimizes.
_Avoid_: disorder, score (ambiguous alone)

**Split**:
The GroupIndex operation that refines an existing partition by incorporating a new
attribute. Computed as `new_index = old_index * values_count + new_values`.
_Avoid_: refine, subdivide

**Compress**:
The GroupIndex operation that renumbers group identifiers to be contiguous (0, 1, 2, ...)
after split has introduced gaps.
_Avoid_: renumber, compact

**Approximate decision reduct**:
A subset B ⊆ A such that disorder_score(B) ≤ threshold, where
threshold = total_disorder + ε · (base_disorder − total_disorder), and no proper subset
of B satisfies this condition.
_Avoid_: ε-reduct, approx-reduct

**Hook**:
A pluggable function in the ProcessingMultiStage pipeline that handles one specific
responsibility (stop conditions, candidate selection, element processing, etc.).
_Avoid_: callback, plugin, middleware

**Stage**:
A named collection of hooks that together implement one phase of the computation (e.g.,
growing the attribute set, reducing redundant attributes).
_Avoid_: phase, step

## Relationships

- A **Decision table** has one **Universe** (U), many **Conditional attributes** (A),
  and one **Decision attribute** (d)
- A set of attributes induces a partition of U into **Equivalence classes / groups**
- A **GroupIndex** compactly represents this partition as an array of group IDs
- **GroupIndex.split** refines a partition by one attribute; **GroupIndex.compress**
  cleans up after split
- A **Disorder measure** evaluates a single group; a **Disorder score** averages it
  over all groups in a **GroupIndex**
- The greedy algorithm builds an **Approximate decision reduct** by iteratively
  selecting attributes that maximally decrease the **Disorder score**
- A **Stage** is configured with **Hooks**; different hook configurations produce
  different algorithms (greedy, DAAR) from the same pipeline

## Flagged ambiguities

- "group" -- used both for equivalence class and for GroupIndex. Resolved: "group"
  alone means equivalence class; "GroupIndex" (capitalized) means the data structure.
- "index" -- used both for the array inside GroupIndex and for object position.
  Resolved: "group index" (lowercase, two words) = the array; "index" alone = object
  position unless context is clear.
