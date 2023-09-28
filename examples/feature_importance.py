# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature importance example

# %%
import pprint

import more_itertools
import numpy as np
import pandas as pd

from skrough.dataprep import prepare_factorized_data
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.disorder_score import get_disorder_score_for_data
from skrough.feature_importance import get_feature_importance

# %% [markdown]
# ## Dataset
#
# Let's prepare a sample data set - "Play Golf Dataset".

# %%
df = pd.DataFrame(
    np.array(
        [
            ["sunny", "hot", "high", "weak", "no"],
            ["sunny", "hot", "high", "strong", "no"],
            ["overcast", "hot", "high", "weak", "yes"],
            ["rain", "mild", "high", "weak", "yes"],
            ["rain", "cool", "normal", "weak", "yes"],
            ["rain", "cool", "normal", "strong", "no"],
            ["overcast", "cool", "normal", "strong", "yes"],
            ["sunny", "mild", "high", "weak", "no"],
            ["sunny", "cool", "normal", "weak", "yes"],
            ["rain", "mild", "normal", "weak", "yes"],
            ["sunny", "mild", "normal", "strong", "yes"],
            ["overcast", "mild", "high", "strong", "yes"],
            ["overcast", "hot", "normal", "weak", "yes"],
            ["rain", "mild", "high", "strong", "no"],
        ],
        dtype=object,
    ),
    columns=["Outlook", "Temperature", "Humidity", "Wind", "Play"],
)
TARGET_COLUMN = "Play"
df

# %% [markdown]
# ## Prepare data
#
# Factorize dataset and obtain the sizes of feature domains.

# %%
x, x_counts, y, y_count = prepare_factorized_data(df, TARGET_COLUMN)
column_names = np.array([col for col in df.columns if col != TARGET_COLUMN])

print("Conditional data:")
print(x)
print()
print("Conditional data feature domain sizes:")
print(x_counts)
print()
print("Target data:")
print(y)
print()
print("Target data feature domain size:")
print(y_count)

# %% [markdown]
# ## Measure of disorder in the dataset - disorder score
#
# In the context of the given dataset, a disorder score values is quantity that
# characterizes a subset of features and, more or less, presents the disorder of
# decisions in the equivalence classes induced by the subsets of features.
#
# In most cases it is reasonable to assume that the disorder score function is monotonic
# with respect to subset relation, i.e., for subsets of features $A \subseteq B$,
# the disorder score for $A$ should be less or equal to that for $B$.
#
# Attributes are given by their ordinal numbers.
#
# Let's try three standard approaches, i.e., `conflicts_count`, `gini_impurity` and
# `entropy`.

# %%
for disorder_function in [conflicts_count, entropy, gini_impurity]:
    print(disorder_function.__name__)
    for attrs in [[0], [0, 1], [0, 1, 3], [0, 1, 2, 3]]:
        print(
            f"disorder score for attrs {attrs}({column_names[attrs]}) = ",
            get_disorder_score_for_data(
                x=x,
                x_counts=x_counts,
                y=y,
                y_count=y_count,
                disorder_fun=disorder_function,
                attrs=attrs,
            ),
        )
    print()
    print()

# %% [markdown]
# ## Assessing feature importance
#
# We can use the above disorder score functions for assessing the features, i.e.,
# we can observe the disorder score change if a given feature is removed.
#
# To follow a more realistic example, we can use an enseble of feature subsets, i.e.,
# a family of subsets of all atributes, and not just a single subset of features,
# computing the total or average disorder score change over several possible appearances
# of the attribute in the ensemble elements.

# %%
attr_subset_ensemble = [
    [[0, 2], [0, 3], [0], [2, 3], [1, 2, 3]],
    [[0], [0, 1], [1, 2]],
    [list(elem) for elem in more_itertools.powerset(range(4))],
]
for disorder_function in [conflicts_count, entropy, gini_impurity]:
    print(disorder_function.__name__)
    for attr_subset in attr_subset_ensemble:
        print("feature importance for attribute subset ensemble: ")
        pprint.pprint(attr_subset, compact=True)
        print(
            get_feature_importance(
                x,
                x_counts,
                y,
                y_count,
                column_names,
                attr_subset,
                disorder_fun=disorder_function,
            )
        )
        print()
    print()
    print()
