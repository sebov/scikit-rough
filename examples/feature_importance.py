# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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

from skrough.chaos_measures import conflicts_count, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data
from skrough.dataprep import prepare_factorized_data
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
x, x_domain_sizes, y, y_domain_size = prepare_factorized_data(df, TARGET_COLUMN)
column_names = np.array([col for col in df.columns if col != TARGET_COLUMN])

print("Conditional data:")
print(x)
print()
print("Conditional data feature domain sizes:")
print(x_domain_sizes)
print()
print("Target data:")
print(y)
print()
print("Target data feature domain size:")
print(y_domain_size)

# %% [markdown]
# ## Measure of disorder in the dataset - chaos score
#
# In the context of the given dataset, a chaos score values is quantity that
# characterizes a subset of features and, more or less, presents the disorder of
# decisions in the equivalence classes induced by the subsets of features.
#
# In most cases it is reasonable to assume that the chaos score function is monotonic
# with respect to subset relation, i.e., for subsets of features $A \subseteq B$,
# the chaos score for $A$ should be less or equal to that for $B$.
#
# Attributes are given by their ordinal numbers.
#
# Let's try three standard approaches, i.e., `conflicts_count`, `gini_impurity` and
# `entropy`.

# %%
for chaos_function in [conflicts_count, entropy, gini_impurity]:
    print(chaos_function.__name__)
    for attrs in [[0], [0, 1], [0, 1, 3], [0, 1, 2, 3]]:
        print(
            f"chaos score for attrs {attrs}({column_names[attrs]}) = ",
            get_chaos_score_for_data(
                x=x,
                x_counts=x_domain_sizes,
                y=y,
                y_count=y_domain_size,
                chaos_fun=chaos_function,
                attrs=attrs,
            ),
        )
    print()
    print()

# %% [markdown]
# ## Assessing feature importance
#
# We can use the above chaos score functions for assessing the features, i.e.,
# we can observe the chaos score change if a given feature is removed.
#
# To follow a more realistic example, we can use an enseble of feature subsets, i.e.,
# a family of subsets of all atributes, and not just a single subset of features,
# computing the total or average chaos score change over several possible appearances
# of the attribute in the ensemble elements.

# %%
attr_subset_ensemble = [
    [[0, 2], [0, 3], [0], [2, 3], [1, 2, 3]],
    [[0], [0, 1], [1, 2]],
    [list(elem) for elem in more_itertools.powerset(range(4))],
]
for chaos_function in [conflicts_count, entropy, gini_impurity]:
    print(chaos_function.__name__)
    for attr_subset in attr_subset_ensemble:
        print("feature importance for attribute subset ensemble: ")
        pprint.pprint(attr_subset, compact=True)
        print(
            get_feature_importance(
                x,
                x_domain_sizes,
                y,
                y_domain_size,
                column_names,
                attr_subset,
                chaos_fun=chaos_function,
            )
        )
        print()
    print()
    print()
