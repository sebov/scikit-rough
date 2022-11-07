# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: 'Python 3.9.12 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rough Set check functions


# %%
import numpy as np
import pandas as pd

from skrough.checks import (
    check_if_consistent_table,
    check_if_functional_dependency,
    check_if_reduct,
)
from skrough.dataprep import prepare_factorized_data

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
x, x_domain_sizes, y, y_domain_size = prepare_factorized_data(df, TARGET_COLUMN)

# %% [markdown]
# ## Data table consistency
#
# Let's check if the data table is consistent:
#
# - check whole table
# - check using a given subset of attributes

# %%
check_if_consistent_table(x, y)

# %%
# check using only first two columns
check_if_consistent_table(x[:, 0:2], y)

# %% [markdown]
# ## Check functional dependency

# %%
# check functional dependency on all objects (using default: `None`) and all attrs
# (using default: `None`)
check_if_functional_dependency(x, y)

# %%
# check on all objects (using default: `None`) and on attrs `0, 2, 3`
check_if_functional_dependency(x, y, attrs=[0, 2, 3])

# %%
# check on all objects (using default: `None`) and on attrs `0, 1`
check_if_functional_dependency(x, y, attrs=[0, 1])

# %%
# check on objects `0, 2, 5` and on attrs `0, 1`
check_if_functional_dependency(x, y, objs=[0, 2, 5], attrs=[0, 1])

# %% [markdown]
# ## Check reducts
#
# For "Play Golf Dataset" there are only two reducts:
#
# - "Outlook", "Temperature", "Humidity" - `attrs == [0, 1, 2]`
# - "Outlook", "Humidity", "Wind" - `attrs == [0, 2, 3]`

# %%
check_if_reduct(x, y, attrs=[0, 2, 3])

# %%
check_if_reduct(x, y, attrs=[0, 2, 3])

# %%
# too few attributes ~ no functional dependency
check_if_reduct(x, y, attrs=[0, 1])

# %%
# too many attributes ~ some of them can be removed
check_if_reduct(x, y, attrs=[0, 1, 2, 3])
