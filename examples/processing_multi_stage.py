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
# # MultiStage processing

# %%
import pprint
from dataclasses import asdict

import numpy as np
import pandas as pd

from skrough.algorithms import hooks
from skrough.algorithms.hooks.names import (
    CONFIG_CHAOS_FUN,
    CONFIG_EPSILON,
    CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT,
    INPUT_X,
    INPUT_Y,
)
from skrough.algorithms.meta import describe, processing, stage
from skrough.chaos_measures import entropy
from skrough.checks import check_if_approx_reduct
from skrough.dataprep import prepare_factorized_data
from skrough.structs.attrs_subset import AttrsSubset

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
x, x_counts, y, y_count = prepare_factorized_data(df, TARGET_COLUMN)

# %% [markdown]
# ## Approximate decision superreduct
#
# Let's prepare a processing procedure to search for approximate decision superreduct.
# Notice that despite of the `ProcessingMultiStage` name, we create the processing with
# only one stage, cf., the below `grow_stage`.
#
# A greedy heuristic algorithm is implemented in the below example. Its brief
# description is as follows:
#
# * initialization steps:
#     * factorize the input data
#     * initialize internal structures - group index and result subset of attributes
#     * compute the approximation threshold, based on the data and the input
#       approximation level $\varepsilon$
# * perform processing defined in stages (here just one processing stage):
#   * grow_stage:
#     * define stop criterion - reaching the approximation threshold
#     * iteratively, until stop criterion
#       * use all remaining attrs as pre-candidates
#       * pass all pre-candidates as candidates
#       * use greedy heuristic to choose the best attribute - maximizing the chaos score
#         gain
#       * update internal structures
# * finalize the processing - prepare the actual return value

# %%
grow_stage = stage.Stage.from_hooks(
    stop_hooks=[
        hooks.stop_hooks.stop_hook_approx_threshold,
    ],
    init_hooks=None,
    pre_candidates_hooks=[
        hooks.pre_candidates_hooks.pre_candidates_hook_remaining_attrs,
    ],
    candidates_hooks=[
        hooks.common.process_elements.process_elements_hook_pass_everything,
    ],
    select_hooks=[
        hooks.select_hooks.select_hook_attrs_chaos_score_based,
    ],
    filter_hooks=None,
    inner_init_hooks=None,
    inner_stop_hooks=hooks.inner_stop_hooks.inner_stop_hook_empty,
    inner_process_hooks=hooks.inner_process_hooks.inner_process_hook_add_first_attr,
    finalize_hooks=None,
)

get_approx_reduct = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_factorize_data_x_y,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_approx_threshold,
    ],
    stages=[grow_stage],
    finalize_hooks=None,
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_attrs_subset,
)

# %% [markdown]
# ## Processing procedure inspection
#
# There are ways to inspect the prepared processing procedures, either for checking or
# debugging.
#
# A structured representation can be obtained and further processed:

# %%
description_graph = describe.describe(get_approx_reduct)
print(pprint.pformat(asdict(description_graph))[:1500], "...")

# %% [markdown]
# A visual representation using the sklearn framework:

# %%
get_approx_reduct

# %% [markdown]
# ## Invoke the prepared procedure
#
# Invoke the prepared procedure passing the input data and appropriate config values.

# %%
eps = 0.4
chaos_measure = entropy
result: AttrsSubset = get_approx_reduct(
    input_data={
        INPUT_X: x,
        INPUT_Y: y,
    },
    config={
        CONFIG_CHAOS_FUN: chaos_measure,
        CONFIG_EPSILON: eps,
        CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT: 1,
    },
)
result

# %%
check_if_approx_reduct(
    x, x_counts, y, y_count, attrs=result.attrs, chaos_fun=chaos_measure, epsilon=eps
)
