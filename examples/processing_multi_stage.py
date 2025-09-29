# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-Stage processing

# %%
from dataclasses import asdict
import pprint

import numpy as np
import pandas as pd

from skrough.algorithms import hooks
from skrough.algorithms.meta import describe, processing, stage
from skrough.checks import check_if_approx_reduct
from skrough.dataprep import prepare_factorized_data
from skrough.disorder_measures import entropy
from skrough.structs.attrs_subset import AttrsSubset
from skrough.structs.state import ProcessingState


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
#       * use greedy heuristic to choose the best attribute - maximizing the disorder
#         score gain
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
        hooks.select_hooks.select_hook_attrs_disorder_score_based,
    ],
    filter_hooks=None,
    inner_init_hooks=None,
    inner_stop_hooks=[hooks.inner_stop_hooks.inner_stop_hook_empty],
    inner_process_hooks=[hooks.inner_process_hooks.inner_process_hook_add_first_attr],
    finalize_hooks=None,
)

get_approx_reduct = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_factorize_data_x_y,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_epsilon_approx_threshold,
    ],
    stages=[grow_stage],
    finalize_hooks=None,
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_attrs_subset,
)

# %% [markdown]
# ## Processing procedure inspection
#
# There are ways to inspect the prepared processing procedures, either for checking or
# debugging purposes.
#
# A structured representation can be obtained and further processed:

# %%
description_graph = describe.describe(get_approx_reduct)
print(pprint.pformat(asdict(description_graph))[:1500], "...")

# %% [markdown]
# ## Invoke the prepared procedure

# %% [markdown]
# Invoke the prepared procedure (processing element) and get the result.

# %%
eps = 0.4
disorder_measure = entropy

state = ProcessingState.from_optional(
    processing_fun=None,
)
state.set_input_data_x(x)
state.set_input_data_y(y)
state.set_config_disorder_fun(disorder_measure)
state.set_config_epsilon(eps)
state.set_config_select_attrs_disorder_score_based_max_count(1)

result: AttrsSubset = get_approx_reduct(state=state)
result

# %% [markdown]
# Check if the obtained result is a decision approximate superreduct - as we expected
# that designing the computing procedure appropriately.

# %%
check_if_approx_reduct(
    x,
    x_counts,
    y,
    y_count,
    attrs=result.attrs,
    disorder_fun=disorder_measure,
    epsilon=eps,
    check_attrs_reduction=False,
)
