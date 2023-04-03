from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy

from skrough.dataprep import DEFAULT_SHUFFLED_PREFIX

ATTR_COL = "attr"
RANK_COL = "rank"
IS_SHUFFLED_COL = "is_shuffled"

ATTR_TYPE_VALUE_ORIGINAL = "original"
ATTR_TYPE_VALUE_SHUFFLED = "shuffled"
TOP_K_VALUE_ALL = "all"


@dataclass
class AttrTopKAvgRank:
    attr_type: Literal["original", "shuffled"]
    top_k: str
    avg_rank: float


def compare_ranks(
    scores, attr_col, score_col, top_ks=None, shuffled_prefix=DEFAULT_SHUFFLED_PREFIX
):
    scores = scores.sort_values([score_col], ascending=False)
    ranks = pd.DataFrame(
        {
            ATTR_COL: scores[attr_col],
            RANK_COL: np.arange(len(scores)) + 1,
            IS_SHUFFLED_COL: scores[attr_col].str.contains(shuffled_prefix),
        }
    )
    ranks[RANK_COL] = ranks[RANK_COL].groupby(scores[score_col]).transform("mean")
    ranks.reset_index(inplace=True, drop=True)
    result = [
        AttrTopKAvgRank(
            attr_type=ATTR_TYPE_VALUE_ORIGINAL,
            top_k=TOP_K_VALUE_ALL,
            avg_rank=ranks[~ranks[IS_SHUFFLED_COL]][RANK_COL].mean(),
        ),
        AttrTopKAvgRank(
            attr_type=ATTR_TYPE_VALUE_SHUFFLED,
            top_k=TOP_K_VALUE_ALL,
            avg_rank=ranks[ranks[IS_SHUFFLED_COL]][RANK_COL].mean(),
        ),
    ]
    if top_ks is not None:
        if isinstance(top_ks, int):
            top_ks = [top_ks]
        for top_k in top_ks:
            result.append(
                AttrTopKAvgRank(
                    attr_type=ATTR_TYPE_VALUE_ORIGINAL,
                    top_k=str(top_k),
                    avg_rank=ranks[~ranks[IS_SHUFFLED_COL]]
                    .iloc[:top_k][RANK_COL]
                    .mean(),
                )
            )
            result.append(
                AttrTopKAvgRank(
                    attr_type=ATTR_TYPE_VALUE_SHUFFLED,
                    top_k=str(top_k),
                    avg_rank=ranks[ranks[IS_SHUFFLED_COL]]
                    .iloc[:top_k][RANK_COL]
                    .mean(),
                )
            )
    return pd.DataFrame(result)


def compare_ranksum(
    scores, attr_col, score_col, shuffled_prefix=DEFAULT_SHUFFLED_PREFIX
):
    shuffled_indicator = scores[attr_col].str.contains(shuffled_prefix)
    scores_original = scores[~shuffled_indicator][score_col]
    scores_shuffled = scores[shuffled_indicator][score_col]
    return scipy.stats.ranksums(scores_original, scores_shuffled)
