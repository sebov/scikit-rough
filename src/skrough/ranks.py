from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy

from skrough.dataprep import DEFAULT_SHUFFLED_PREFIX

COMPARE_RANKS_COL_ATTR_TYPE = "attr_type"
COMPARE_RANKS_COL_TOP_K = "top_k"
COMPARE_RANKS_COL_AVG_RANK = "avg_rank"

ATTR_TYPE_VALUE_ORIGINAL = "original"
ATTR_TYPE_VALUE_SHUFFLED = "shuffled"
TOP_K_VALUE_ALL = "all"


@dataclass
class AttrRanks:
    original: np.ndarray
    shuffled: np.ndarray


def get_attr_ranks(
    scores, attr_col, score_col, shuffled_prefix=DEFAULT_SHUFFLED_PREFIX
) -> AttrRanks:
    scores = scores.sort_values([score_col], ascending=False).reset_index(drop=True)
    ranks = (
        pd.Series(np.arange(len(scores)) + 1)
        .groupby(scores[score_col])
        .transform(np.mean)
    )
    is_shuffled = scores[attr_col].str.startswith(shuffled_prefix)
    return AttrRanks(
        original=ranks[~is_shuffled].to_numpy(),
        shuffled=ranks[is_shuffled].to_numpy(),
    )


def compare_ranks(
    scores, attr_col, score_col, top_ks=None, shuffled_prefix=DEFAULT_SHUFFLED_PREFIX
):
    attr_ranks = get_attr_ranks(
        scores=scores,
        attr_col=attr_col,
        score_col=score_col,
        shuffled_prefix=shuffled_prefix,
    )

    result = [
        [ATTR_TYPE_VALUE_ORIGINAL, TOP_K_VALUE_ALL, attr_ranks.original.mean()],
        [ATTR_TYPE_VALUE_SHUFFLED, TOP_K_VALUE_ALL, attr_ranks.shuffled.mean()],
    ]

    if top_ks is not None:
        if isinstance(top_ks, int):
            top_ks = [top_ks]
        for top_k in top_ks:
            result.extend(
                [
                    [
                        ATTR_TYPE_VALUE_ORIGINAL,
                        str(top_k),
                        attr_ranks.original[:top_k].mean(),
                    ],
                    [
                        ATTR_TYPE_VALUE_SHUFFLED,
                        str(top_k),
                        attr_ranks.shuffled[:top_k].mean(),
                    ],
                ]
            )
    return pd.DataFrame(
        result,
        columns=[
            COMPARE_RANKS_COL_ATTR_TYPE,
            COMPARE_RANKS_COL_TOP_K,
            COMPARE_RANKS_COL_AVG_RANK,
        ],
    )


def compare_ranksum(
    scores, attr_col, score_col, shuffled_prefix=DEFAULT_SHUFFLED_PREFIX
):
    shuffled_indicator = scores[attr_col].str.contains(shuffled_prefix)
    scores_original = scores[~shuffled_indicator][score_col]
    scores_shuffled = scores[shuffled_indicator][score_col]
    return scipy.stats.ranksums(scores_original, scores_shuffled)
