"""Process elements common hook functions."""

import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def process_elements_hook_random_choice(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    """Process elements hook returning a random sample from the input ``elements``.

    Process elements hook returning a random sample from the input ``elements``. The
    number of elements that should be drawn randomly is stored in
    :code:`state.config` under the ``candidates_count_config_key`` key. If the given
    key is not available in :code:`state.config` or is `None` then the number of
    elements to be drawn will fall back to the total number of elements. The value
    of the ``elements_count_config_key`` comes from the enclosing scope. The hook
    function uses :obj:`state.get_rng()` random generator to perform the random
    choice operation. If the number of elements to be drawn from the config is
    larger than the actual size of the input elements then the sample size is
    decreased to the size of the input.

    Args:
        state: An object representing processing state.
        elements: An input sequence of elements to be processed by the hook
            function.

    Returns:
        A random sample from the input ``elements``.
    """
    if state.is_set_config_candidates_select_random_max_count():
        candidates_count = state.get_config_candidates_select_random_max_count()
    else:
        logger.debug(
            "fallback to the total number of elements in the collection",
        )
        candidates_count = len(elements)
    candidates_attrs_count = min(len(elements), candidates_count)
    candidates = state.get_rng().choice(
        elements,
        size=candidates_attrs_count,
        replace=False,
    )
    logger.debug("candidates = %s", candidates)
    return candidates


@log_start_end(logger)
def process_elements_hook_pass_everything(
    state: ProcessingState,  # pylint: disable=unused-argument
    elements: rght.Elements,
) -> rght.Elements:
    """Process elements hook returning the original input ``elements`` without change.

    Args:
        state: An object representing processing state.
        elements: An input sequence of elements to be processed by the hook function.

    Returns:
        The original input ``elements``.
    """
    return elements


@log_start_end(logger)
def process_elements_hook_reverse_elements(
    state: ProcessingState,  # pylint: disable=unused-argument
    elements: rght.Elements,
) -> rght.Elements:
    """Process elements hook reversing the input ``elements``.

    Args:
        state: An object representing processing state.
        elements: An input sequence of elements to be processed by the hook function.

    Returns:
        The input ``elements`` in reverse order.
    """
    return np.asarray(list(reversed(elements)))
