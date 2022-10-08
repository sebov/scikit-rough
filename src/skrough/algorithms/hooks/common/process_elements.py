import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def create_process_elements_hook_random_choice(config_key: str):

    logger.debug(
        "create process_elements_hook_random_choice hook with config_key = %s",
        config_key,
    )

    @log_start_end(logger)
    def process_elements_hook_random_choice(
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        candidates_count = state.config.get(config_key)
        if candidates_count is None:
            candidates_count = len(elements)
        candidates_attrs_count = min(len(elements), candidates_count)
        candidates = state.rng.choice(
            elements,
            size=candidates_attrs_count,
            replace=False,
        )
        logger.debug("candidates = %s", candidates)
        return candidates

    return process_elements_hook_random_choice


@log_start_end(logger)
def process_elements_hook_pass_everything(
    state: ProcessingState,  # pylint: disable=unused-argument
    elements: rght.Elements,
) -> rght.Elements:
    """Process elements hook returning the original input ``elements`` without change.

    Args:
        state: An object representing processing state.

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

    Returns:
        The input ``elements`` in reverse order.
    """
    return np.asarray(list(reversed(elements)))
