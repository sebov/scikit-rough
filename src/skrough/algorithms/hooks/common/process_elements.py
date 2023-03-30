"""Process elements common hook functions."""

import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def create_process_elements_hook_random_choice(elements_count_config_key: str):
    """Create ``process_elements_hook_random_choice`` hook function.

    Create ``process_elements_hook_random_choice`` hook function as a closure function
    with ``candidates_count_config_key`` set that store the config key from
    :code:`state.config` that should be used to determine the number of elements that
    should be drawn on function invoke. If the given key is not available in
    :code:`state.config` or is `None` then the number of elements to be draw will
    fallback to the total number of elements.

    Args:
        elements_count_config_key: A name of the key from :code:`state.config` that will
            be used to determine the number of elements to be drawn on the returned
            function call.

    Returns:
        ``process_elements_hook_random_choice`` hook function that randomly sample input
        ``elements`` according to the ``elements_count_config_key`` setting.
    """

    logger.debug(
        "create process_elements_hook_random_choice hook with config_key = %s",
        elements_count_config_key,
    )

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
        function uses :obj:`state.rng` random generator to perform the random choice
        operation. If the number of elements to be drawn from the config is larger than
        the actual size of the input elements then the sample size is decreased to the
        size of the input.

        Args:
            state: An object representing processing state.
            elements: An input sequence of elements to be processed by the hook
                function.

        Returns:
            A random sample from the input ``elements``.
        """
        candidates_count = state.config.get(elements_count_config_key)
        if candidates_count is None:
            logger.debug(
                "config `%s` value not available - fallback to the total number of "
                "elements in the collection",
                elements_count_config_key,
            )
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
