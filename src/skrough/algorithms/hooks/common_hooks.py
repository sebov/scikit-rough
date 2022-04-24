import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def common_hook_pass_everything(
    state: GrowShrinkState,
    elements: rght.Elements,
) -> rght.Elements:
    return elements


@log_start_end(logger)
def common_hook_reverse_elements(
    state: GrowShrinkState,
    elements: rght.Elements,
) -> rght.Elements:
    return np.asarray(list(reversed(elements)))
