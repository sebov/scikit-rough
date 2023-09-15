import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex

logger = logging.getLogger(__name__)


@log_start_end(logger)
def check_if_attr_better_than_shuffled(
    group_index: GroupIndex,
    attr_values: np.ndarray,
    attr_values_count: int,
    values: np.ndarray,
    values_count: int,
    allowed_randomness: float,
    probes_count: int,
    smoothing_parameter: float,
    fast: bool,
    chaos_fun: rght.ChaosMeasure,
    rng: np.random.Generator,
) -> bool:
    # for result to be True we need `attr_probe_score >= (1 - allowed_randomness)`
    #
    # where `attr_probe_score` is estimated using the Laplace smoothing
    # ```
    # attr_probe_score = (attr_is_better_count + smoothing_parameter) / (
    #     probes_count + smoothing_parameter * smoothing_dims
    # )
    #
    # attr_is_better_count = number of times attr is better than shuffled
    # ```
    #
    # the inequality can be transformed to the following
    #
    # ```
    # attr_is_better_count >= threshold
    #
    # threshold = (1 - allowed_randomness)
    #   * (probes_count + smoothing_parameter * smoothing_dims) - smoothing_parameter
    #
    # ```
    #
    # alternatively, as `attr_is_better_count = probe_count - attr_is_worse_equal_count`
    # we can transform the above to
    #
    # ```
    # probe_count - attr_is_worse_equal_count >= threshold
    # attr_is_worse_equal_count <= probe_count - threshold
    # ```
    #
    # and therefore we can say that result is False if
    # ```
    # attr_is_worse_equal_count > probe_count - threshold
    # ```
    #
    # therefore (early stopping), sometimes we can determine (even before the loop ends)
    # that the result is:
    # - True, if `CURRENT_attr_is_BETTER_count >= threshold`
    # - False, if `CURRENT_attr_is_WORSE_EQUAL_count > probe_count - threshold`

    result = True

    if smoothing_parameter < 0:
        raise ValueError("smoothing parameter cannot be less than zero")
    smoothing_dims = 2  # binomial distribution, i.e., better/worse
    threshold = (1 - allowed_randomness) * (
        probes_count + smoothing_parameter * smoothing_dims
    ) - smoothing_parameter

    attr_chaos_score = group_index.get_chaos_score_after_split(
        attr_values,
        attr_values_count,
        values,
        values_count,
        chaos_fun,
    )
    attr_values_shuffled: np.ndarray = np.array(attr_values)

    # let us prepare a function that shuffles `attr_values_shuffled`
    if fast:
        permutation = rng.permutation(len(attr_values_shuffled))

        def shuffle_values():
            nonlocal attr_values_shuffled
            attr_values_shuffled = attr_values_shuffled[permutation]

    else:

        def shuffle_values():
            rng.shuffle(attr_values_shuffled)

    iterations = 0
    current_attr_is_better_count = 0
    for _ in range(probes_count):
        iterations += 1
        shuffle_values()
        shuffled_chaos_score = group_index.get_chaos_score_after_split(
            attr_values_shuffled,
            attr_values_count,
            values,
            values_count,
            chaos_fun,
        )
        if attr_chaos_score < shuffled_chaos_score:
            current_attr_is_better_count += 1

        # early stopping - positive case
        if current_attr_is_better_count >= threshold:
            result = True
            break

        # early stopping - negative case
        # current_attrs_is_worse_equal_count
        #   == iterations - current_attr_is_better_count
        if iterations - current_attr_is_better_count > probes_count - threshold:
            result = False
            break

    logger.debug("smoothing_parameter == %f", smoothing_parameter)
    logger.debug("threshold == %f", threshold)
    logger.debug("probes_count == %d", probes_count)
    logger.debug("iterations == %d", iterations)
    logger.debug("current_attr_is_better_count == %d", current_attr_is_better_count)
    logger.debug("allowed_randomness == %f", allowed_randomness)

    return result
