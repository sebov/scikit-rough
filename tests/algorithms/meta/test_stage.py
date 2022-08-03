from unittest.mock import MagicMock

import pytest

from skrough.algorithms.meta.stage import Stage
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "outer_iters",
    range(4),
)
@pytest.mark.parametrize(
    "inner_iters",
    range(4),
)
def test_stage_outer_inner_loop_k_m_iters(
    outer_iters,
    inner_iters,
    state_fixture: ProcessingState,
):
    # we want to test `outer_iters` outer loops iteration with `inner_iters` inner loop
    # iterations each
    #
    # let's prepare stop values
    # outer stop values:
    #   * 1 x check at the beginning
    #   * generally check every time in the inner iteration after inner_process_agg for
    #     early stopping = `outer_iters` * (generally `inner_iters`, but at least once
    #     per outer loop iteration)
    #   * all but the last value should be False
    # inner stop values:
    #   * for each inner loop it should have `inner_iters + 1` return values - all but
    #     the last one should be False
    #
    # e.g. `outer_iters=2, inner_iters=3`
    # outer_stop_values = [F(start), <F, F, F>(first inner), <F, F, T> (second inner)]
    # inner_stop_values = [<F, F, F, T>(first inner), <F, F, F, T>(second inner)]
    #
    # e.g. `outer_iters=2, inner_iters=0`
    # outer_stop_values = [F(start), <F>(first inner), <T> (second inner)]
    # inner_stop_values = [<T>(first inner), <T>(second inner)]
    #
    # if `inner_iters == 0` then the number of call of inner stop hook should be equal
    #   to `len(inner_stop_values)`
    # if `inner_iters > 0` the last `True` of inner_stop_values will not be used as
    #   outer stop check will finish the whole process earlier
    # if `outer_iters == 0` then inner stop should not be called at all

    outer_stop_values = [False] * outer_iters * max(1, inner_iters) + [True]
    expected_outer_stop_call_count = len(outer_stop_values)
    inner_stop_values = ([False] * inner_iters + [True]) * outer_iters
    if outer_iters == 0:
        expected_inner_stop_call_count = 0
    else:
        expected_inner_stop_call_count = len(inner_stop_values) - int(inner_iters > 0)

    stop_hook = MagicMock(side_effect=outer_stop_values)
    inner_stop_hook = MagicMock(side_effect=inner_stop_values)
    init_hook = MagicMock()
    pre_candidates_hook = MagicMock()
    candidates_hook = MagicMock()
    select_hook = MagicMock()
    filter_hook = MagicMock()
    inner_init_hook = MagicMock()
    inner_process_hook = MagicMock()
    finalize_hook = MagicMock()

    stage = Stage.from_hooks(
        stop_hooks=stop_hook,
        init_hooks=init_hook,
        pre_candidates_hooks=pre_candidates_hook,
        candidates_hooks=candidates_hook,
        select_hooks=select_hook,
        filter_hooks=filter_hook,
        inner_init_hooks=inner_init_hook,
        inner_stop_hooks=inner_stop_hook,
        inner_process_hooks=inner_process_hook,
        finalize_hooks=finalize_hook,
    )
    assert stage.stop_agg.normalized_hooks[0] == stop_hook
    assert stage.init_agg.normalized_hooks[0] == init_hook
    assert stage.pre_candidates_agg.normalized_hooks[0] == pre_candidates_hook
    assert stage.candidates_agg.normalized_hooks[0] == candidates_hook
    assert stage.select_agg.normalized_hooks[0] == select_hook
    assert stage.filter_agg.normalized_hooks[0] == filter_hook
    assert stage.inner_init_agg.normalized_hooks[0] == inner_init_hook
    assert stage.inner_stop_agg.normalized_hooks[0] == inner_stop_hook
    assert stage.inner_process_agg.normalized_hooks[0] == inner_process_hook
    assert stage.finalize_agg.normalized_hooks[0] == finalize_hook

    stage(state=state_fixture)

    assert stop_hook.call_count == expected_outer_stop_call_count
    assert inner_stop_hook.call_count == expected_inner_stop_call_count

    assert init_hook.call_count == 1
    assert pre_candidates_hook.call_count == outer_iters
    assert candidates_hook.call_count == outer_iters
    assert select_hook.call_count == outer_iters
    assert filter_hook.call_count == outer_iters
    assert inner_init_hook.call_count == outer_iters
    assert inner_process_hook.call_count == outer_iters * inner_iters
    assert finalize_hook.call_count == 1
