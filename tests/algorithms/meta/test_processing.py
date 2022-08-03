from unittest.mock import MagicMock

import pytest

from skrough.algorithms.meta.processing import ProcessingMultiStage
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "run_with_state",
    [False, True],
)
def test_processing_multi_stage(
    run_with_state,
    state_fixture: ProcessingState,
):

    prepare_result_fun = MagicMock()
    init_multi_stage_hook = MagicMock()
    init_hook = MagicMock()
    stage0 = MagicMock()
    stage1 = MagicMock()
    finalize_hook = MagicMock()

    processing = ProcessingMultiStage.from_hooks(
        prepare_result_fun=prepare_result_fun,
        init_multi_stage_hooks=init_multi_stage_hook,
        init_hooks=init_hook,
        stages=[stage0, stage1],
        finalize_hooks=finalize_hook,
    )
    assert processing.init_multi_stage_agg.normalized_hooks[0] == init_multi_stage_hook
    assert processing.init_agg.normalized_hooks[0] == init_hook
    assert processing.finalize_agg.normalized_hooks[0] == finalize_hook
    assert processing.stages == [stage0, stage1]
    assert processing.prepare_result_fun == prepare_result_fun

    processing(state=(state_fixture if run_with_state else None))

    # it will be called when the stage is not given
    assert init_multi_stage_hook.call_count == int(run_with_state is False)

    assert init_hook.call_count == 1
    assert stage0.call_count == 1
    assert stage1.call_count == 1
    assert finalize_hook.call_count == 1
    assert prepare_result_fun.call_count == 1
