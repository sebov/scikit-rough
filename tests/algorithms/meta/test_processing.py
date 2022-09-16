import copy
from unittest.mock import MagicMock

import pytest

from skrough.algorithms.meta.processing import ProcessingMultiStage
from skrough.structs.state import ProcessingState
from tests.algorithms.meta.helpers import (
    DUMMY_NODE,
    DUMMY_NODE_NAME,
    LEAF_VALUE,
    get_describe_dict,
)


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


DESCRIBE_PREPARE_RESULT_NODE_NAME = "prepare_result"


def test_describe():
    mock = MagicMock()
    mock.describe.return_value = DUMMY_NODE

    # separate mock for prepare_result is needed as otherwise node_name override on it
    # would cause changes also for other values returned by mock.describe (as it is the
    # same DUMMY_NODE object)
    # for other it is not an issue because they are wrapped over by aggregators and
    # node_name override is done rather on these wrapper objects than on the DUMMY_NODE
    mock_prepare_result = MagicMock()
    prepare_result_node = copy.deepcopy(DUMMY_NODE)
    prepare_result_node.node_name = DESCRIBE_PREPARE_RESULT_NODE_NAME
    mock_prepare_result.describe.return_value = prepare_result_node

    multi_stage = ProcessingMultiStage.from_hooks(
        prepare_result_fun=mock_prepare_result,
        init_multi_stage_hooks=mock,
        init_hooks=mock,
        stages=mock,
        finalize_hooks=mock,
    )

    result = multi_stage.describe()

    assert result.name == ProcessingMultiStage.__name__

    multi_stage_dict = get_describe_dict(result)

    assert (
        multi_stage_dict["init_multi_stage"][DUMMY_NODE_NAME][LEAF_VALUE] == DUMMY_NODE
    )
    assert multi_stage_dict["init"][DUMMY_NODE_NAME][LEAF_VALUE] == DUMMY_NODE
    assert multi_stage_dict["stages"][DUMMY_NODE_NAME][LEAF_VALUE] == DUMMY_NODE
    assert multi_stage_dict["finalize"][DUMMY_NODE_NAME][LEAF_VALUE] == DUMMY_NODE
    # compare with custom prepare_result_node
    assert (
        multi_stage_dict[DESCRIBE_PREPARE_RESULT_NODE_NAME][LEAF_VALUE]
        == prepare_result_node
    )
