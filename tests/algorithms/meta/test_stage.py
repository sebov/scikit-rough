from unittest.mock import MagicMock

from skrough.algorithms.meta.stage import Stage


def test_stage():
    mock = MagicMock()
    stage = Stage.from_hooks(
        stop_hooks=mock.stop_hook,
        init_hooks=None,
        pre_candidates_hooks=None,
        candidates_hooks=None,
        select_hooks=None,
        filter_hooks=None,
        inner_init_hooks=None,
        inner_stop_hooks=mock.inner_stop_hook,
        inner_process_hooks=mock.inner_process_hooks,
        finalize_hooks=None,
    )
    assert stage.stop_agg.normalized_hooks[0] == mock.stop_hook
