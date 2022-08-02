from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest

from skrough.algorithms.meta.helpers import normalize_hook_sequence

norm_mock = MagicMock()


@pytest.mark.parametrize(
    "hooks, optional, expected, exception_raise",
    [
        (None, False, None, pytest.raises(ValueError, match="should not be empty")),
        (None, True, [], does_not_raise()),
        ([], False, None, pytest.raises(ValueError, match="should not be empty")),
        ([], True, [], does_not_raise()),
        (norm_mock, False, [norm_mock], does_not_raise()),
        (norm_mock, True, [norm_mock], does_not_raise()),
        (norm_mock.another, False, [norm_mock.another], does_not_raise()),
        (norm_mock.another, True, [norm_mock.another], does_not_raise()),
        ([norm_mock.other], False, [norm_mock.other], does_not_raise()),
        ([norm_mock.other], True, [norm_mock.other], does_not_raise()),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            False,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            does_not_raise(),
        ),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            True,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            does_not_raise(),
        ),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            False,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            does_not_raise(),
        ),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            True,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            does_not_raise(),
        ),
    ],
)
def test_normalize_hook_sequence(hooks, optional, expected, exception_raise):
    with exception_raise:
        result = normalize_hook_sequence(hooks=hooks, optional=optional)
        assert result == expected
