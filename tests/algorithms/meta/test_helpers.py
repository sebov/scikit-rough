from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest

from skrough.algorithms.meta.helpers import normalize_hook_sequence

mock = MagicMock()


@pytest.mark.parametrize(
    "hooks, optional, expected, exception_raise",
    [
        (None, False, None, pytest.raises(ValueError, match="should not be empty")),
        (None, True, [], does_not_raise()),
        (mock, False, [mock], does_not_raise()),
        (mock, True, [mock], does_not_raise()),
        (mock.another, False, [mock.another], does_not_raise()),
        (mock.another, True, [mock.another], does_not_raise()),
        ([mock.other], False, [mock.other], does_not_raise()),
        ([mock.other], True, [mock.other], does_not_raise()),
        (
            [mock.a1, mock.a2, mock.a3],
            False,
            [mock.a1, mock.a2, mock.a3],
            does_not_raise(),
        ),
        (
            [mock.a1, mock.a2, mock.a3],
            True,
            [mock.a1, mock.a2, mock.a3],
            does_not_raise(),
        ),
        (
            [mock.a1, mock.a2, mock.a3, mock.a4, mock.a5],
            False,
            [mock.a1, mock.a2, mock.a3, mock.a4, mock.a5],
            does_not_raise(),
        ),
        (
            [mock.a1, mock.a2, mock.a3, mock.a4, mock.a5],
            True,
            [mock.a1, mock.a2, mock.a3, mock.a4, mock.a5],
            does_not_raise(),
        ),
    ],
)
def test_normalize_hook_sequence(hooks, optional, expected, exception_raise):
    with exception_raise:
        result = normalize_hook_sequence(hooks=hooks, optional=optional)
        assert result == expected
