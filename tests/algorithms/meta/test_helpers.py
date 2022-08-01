from contextlib import nullcontext as does_not_raise

import pytest

from skrough.algorithms.meta.helpers import normalize_hook_sequence

# a collection of dummy functions
dummy = [lambda x=k: x for k in range(10)]


@pytest.mark.parametrize(
    "hooks, optional, expected, exception_raise",
    [
        (None, False, None, pytest.raises(ValueError, match="should not be empty")),
        (None, True, [], does_not_raise()),
        (dummy[0], False, [dummy[0]], does_not_raise()),
        (dummy[0], True, [dummy[0]], does_not_raise()),
        (dummy[1], False, [dummy[1]], does_not_raise()),
        (dummy[1], True, [dummy[1]], does_not_raise()),
        ([dummy[5]], False, [dummy[5]], does_not_raise()),
        ([dummy[5]], True, [dummy[5]], does_not_raise()),
        (dummy[:5], False, dummy[:5], does_not_raise()),
        (dummy[:5], True, dummy[:5], does_not_raise()),
        (dummy, False, dummy, does_not_raise()),
        (dummy, True, dummy, does_not_raise()),
    ],
)
def test_normalize_hook_sequence(hooks, optional, expected, exception_raise):
    with exception_raise:
        result = normalize_hook_sequence(hooks=hooks, optional=optional)
        assert result == expected
