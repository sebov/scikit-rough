from textwrap import dedent

import pytest

from skrough.algorithms.meta.describe import describe


def function_1():
    """Short description.

    Long description. Long description.

    Long description. Long description.
    """


SHORT_DESCRIPTION = "Short description."
LONG_DESCRIPTION = dedent(
    """\
    Long description. Long description.

    Long description. Long description."""
)


@pytest.mark.parametrize(
    "element, override_node_name, override_short,"
    "expected_node_name, expected_name, expected_short, expected_long",
    [
        (
            function_1,
            None,
            None,
            None,
            function_1.__name__,
            SHORT_DESCRIPTION,
            LONG_DESCRIPTION,
        ),
        (
            function_1,
            "xxx",
            None,
            "xxx",
            function_1.__name__,
            SHORT_DESCRIPTION,
            LONG_DESCRIPTION,
        ),
        (
            function_1,
            None,
            "short",
            None,
            function_1.__name__,
            "short",
            None,
        ),
        (
            function_1,
            "xxx",
            "short",
            "xxx",
            function_1.__name__,
            "short",
            None,
        ),
        (
            1,
            None,
            "short",
            None,
            None,
            "short",
            None,
        ),
        (
            1,
            "xxx",
            "short",
            "xxx",
            None,
            "short",
            None,
        ),
    ],
)
def test_describe_function(
    element,
    override_node_name,
    override_short,
    expected_node_name,
    expected_name,
    expected_short,
    expected_long,
):
    result = describe(
        element,
        override_node_name=override_node_name,
        override_short_description=override_short,
    )
    assert result.node_name == expected_node_name
    assert result.name == expected_name
    assert result.short_description == expected_short
    assert result.long_description == expected_long
