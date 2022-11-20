from textwrap import dedent

import pytest
from attrs import evolve

from skrough.algorithms.meta.describe import describe
from skrough.structs.description_node import DescriptionNode


def function_1():
    """Short description.

    Long description. Long description.

    Long description. Long description.
    """


FUN_SHORT_DESCRIPTION = "Short description."
FUN_LONG_DESCRIPTION = dedent(
    """\
    Long description. Long description.

    Long description. Long description."""
)
FUN_DESCRIPTION_NODE = describe(function_1)


class ClassNoMethodAndNonCallable:
    """Another short description.

    Another long description.
    """

    def __init__(self) -> None:
        self.x = 1
        self.fun = function_1


class ClassNoMethodAndCallable:
    """Another short description.

    Another long description.
    """

    def __init__(self) -> None:
        self.x = 1
        self.fun = function_1

    def __call__(self):
        pass


class ClassMethodAndNonCallable:
    """Another short description.

    Another long description.
    """

    def __init__(self) -> None:
        self.x = 1
        self.fun = function_1

    def get_description_graph(self):
        result = DescriptionNode(
            name="non-callable",
            short_description="short",
            long_description="long",
            children=[describe(self.x, "x"), describe(self.fun, "fun")],
        )
        return result


class ClassMethodAndCallable:
    """Another short description.

    Another long description.
    """

    def __init__(self) -> None:
        self.x = 1
        self.fun = function_1

    def get_description_graph(self):
        result = DescriptionNode(
            name="callable",
            short_description="short",
            long_description="long",
            children=[describe(self.x, "x"), describe(self.fun, "fun")],
        )
        return result

    def __call__(self):
        pass


CLASS_SHORT_DESCRIPTION = "Another short description."
CLASS_LONG_DESCRIPTION = "Another long description."
INT_DESCRIPTION_NODE = describe(0)


@pytest.mark.parametrize(
    "override_node_name",
    [None, "node_name"],
)
@pytest.mark.parametrize(
    "override_node_meta",
    [None, {}, {"a": 1}, {"a": 1, "b": 2}],
)
@pytest.mark.parametrize(
    "element, override_short, expected_name, expected_short, expected_long",
    [
        (
            function_1,
            None,
            function_1.__name__,
            FUN_SHORT_DESCRIPTION,
            FUN_LONG_DESCRIPTION,
        ),
        (
            function_1,
            "short",
            function_1.__name__,
            "short",
            None,
        ),
        (
            1,
            "short",
            None,
            "short",
            None,
        ),
    ],
)
def test_describe_function(
    element,
    override_node_name,
    override_node_meta,
    override_short,
    expected_name,
    expected_short,
    expected_long,
):
    result = describe(
        element,
        override_node_name=override_node_name,
        override_node_meta=override_node_meta,
        override_short_description=override_short,
    )
    assert result.node_name == override_node_name
    assert result.node_meta == override_node_meta
    assert result.name == expected_name
    assert result.short_description == expected_short
    assert result.long_description == expected_long


@pytest.mark.parametrize(
    "element",
    [
        [],
        [function_1],
        [0],
        [function_1, 0, function_1, 2, 2, function_1],
    ],
)
@pytest.mark.parametrize(
    "override_node_name",
    [None, "node_name"],
)
@pytest.mark.parametrize(
    "override_node_meta",
    [None, {}, {"a": 1}, {"a": 1, "b": 2}],
)
@pytest.mark.parametrize(
    "override_short",
    [None, "short"],
)
def test_describe_list(
    element,
    override_node_name,
    override_node_meta,
    override_short,
):
    """Test if lists are handled/described correctly.

    For the purpose of the test let's assume that the input ``element`` is a list that
    contains a sequence of integers and ``function_1``s (possibly multiple occurrences).
    """
    result = describe(
        element,
        override_node_name=override_node_name,
        override_node_meta=override_node_meta,
        override_short_description=override_short,
    )
    assert result.node_name == override_node_name
    assert result.node_meta == override_node_meta
    assert result.name is None
    assert result.short_description == override_short
    assert result.long_description is None
    assert result.children is not None
    assert len(result.children) == len(element)

    # we assume that subelements are equal to either ``function_1`` or an integer
    for i, subelement in enumerate(element):
        if subelement is function_1:
            expected = FUN_DESCRIPTION_NODE
        else:
            expected = INT_DESCRIPTION_NODE
        # set expected node_name, i.e., str(position) in children list
        expected = evolve(expected, node_name=str(i))
        assert result.children[i] == expected


@pytest.mark.parametrize(
    "override_node_name",
    [None, "node_name"],
)
@pytest.mark.parametrize(
    "override_node_meta",
    [None, {}, {"a": 1}, {"a": 1, "b": 2}],
)
@pytest.mark.parametrize(
    "override_short",
    [None, "short"],
)
def test_describe_no_method_and_non_callable(
    override_node_name,
    override_node_meta,
    override_short,
):
    """No describe method available and non-callable."""
    result = describe(
        ClassNoMethodAndNonCallable(),
        override_node_name=override_node_name,
        override_node_meta=override_node_meta,
        override_short_description=override_short,
    )
    assert result.node_name == override_node_name
    assert result.node_meta == override_node_meta
    assert result.name is None
    assert result.short_description == override_short
    assert result.long_description is None
    assert result.children is None


@pytest.mark.parametrize(
    "override_node_name",
    [None, "node_name"],
)
@pytest.mark.parametrize(
    "override_node_meta",
    [None, {}, {"a": 1}, {"a": 1, "b": 2}],
)
@pytest.mark.parametrize(
    "override_short",
    [None, "short"],
)
def test_describe_no_method_and_callable(
    override_node_name,
    override_node_meta,
    override_short,
):
    """No describe method available and callable."""
    result = describe(
        ClassNoMethodAndCallable(),
        override_node_name=override_node_name,
        override_node_meta=override_node_meta,
        override_short_description=override_short,
    )
    assert result.node_name == override_node_name
    assert result.node_meta == override_node_meta
    assert result.name is ClassNoMethodAndCallable.__name__
    assert result.short_description == override_short or CLASS_SHORT_DESCRIPTION
    assert result.long_description == (
        None if override_short is not None else CLASS_LONG_DESCRIPTION
    )
    assert result.children is None


@pytest.mark.parametrize(
    "klass",
    [ClassMethodAndNonCallable, ClassMethodAndCallable],
)
@pytest.mark.parametrize(
    "override_node_name",
    [None, "node_name"],
)
@pytest.mark.parametrize(
    "override_node_meta",
    [None, {}, {"a": 1}, {"a": 1, "b": 2}],
)
@pytest.mark.parametrize(
    "override_short",
    [None, "short"],
)
def test_get_description_graph_method(
    klass,
    override_node_name,
    override_node_meta,
    override_short,
):
    """Describe method available."""
    result = describe(
        klass(),
        override_node_name=override_node_name,
        override_node_meta=override_node_meta,
        override_short_description=override_short,
    )

    expected_result = klass().get_description_graph()
    # apply override manually
    if override_node_name is not None:
        expected_result.node_name = override_node_name
    if override_node_meta is not None:
        expected_result.node_meta = override_node_meta
    if override_short is not None:
        expected_result.short_description = override_short
        expected_result.long_description = None

    assert result == expected_result
