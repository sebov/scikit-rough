import json

import pytest
from attrs import evolve

from skrough.algorithms.meta.describe import DescriptionNode, NodeMeta
from skrough.algorithms.meta.visual_block import (
    VB_META_KEY,
    VB_NAME_DETAILS_SEP,
    VB_NAMES_NODE_META_SEP,
    VB_NAMES_SEP,
    _description_node_to_vb,
    _prepare_vb_name_details,
    _prepare_vb_names,
    sk_visual_block,
)

DUMMY_NODE_NAME = "node_name"
DUMMY_NODE_META: NodeMeta = {"node_meta": True}
DUMMY_NAME = "name"
DUMMY_SHORT = "short"
DUMMY_LONG = "long"

DUMMY_DESCRIPTION_NODE = DescriptionNode(
    node_name=DUMMY_NODE_NAME,
    node_meta=DUMMY_NODE_META,
    name=DUMMY_NAME,
    short_description=DUMMY_SHORT,
    long_description=DUMMY_LONG,
)

VB_RICH_NAMES = (
    f"{DUMMY_NODE_NAME}{VB_NAMES_SEP}{DUMMY_NAME}"
    f"{VB_NAMES_NODE_META_SEP}{json.dumps({VB_META_KEY: DUMMY_NODE_META})}"
)
VB_RICH_NAME_DETAILS = f"{DUMMY_SHORT}{VB_NAME_DETAILS_SEP}{DUMMY_LONG}"


class DummyEstimator:
    def get_description_graph(self):
        return evolve(
            DUMMY_DESCRIPTION_NODE,
            children=[
                evolve(DUMMY_DESCRIPTION_NODE),
            ],
        )


@pytest.mark.parametrize(
    "node_name, node_meta, name, expected",
    [
        (None, None, None, VB_NAMES_SEP),
        (
            DUMMY_NODE_NAME,
            None,
            DUMMY_NAME,
            f"{DUMMY_NODE_NAME}{VB_NAMES_SEP}{DUMMY_NAME}",
        ),
        (
            DUMMY_NODE_NAME,
            DUMMY_NODE_META,
            DUMMY_NAME,
            VB_RICH_NAMES,
        ),
    ],
)
def test_prepare_vb_names(node_name, node_meta, name, expected):
    assert _prepare_vb_names(node_name, node_meta, name) == expected


@pytest.mark.parametrize(
    "short, long, expected",
    [
        (None, None, ""),
        (DUMMY_SHORT, None, DUMMY_SHORT),
        (None, DUMMY_LONG, DUMMY_LONG),
        (DUMMY_SHORT, DUMMY_LONG, f"{DUMMY_SHORT}{VB_NAME_DETAILS_SEP}{DUMMY_LONG}"),
    ],
)
def test_prepare_vb_name_details(short, long, expected):
    assert _prepare_vb_name_details(short, long) == expected


@pytest.mark.parametrize(
    "description, kind, names, name_details",
    [
        (
            DUMMY_DESCRIPTION_NODE,
            "single",
            VB_RICH_NAMES,
            VB_RICH_NAME_DETAILS,
        ),
        (
            evolve(
                DUMMY_DESCRIPTION_NODE,
                children=[
                    evolve(DUMMY_DESCRIPTION_NODE),
                ],
            ),
            "serial",
            VB_RICH_NAMES,
            VB_RICH_NAME_DETAILS,
        ),
    ],
)
def test_description_node_to_vb(description, kind, names, name_details):
    result = _description_node_to_vb(description)
    assert result.kind == kind

    if kind == "single":
        assert result.names == names
        assert result.name_details == name_details
    else:
        assert result.names == [names] * len(description.children)
        assert result.name_details == [name_details] * len(description.children)


def test_sk_visual_block():
    estimator = DummyEstimator()
    result = sk_visual_block(estimator)
    assert result.names == [VB_RICH_NAMES]
    assert result.name_details == [VB_RICH_NAME_DETAILS]
    assert result.kind == "serial"
    assert result.estimators is not None
    assert result.estimators[0].names == VB_RICH_NAMES
    assert result.estimators[0].name_details == VB_RICH_NAME_DETAILS
    assert result.estimators[0].kind == "single"
