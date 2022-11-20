import json
from typing import List, Optional, Union

from sklearn.utils._estimator_html_repr import _VisualBlock

from skrough.algorithms.meta.describe import describe
from skrough.structs.description_node import DescriptionNode, NodeMeta

VB_META_KEY = "meta"
VB_NAMES_SEP = ": "
VB_NAMES_NODE_META_SEP = " + "
VB_NAME_DETAILS_SEP = "\n\n"


def _prepare_vb_names(
    node_name: Optional[str],
    node_meta: Optional[NodeMeta],
    name: Optional[str],
) -> str:
    prefix = node_name or ""
    suffix = name or ""
    result = f"{prefix}{VB_NAMES_SEP}{suffix}"
    meta = json.dumps({VB_META_KEY: node_meta}) if node_meta else ""
    return VB_NAMES_NODE_META_SEP.join(filter(None, [result, meta]))


def _prepare_vb_name_details(
    short_description: Optional[str],
    long_description: Optional[str],
) -> str:
    short_description = short_description or ""
    long_description = long_description or ""
    return VB_NAME_DETAILS_SEP.join(filter(None, [short_description, long_description]))


def _description_node_to_vb(description: DescriptionNode):
    kind: str
    estimators: Optional[List[_VisualBlock]]
    names: Union[List[Optional[str]], Optional[str]]
    name_details: Union[List[Optional[str]], Optional[str]]
    if description.children is not None:
        kind = "serial"
        estimators = [_description_node_to_vb(child) for child in description.children]
        names = [
            _prepare_vb_names(child.node_name, child.node_meta, child.name)
            for child in description.children
        ]
        name_details = [
            _prepare_vb_name_details(child.short_description, child.long_description)
            for child in description.children
        ]
    else:
        kind = "single"
        estimators = None
        names = _prepare_vb_names(
            description.node_name, description.node_meta, description.name
        )
        name_details = _prepare_vb_name_details(
            description.short_description, description.long_description
        )

    return _VisualBlock(
        kind=kind,
        estimators=estimators,
        names=names,
        name_details=name_details,
        dash_wrapped=True,
    )


def sk_visual_block(estimator):
    description = describe(estimator)
    return _description_node_to_vb(description)
