import inspect
from dataclasses import dataclass
from typing import List, Optional

import docstring_parser


@dataclass
class DescriptionNode:
    name: Optional[str] = None
    node_name: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    children: Optional[List["DescriptionNode"]] = None


def describe(
    processing_element,
    node_name: Optional[str] = None,
    short_description: Optional[str] = None,
):
    try:
        result: DescriptionNode = processing_element.describe()
        if node_name is not None:
            result.node_name = node_name
        if short_description is not None:
            result.short_description = short_description
            result.long_description = None
        return result
    except AttributeError:
        name = None
        if hasattr(processing_element, "__name__"):
            name = processing_element.__name__
        long_description = None
        children = None
        if isinstance(processing_element, List):
            children = [describe(child) for child in processing_element]
        else:
            if short_description is None:
                docstring = docstring_parser.parse(
                    inspect.getdoc(processing_element) or ""
                )
                short_description = docstring.short_description
                long_description = docstring.long_description
        return DescriptionNode(
            name=name,
            node_name=node_name,
            short_description=short_description,
            long_description=long_description,
            children=children,
        )
