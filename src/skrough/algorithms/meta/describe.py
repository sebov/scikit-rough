import inspect
import logging
from typing import Callable, Sequence

import docstring_parser

from skrough.structs.description_node import DescriptionNode, NodeMeta

SKROUGH_DOCSTRING_STYLE = docstring_parser.common.DocstringStyle.GOOGLE

logger = logging.getLogger(__name__)


def _get_metadata_for_callable(
    element: Callable,
    process_docstring: bool,
) -> tuple[str, str | None, str | None]:
    """Obtain ``processing_element`` metadata.

    Prepare metadata for a Callable element. The function retrieve element's name basing
    on special (dunder) attributes, i.e., either ``__name__`` or ``__class__.__name__``
    whichever occurs first in the given order. The `process_docstring` parameter
    instructs the function to also parse element's docstring to search for and to return
    the element's short and long descriptions.

    Args:
        element: An element to be analyzed.
        process_docstring: A flag indicating whether a processing element's docstring
            should be analyzed for retrieving short and long descriptions.

    Returns:
        Result is consisted of the following elements

        - the element's name
        - an optional short element's description retrieved from docstring
        - an optional long element's description retrieved from docstring
    """
    short_description = None
    long_description = None
    if hasattr(element, "__name__"):
        # either directly
        name = element.__name__
    else:
        # or from the element's class
        name = element.__class__.__name__
    if process_docstring:
        # try to obtain short and long descriptions from docstring
        docstring = docstring_parser.parse(
            inspect.getdoc(element) or "",
            style=SKROUGH_DOCSTRING_STYLE,
        )
        short_description = docstring.short_description
        long_description = docstring.long_description
    return name, short_description, long_description


def autogenerate_description_node(
    processing_element,
    process_docstring: bool,
) -> DescriptionNode:
    # otherwise, try to autogenerate
    name = None
    short_description = None
    long_description = None
    children = None
    if isinstance(processing_element, Sequence):
        children = [
            describe(child, override_node_name=str(i))
            for i, child in enumerate(processing_element)
        ]
    else:
        # obtain data from a callable element
        if callable(processing_element):
            name, short_description, long_description = _get_metadata_for_callable(
                element=processing_element,
                process_docstring=process_docstring,
            )
    result = DescriptionNode(
        name=name,
        short_description=short_description,
        long_description=long_description,
        children=children,
    )
    return result


def describe(
    processing_element,
    override_node_name: str | None = None,
    override_node_meta: NodeMeta | None = None,
    override_short_description: str | None = None,
) -> DescriptionNode:
    """Get a description graph of a given ``processing_element``.

    Prepare a description structure for a given ``processing_element``. The function
    will use ``processing_element`` method if available for the given object, i.e., it
    will use the method to let ``processing_element`` self-describe itself. Otherwise,
    if ``processing_element`` is a callable then the function will automatically
    generate the description structure using available information (using
    ``autogenerate_description_node`` function), e.g., a name attribute stored directly
    in the element or in its class, or parsing the element's docstring to obtain textual
    description. For non-callable elements the function fill produce a dummy description
    structure filled with :obj:`None`.

    Args:
        processing_element: A processing element to be described.
        override_node_name: If a string value is given then it will override the
            results's ``node_name`` attribute. Defaults to :obj:`None`.
        override_node_meta: If a ``NodeMeta`` value is given them it will override the
            results's ``node_meta`` attribute. Defaults to :obj:`None`.
        override_short_description: If a string value is give then it will override the
            result's ``short_description`` attribute and it will also set
            ``long_description`` to :obj:`None`. Defaults to :obj:`None`.

    Returns:
        A description graph structure representing the input ``processing_element``.
    """
    result: DescriptionNode
    try:
        # try to use element's describe method
        result = processing_element.get_description_graph()
    except AttributeError:
        result = autogenerate_description_node(
            processing_element=processing_element,
            process_docstring=override_short_description is None,
        )

    # override result's attributes if given
    if override_node_name is not None:
        result.node_name = override_node_name
    if override_node_meta is not None:
        result.node_meta = override_node_meta
    if override_short_description is not None:
        result.short_description = override_short_description
        result.long_description = None
    return result
