import inspect
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import docstring_parser
from attrs import define

NODE_META_OPTIONAL_KEY = "optional"
NodeMeta = Dict[str, Union[str, bool, int, float]]


@define
class DescriptionNode:
    """Description of a processing element.

    Description node represents a single element in the processing graph. It is
    consisted of:

    - node_name - name for the vertex in the processing graph; for example, it can
      correspond to an attribute name of the higher level structure that stores the
      element or it can represent a "virtual" loop element
    - node_meta - additional meta info (a dict) for the vertex in the processing graph;
      for example it can be used to store information that the given node (and
      processing it represents) is optional, etc.
    - name - name of the processing element; for example, it can be the name of the
      function implementing the processing element or it can be :obj:`None` when the
      element is a container for other processing elements
    - short_description - a short description (usually a sentence) describing the
      processing element; for example the short summary from a function's docstring
    - long_description - an extended description providing details for the processing
      element; for example the extended summary from a function's docstring
    - children - subelements (subconcepts) of the processing element; for example a
      list/chain of functions will have its elements described in the ``children`` list
    """

    node_name: Optional[str] = None
    node_meta: Optional[NodeMeta] = None
    name: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    children: Optional[List["DescriptionNode"]] = None


def _get_metadata_for_callable(
    element: Callable,
    process_docstring: bool,
) -> Tuple[str, Optional[str], Optional[str]]:
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
        docstring = docstring_parser.parse(inspect.getdoc(element) or "")
        short_description = docstring.short_description
        long_description = docstring.long_description
    return name, short_description, long_description


def describe(
    processing_element,
    override_node_name: Optional[str] = None,
    override_node_meta: Optional[NodeMeta] = None,
    override_short_description: Optional[str] = None,
) -> DescriptionNode:
    """Get a description of a given ``processing_element``.

    Prepare a description structure for a given ``processing_element``. The function
    will use ``processing_element`` method if available for the given object, i.e., it
    will use the method to let ``processing_element`` self-describe itself. Otherwise,
    if ``processing_element`` is callable then the function will automatically generate
    the description structure using available information, e.g., a name attribute stored
    directly in the element or in its class, or parsing the element's docstring to
    obtain textual description. For non-callable elements the function fill produce a
    dummy description structure filled with :obj:`None`.

    Args:
        processing_element: A processing element to be described.
        override_node_name: If a string value is given then it will override the
            results's ``node_name`` attribute. Defaults to :obj:`None`.
        override_short_description: If a string value is give then it will override the
            result's ``short_description`` attribute and it will also set
            ``long_description`` to :obj:`None`. Defaults to :obj:`None`.

    Returns:
        Description structure representing the input ``processing_element``.
    """
    try:
        # try to use element's describe method
        result: DescriptionNode = processing_element.get_description_graph()
    except AttributeError:
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
                _name, _short, _long = _get_metadata_for_callable(
                    element=processing_element,
                    process_docstring=override_short_description is None,
                )
                name = _name
                if _short is not None:
                    short_description = _short
                if _long is not None:
                    long_description = _long
        result = DescriptionNode(
            name=name,
            short_description=short_description,
            long_description=long_description,
            children=children,
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
