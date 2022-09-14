import inspect
from dataclasses import dataclass
from typing import List, Optional, Sequence

import docstring_parser


@dataclass
class DescriptionNode:
    """Description of a processing element.

    Description node represents a single element in the processing graph. It is
    consisted of:

    * node_name - meta name for the vertex in the processing graph; for example, it can
      correspond to an attribute name of the higher level structure that stores the
      element or it can represent a "virtual" loop element
    * name - name of the processing element; for example, it can be the name of the
      function implementing the processing element or it can be ``None`` when the
      element is a container for other processing elements
    * short_description - a short description (usually a sentence) describing the
      processing element; for example the short summary from a function's docstring
    * long_description - an extended description providing details for the processing
      element; for example the extended summary from a function's docstring
    * children - subelements (subconcepts) of the processing element; for example a
      list/chain of functions will have its elements described in the ``children`` list
    """

    node_name: Optional[str] = None
    name: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    children: Optional[List["DescriptionNode"]] = None


def describe(
    processing_element,
    override_node_name: Optional[str] = None,
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
    dummy description structure filled with `None`s.

    Args:
        processing_element: A processing element to be described.
        override_node_name: If a string value is given then it will override the
            results's ``node_name`` attribute. Defaults to ``None``.
        override_short_description: If a string value is give then it will override the
            result's ``short_description`` attribute and it will also set
            ``long_description`` to ``None``. Defaults to ``None``.

    Returns:
        Description structure representing the input ``processing_element``.
    """
    try:
        # try to use element's describe method
        result: DescriptionNode = processing_element.describe()
        print("a")
    except AttributeError:
        # otherwise, try to autogenerate
        name = None
        short_description = None
        long_description = None
        children = None
        if isinstance(processing_element, Sequence):
            children = [describe(child) for child in processing_element]
        else:
            # obtain name from a callable element
            if callable(processing_element):
                if hasattr(processing_element, "__name__"):
                    # either directly
                    name = processing_element.__name__
                elif hasattr(processing_element, "__class__"):
                    # or from the element's class
                    name = processing_element.__class__.__name__
                if override_short_description is None:
                    # try to obtain short and long descriptions from docstring
                    docstring = docstring_parser.parse(
                        inspect.getdoc(processing_element) or ""
                    )
                    short_description = docstring.short_description
                    long_description = docstring.long_description
        result = DescriptionNode(
            name=name,
            short_description=short_description,
            long_description=long_description,
            children=children,
        )

    # override result's attributes if given
    if override_node_name is not None:
        result.node_name = override_node_name
    if override_short_description is not None:
        result.short_description = override_short_description
        result.long_description = None
    return result
