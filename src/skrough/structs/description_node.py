"""Description node structures."""

from typing import Dict, List, Optional, Union

from attrs import define

NodeMeta = Dict[str, Union[str, bool, int, float]]

NODE_META_OPTIONAL_KEY = "optional"


@define
class DescriptionNode:
    """Description node of a processing element.

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
    - config_keys - a list of "config" keys used by the processing element and its
      descendants in the processing graph
    - input_keys - a list of "input" keys used by the processing element and its
      descendants in the processing graph
    - values_keys - a list of "values" keys used by the processing element and its
      descendants in the processing graph
    - children - subelements (subconcepts) of the processing element; for example a
      list/chain of functions will have its elements described in the ``children`` list
    """

    node_name: Optional[str] = None
    node_meta: Optional[NodeMeta] = None
    name: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    config_keys: Optional[List[str]] = None
    input_keys: Optional[List[str]] = None
    values_keys: Optional[List[str]] = None
    children: Optional[List["DescriptionNode"]] = None
