import inspect
import logging
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import docstring_parser

import skrough.typing as rght
from skrough.algorithms import key_names
from skrough.algorithms.key_names import (
    CONFIG_KEYS_DOCSTRING_REGEX,
    INPUT_DATA_KEYS_DOCSTRING_REGEX,
    VALUES_KEYS_DOCSTRING_REGEX,
)
from skrough.structs.description_node import DescriptionNode, NodeMeta
from skrough.structs.state import StateConfig, StateInputData

SKROUGH_DOCSTRING_STYLE = docstring_parser.common.DocstringStyle.GOOGLE

logger = logging.getLogger(__name__)


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
    config_keys = None
    input_keys = None
    values_keys = None
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
            config_keys = inspect_config_keys(processing_element)
            input_keys = inspect_input_data_keys(processing_element)
            values_keys = inspect_values_keys(processing_element)
    result = DescriptionNode(
        name=name,
        short_description=short_description,
        long_description=long_description,
        config_keys=config_keys,
        input_keys=input_keys,
        values_keys=values_keys,
        children=children,
    )
    return result


def describe(
    processing_element,
    override_node_name: Optional[str] = None,
    override_node_meta: Optional[NodeMeta] = None,
    override_short_description: Optional[str] = None,
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


def _inspect_keys(
    processing_element,
    key_method_name: str,
    regex_pattern: str,
) -> List[str]:
    try:
        result: List[str] = getattr(processing_element, key_method_name)()
    except AttributeError:
        _, _short, _long = _get_metadata_for_callable(
            processing_element, process_docstring=True
        )
        result = re.findall(regex_pattern, (_short or "") + (_long or ""))
        # in case of the docstring parsing - we also need to decode key names used
        # commonly in docstring descriptions, i.e., constant names from
        # skrough.algorithms.key_names, to actual keys
        result = list(filter(None, [getattr(key_names, name, None) for name in result]))
    return result


def inspect_config_keys(processing_element) -> List[str]:
    return _inspect_keys(
        processing_element,
        key_method_name=rght.Describable.get_config_keys.__name__,
        regex_pattern=CONFIG_KEYS_DOCSTRING_REGEX,
    )


def inspect_input_data_keys(processing_element) -> List[str]:
    return _inspect_keys(
        processing_element,
        key_method_name=rght.Describable.get_input_data_keys.__name__,
        regex_pattern=INPUT_DATA_KEYS_DOCSTRING_REGEX,
    )


def inspect_values_keys(processing_element) -> List[str]:
    return _inspect_keys(
        processing_element,
        key_method_name=rght.Describable.get_values_keys.__name__,
        regex_pattern=VALUES_KEYS_DOCSTRING_REGEX,
    )


def check_compatibility(
    processing_element,
    config: StateConfig,
    input_data: StateInputData,
    verbose: bool = False,
) -> Union[bool, Tuple[bool, Dict[str, List[str]]]]:
    config_keys_ok = True
    input_data_keys_ok = True
    verbose_report = {}
    actual_config_keys = inspect_config_keys(processing_element)
    if not set(actual_config_keys).issubset(config.keys()):
        logger.info("some of the required config keys are not present in the state")
        config_keys_ok = False
        if verbose:
            verbose_report["missing_config_keys"] = list(
                set(actual_config_keys).difference(config.keys())
            )
    actual_input_data_keys = inspect_input_data_keys(processing_element)
    if not set(actual_input_data_keys).issubset(input_data.keys()):
        logger.info("some of the required input data keys are not present in the state")
        input_data_keys_ok = False
        if verbose:
            verbose_report["missing_input_data_keys"] = list(
                set(actual_input_data_keys).difference(input_data.keys())
            )

    result = config_keys_ok and input_data_keys_ok
    if verbose:
        return result, verbose_report
    return result
