from typing import List

import pytest

from skrough.algorithms.meta.describe import (
    determine_config_keys,
    determine_input_data_keys,
    determine_values_keys,
)

DETERMINE_FUNCTIONS = {
    "config": determine_config_keys,
    "input": determine_input_data_keys,
    "values": determine_values_keys,
}

KEYS_FROM_DOCSTRING = {
    "config": ["CONFIG_KEY_1", "CONFIG_KEY_2", "CONFIG_KEY_3_3_", "CONFIG___YEK_4__"],
    "input": [
        "INPUT_DATA_KEY_1",
        "INPUT_DATA_KEY_2",
        "INPUT_DATA_KEY_3_3_",
        "INPUT_DATA___YEK_4__",
    ],
    "values": ["VALUES_KEY_1", "VALUES_KEY_2", "VALUES_KEY_3_3_", "VALUES___YEK_4__"],
}

KEYS_FROM_METHODS = {
    "config": ["CONFIG_KEY_5", "CONFIG_KEY_6", "CONFIG_KEY_7_7_", "CONFIG___YEK_8__"],
    "input": [
        "INPUT_DATA_KEY_5",
        "INPUT_DATA_KEY_6",
        "INPUT_DATA_KEY_7_7_",
        "INPUT_DATA___YEK_8__",
    ],
    "values": ["VALUES_KEY_5", "VALUES_KEY_6", "VALUES_KEY_7_7_", "VALUES___YEK_8__"],
}


class ClassNoMethodsAndNoDocstring:
    pass


class ClassNoMethodsAndEmptyDocstring:  # pylint: disable=empty-docstring
    """"""


class ClassNoMethodsAndNoKeysDocstring:
    """aaa bbb ccc"""


class ClassNoMethodsAndDocstring:
    """
    CONFIG_KEY_1, `CONFIG_KEY_2`, aaabbccc, ``CONFIG_KEY_3_3_``
    INPUT_DATA_KEY_1, `INPUT_DATA_KEY_2`, aaabbccc, ``INPUT_DATA_KEY_3_3_``
    VALUES_KEY_1, `VALUES_KEY_2`, aaabbccc, ``VALUES_KEY_3_3_``

    VALUES___YEK_4__

    INPUT_DATA___YEK_4__

    CONFIG___YEK_4__
    """


class ClassMethodsAndNoDocstring:
    def get_config_keys(self) -> List[str]:
        return ["CONFIG_KEY_5", "CONFIG_KEY_6", "CONFIG_KEY_7_7_", "CONFIG___YEK_8__"]

    def get_input_data_keys(self) -> List[str]:
        return [
            "INPUT_DATA_KEY_5",
            "INPUT_DATA_KEY_6",
            "INPUT_DATA_KEY_7_7_",
            "INPUT_DATA___YEK_8__",
        ]

    def get_values_keys(self) -> List[str]:
        return ["VALUES_KEY_5", "VALUES_KEY_6", "VALUES_KEY_7_7_", "VALUES___YEK_8__"]


class ClassMethodsAndCallable:
    """
    CONFIG_KEY_1, `CONFIG_KEY_2`, aaabbccc, ``CONFIG_KEY_3_3_``
    INPUT_DATA_KEY_1, `INPUT_DATA_KEY_2`, aaabbccc, ``INPUT_DATA_KEY_3_3_``
    VALUES_KEY_1, `VALUES_KEY_2`, aaabbccc, ``VALUES_KEY_3_3_``

    VALUES___YEK_4__

    INPUT_DATA___YEK_4__

    CONFIG___YEK_4__
    """

    def get_config_keys(self) -> List[str]:
        return ["CONFIG_KEY_5", "CONFIG_KEY_6", "CONFIG_KEY_7_7_", "CONFIG___YEK_8__"]

    def get_input_data_keys(self) -> List[str]:
        return [
            "INPUT_DATA_KEY_5",
            "INPUT_DATA_KEY_6",
            "INPUT_DATA_KEY_7_7_",
            "INPUT_DATA___YEK_8__",
        ]

    def get_values_keys(self) -> List[str]:
        return ["VALUES_KEY_5", "VALUES_KEY_6", "VALUES_KEY_7_7_", "VALUES___YEK_8__"]


@pytest.mark.parametrize(
    "klass",
    [
        ClassNoMethodsAndNoDocstring,
        ClassNoMethodsAndEmptyDocstring,
        ClassNoMethodsAndNoKeysDocstring,
    ],
)
def test_no_keys(klass):
    processing_element = klass()
    for fun in DETERMINE_FUNCTIONS.values():
        assert fun(processing_element) == []


@pytest.mark.parametrize(
    "klass, expected_map",
    [
        (ClassNoMethodsAndDocstring, KEYS_FROM_DOCSTRING),
        (ClassMethodsAndNoDocstring, KEYS_FROM_METHODS),
        (ClassMethodsAndCallable, KEYS_FROM_METHODS),
    ],
)
def test_some_keys(klass, expected_map):
    processing_element = klass()
    for key_kind, fun in DETERMINE_FUNCTIONS.items():
        assert fun(processing_element) == expected_map[key_kind]
