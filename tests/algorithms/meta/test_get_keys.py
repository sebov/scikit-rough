from typing import List

import pytest

from skrough.algorithms.meta.describe import (
    determine_config_keys,
    determine_input_data_keys,
    determine_values_keys,
)


class MockedKeyNames:
    def __getattribute__(self, name: str) -> str:
        return name.lower()


@pytest.fixture(autouse=True)
def mock_key_names(monkeypatch):
    monkeypatch.setattr("skrough.algorithms.meta.describe.key_names", MockedKeyNames())


DETERMINE_FUNCTIONS = {
    "config": determine_config_keys,
    "input_data": determine_input_data_keys,
    "values": determine_values_keys,
}

KEYS_FROM_DOCSTRING = {
    "config": ["config_key_1", "config_key_2", "config_key_3_3_", "config___yek_4__"],
    "input_data": [
        "input_data_key_1",
        "input_data_key_2",
        "input_data_key_3_3_",
        "input_data___yek_4__",
    ],
    "values": ["values_key_1", "values_key_2", "values_key_3_3_", "values___yek_4__"],
}

KEYS_FROM_METHODS = {
    "config": ["config_key_5", "config_key_6", "config_key_7_7_", "config___yek_8__"],
    "input_data": [
        "input_data_key_5",
        "input_data_key_6",
        "input_data_key_7_7_",
        "input_data___yek_8__",
    ],
    "values": ["values_key_5", "values_key_6", "values_key_7_7_", "values___yek_8__"],
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
        return list(reversed(KEYS_FROM_METHODS["config"]))

    def get_input_data_keys(self) -> List[str]:
        return list(reversed(KEYS_FROM_METHODS["input_data"]))

    def get_values_keys(self) -> List[str]:
        return list(reversed(KEYS_FROM_METHODS["values"]))


class ClassMethodsAndCallable(ClassMethodsAndNoDocstring):
    """
    CONFIG_KEY_1, `CONFIG_KEY_2`, aaabbccc, ``CONFIG_KEY_3_3_``
    INPUT_DATA_KEY_1, `INPUT_DATA_KEY_2`, aaabbccc, ``INPUT_DATA_KEY_3_3_``
    VALUES_KEY_1, `VALUES_KEY_2`, aaabbccc, ``VALUES_KEY_3_3_``

    VALUES___YEK_4__

    INPUT_DATA___YEK_4__

    CONFIG___YEK_4__
    """


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
        actual = fun(processing_element)
        expected = expected_map[key_kind]
        assert len(actual) == len(expected)
        assert set(actual) == set(expected)
