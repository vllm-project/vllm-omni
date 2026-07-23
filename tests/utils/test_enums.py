from enum import auto

import pytest

from vllm_omni.utils.enums import StrEnum

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class ExampleStrEnum(StrEnum):
    EXPLICIT = "explicit"
    AUTO_VALUE = auto()


def test_str_enum_explicit_value_semantics():
    assert ExampleStrEnum.EXPLICIT == "explicit"
    assert str(ExampleStrEnum.EXPLICIT) == "explicit"
    assert f"{ExampleStrEnum.EXPLICIT}" == "explicit"


def test_str_enum_auto_uses_lowercase_member_name():
    assert ExampleStrEnum.AUTO_VALUE.value == "auto_value"
    assert str(ExampleStrEnum.AUTO_VALUE) == "auto_value"
