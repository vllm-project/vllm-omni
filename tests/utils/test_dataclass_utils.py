"""Tests for dataclass utils / helpers."""

from dataclasses import dataclass

import pytest

from vllm_omni.utils.dataclass_utils import Trackable, trackable, trackable_to_kwargs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_trackable_args():
    """Ensure we can track dataclasses created with positional args."""

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(32, 64)
    assert obj._init_kwargs == {"foo", "bar"}


def test_trackable_kwargs():
    """Ensure we can track dataclasses created with keyword args."""

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(foo=32, bar=64)
    assert obj._init_kwargs == {"foo", "bar"}


def test_trackable_args_and_kwargs():
    """Ensure we can track dataclasses created with positional & keyword args."""

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(32, bar=64)
    assert obj._init_kwargs == {"foo", "bar"}


def test_trackable_rejects_non_dataclass():
    """Ensure @trackable raises TypeError on non-dataclass classes."""

    with pytest.raises(TypeError, match="currently requires classes to be dataclasses"):

        @trackable
        class NotADataClass:
            def __init__(self, foo, bar):
                pass


def test_trackable_to_kwargs():
    """Ensure a registered trackable can be filtered down to set values."""

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(32, bar=64)
    res = trackable_to_kwargs(obj)
    assert res == {"foo": 32, "bar": 64}


def test_trackable_to_kwargs_raises_with_bad_types():
    """Ensure a non trackable raises TypeError if we try to filter to kwargs."""

    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(foo=32, bar=64)
    with pytest.raises(TypeError):
        trackable_to_kwargs(obj)


def test_trackable_subclasses_are_nontrackable_by_default():
    """Ensure that by default, inheriting from a @trackable dataclass is not
    @trackable, i.e., we just inherit from the dataclass. If the subclass needs
    to be trackable, it should explicit use the decorator.
    """

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    @dataclass
    class NonTrackableSubClass(MyDataClass):
        baz: int = 256

    obj = NonTrackableSubClass()
    assert not isinstance(obj, Trackable)
