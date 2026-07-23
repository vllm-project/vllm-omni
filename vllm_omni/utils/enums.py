"""Enum compatibility helpers."""

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    # TODO: Remove this fallback when Python 3.10 support is dropped.
    class StrEnum(str, Enum):
        """Python 3.10-compatible backport of :class:`enum.StrEnum`."""

        __str__ = str.__str__
        __format__ = str.__format__

        @staticmethod
        def _generate_next_value_(name: str, start: int, count: int, last_values: list[object]) -> str:
            return name.lower()


__all__ = ["StrEnum"]
