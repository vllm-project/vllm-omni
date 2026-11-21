# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OmniCodec:
    """Encode/decode pair for one Python type and its wire tag."""

    tag: str
    python_type: type
    encode: Callable[[Any], Any] | None = None
    decode: Callable[[dict[str, Any]], Any] | None = None


class OmniCodecRegistry:
    """O(1) codec lookup by Python type, with MRO fallback and tag lookup."""

    def __init__(self) -> None:
        self.codecs_by_type: dict[type, OmniCodec] = {}
        self.codecs_by_tag: dict[str, OmniCodec] = {}

    def register(
        self,
        python_type: type,
        tag: str,
        encode: Callable[[Any], Any] | None = None,
        decode: Callable[[dict[str, Any]], Any] | None = None,
        *,
        override: bool = False,
    ) -> None:
        if encode is None and decode is None:
            raise ValueError("At least one of encode or decode is required")
        if not override and (python_type in self.codecs_by_type or tag in self.codecs_by_tag):
            raise ValueError(f"Codec already registered for {python_type} or {tag}")

        if override:
            previous = self.codecs_by_type.pop(python_type, None)
            if previous is not None and self.codecs_by_tag.get(previous.tag) is previous:
                del self.codecs_by_tag[previous.tag]
            previous_tag = self.codecs_by_tag.pop(tag, None)
            if previous_tag is not None and self.codecs_by_type.get(previous_tag.python_type) is previous_tag:
                del self.codecs_by_type[previous_tag.python_type]

        codec = OmniCodec(tag=tag, python_type=python_type, encode=encode, decode=decode)
        self.codecs_by_type[python_type] = codec
        self.codecs_by_tag[tag] = codec

    def lookup(self, python_type: type) -> OmniCodec | None:
        for candidate in python_type.__mro__:
            codec = self.codecs_by_type.get(candidate)
            if codec is not None:
                return codec
        return None

    def lookup_tag(self, tag: str) -> OmniCodec | None:
        return self.codecs_by_tag.get(tag)

    def lookup_marker(self, payload: dict[str, Any]) -> OmniCodec | None:
        """Return the codec whose registered tag marks ``payload``.

        Decoding walks the registry instead of a fixed list of built-in
        markers, so a newly registered leaf type is restored by its own codec.
        """
        for tag, codec in self.codecs_by_tag.items():
            if payload.get(tag):
                return codec
        return None
