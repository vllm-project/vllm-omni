# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenPI client NumPy msgpack codec.

Standalone copy of the wire format in Physical-Intelligence/openpi
``packages/openpi-client/src/openpi_client/msgpack_numpy.py`` at
``231a1cf7caddaae1d0ed1b4956a55de332feb27e``. PyPI ``openpi-client`` pins
``numpy<2``, so weekly CI cannot install that package.

This module must stay independent of
``vllm_omni.entrypoints.openpi.connection`` ``_pack`` / ``_unpack``. Calling
those functions tests the server against itself and stays green if the wire
format is wrong.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
