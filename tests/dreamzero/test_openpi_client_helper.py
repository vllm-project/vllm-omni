# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from tests.dreamzero import openpi_client_helper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeMsgpackNumpy:
    payloads = {}

    @staticmethod
    def packb(obj):
        key = str(len(FakeMsgpackNumpy.payloads)).encode()
        FakeMsgpackNumpy.payloads[key] = obj
        return key

    @staticmethod
    def unpackb(data):
        return FakeMsgpackNumpy.payloads[data]


def test_decode_action_response_surfaces_structured_error(monkeypatch):
    monkeypatch.setattr(openpi_client_helper, "msgpack_numpy", FakeMsgpackNumpy)
    payload = FakeMsgpackNumpy.packb(
        {
            "type": "error",
            "message": "Internal inference error",
        }
    )

    with pytest.raises(RuntimeError, match="Internal inference error"):
        openpi_client_helper._decode_action_response(payload)


def test_decode_action_response_converts_action_payload_to_float32(monkeypatch):
    monkeypatch.setattr(openpi_client_helper, "msgpack_numpy", FakeMsgpackNumpy)
    payload = FakeMsgpackNumpy.packb(np.asarray([[1.0, 2.0]], dtype=np.float64))

    actions = openpi_client_helper._decode_action_response(payload)

    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions, np.asarray([[1.0, 2.0]], dtype=np.float32))
