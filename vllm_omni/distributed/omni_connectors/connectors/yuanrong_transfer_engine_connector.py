# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility import for the NPU-only Yuanrong TransferEngine connector."""

from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
    YuanrongTransferEngineConnector,
)

__all__ = ["YuanrongTransferEngineConnector"]
