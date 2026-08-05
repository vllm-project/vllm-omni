# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

try:
    from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector,
    )
except ImportError:
    YuanrongTransferEngineConnector = None

__all__ = ["YuanrongTransferEngineConnector"]
