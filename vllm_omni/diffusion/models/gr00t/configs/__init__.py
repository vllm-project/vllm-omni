# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

MODEL_CONFIG_TYPES: dict[str, type] = {}


def register_model_config(shortname: str, configtype: type) -> None:
    MODEL_CONFIG_TYPES[shortname] = configtype
