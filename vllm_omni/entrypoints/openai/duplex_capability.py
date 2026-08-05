# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def should_enable_duplex_endpoint(
    engine_client: object | None = None,
) -> bool:
    """Enable duplex routes when the pipeline declares duplex support."""
    return bool(getattr(engine_client, "duplex_serving_adapter_path", None))


__all__ = ["should_enable_duplex_endpoint"]
