# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speaker-related utilities for Qwen3 TTS."""


def normalize_spk_id_map(raw_map: dict[str, int] | None) -> dict[str, int]:
    """Normalize speaker ID map keys to lowercase for case-insensitive lookup.

    The model config (``talker_config.spk_id``) stores speaker names in mixed
    case (e.g. ``"Ryan"``, ``"Vivian"``), while the serving layer normalizes
    user input to lowercase.  This helper ensures the lookup map uses lowercase
    keys so that the two sides always agree.
    """
    return {k.lower(): v for k, v in (raw_map or {}).items()}
