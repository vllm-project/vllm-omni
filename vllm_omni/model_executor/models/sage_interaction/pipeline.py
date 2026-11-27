# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""InternLM Sage-Interaction pipeline topology.

Reuses the MiniCPM-o 4.5 three-stage execution topology (Thinker -> Talker ->
Code2Wav) with Sage-Interaction's duplex plugin and checkpoint disambiguation.
"""

from __future__ import annotations

from dataclasses import replace

from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

SAGE_INTERACTION_VOCAB_SIZE = 151749


def is_sage_interaction_config(config: object) -> bool:
    """Distinguish InternLM/Sage-Interaction from stock MiniCPM-o 4.5 checkpoints.

    Both checkpoints report ``model_type="minicpmo"``, ``architectures=["MiniCPMO"]``,
    and ``version="4.5"``. Sage-Interaction extends the vocabulary by one special
    token (``<|delegate|>`` at id 151748), raising ``vocab_size`` from 151748 to
    151749.
    """
    if str(getattr(config, "version", "")) != "4.5":
        return False
    if getattr(config, "sage_interaction", False) is True:
        return True
    try:
        vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    except (TypeError, ValueError):
        return False
    return vocab_size == SAGE_INTERACTION_VOCAB_SIZE


SAGE_INTERACTION_PIPELINE = replace(
    MINICPMO_4_5_PIPELINE,
    model_type="sage_interaction",
    default_deploy_config_name="sage_interaction.yaml",
    duplex_plugin="vllm_omni.model_executor.models.sage_interaction.duplex.plugin.SageInteractionDuplexPlugin",
    hf_config_predicate=is_sage_interaction_config,
)

__all__ = [
    "SAGE_INTERACTION_PIPELINE",
    "SAGE_INTERACTION_VOCAB_SIZE",
    "is_sage_interaction_config",
]
