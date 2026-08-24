# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm.reasoning import ReasoningParserManager

# Step-Audio specific reasoning parser.
# Registered at import time so it is available before the engine
# resolves ``--reasoning-parser step_audio``.

_OMNI_REASONING_PARSERS = {
    "step_audio": (
        "vllm_omni.reasoning.step_audio_reasoning_parser",
        "StepAudioReasoningParser",
    ),
}


def register_omni_reasoning_parsers():
    """Register vllm-omni reasoning parsers with vLLM's ReasoningParserManager."""
    for name, (module_path, class_name) in _OMNI_REASONING_PARSERS.items():
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


register_omni_reasoning_parsers()
