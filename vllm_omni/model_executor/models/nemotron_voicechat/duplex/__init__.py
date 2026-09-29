# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Full-Duplex integration for NVIDIA NemotronLabs VoiceChat.

The single ``duplex_plugin`` seam (RFC vllm-omni#7181):
``NemotronVoiceChatDuplexPlugin`` carries the engine and session policies,
``data_plane.py`` projects stage outputs and ``input.py`` packetizes 80 ms PCM
frames into the framework's per-session model state.
"""

from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import (
    NemotronVoiceChatDuplexPlugin,
)

__all__ = ["NemotronVoiceChatDuplexPlugin"]
