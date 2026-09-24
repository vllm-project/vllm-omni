# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-resident full-duplex sessions: the merged session state, the per-session
runner, the session manager, the model plugin contract, and the engine's binding of
the wire protocol (``mailbox``, ``projection``). The typed commands and events
themselves are ``vllm_omni.protocol.duplex``."""
