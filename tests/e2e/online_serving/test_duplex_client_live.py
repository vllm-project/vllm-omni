# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live E2E coverage for the public duplex client (``vllm_omni.clients.duplex``).

Drives a real MiniCPM-o 4.5 duplex session end to end through
:class:`DuplexClient`: session handshake with reference audio, paced PCM
streaming, a speak response consumed via :class:`ResponseHandle` with
incremental playback acks, a forced transport drop recovered by automatic
``session.resume``, and a clean close.

This requires real Thinker/Talker weights: core-model dummy weights cannot
produce a natural reply or its terminal decision.
"""

from __future__ import annotations

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    EAGER_SERVER_PARAMS,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import send_duplex_client_session_request

pytestmark = pytest.mark.omni


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", EAGER_SERVER_PARAMS, indirect=True)
def test_duplex_client_live_session(omni_server) -> None:
    send_duplex_client_session_request(
        server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
    )
