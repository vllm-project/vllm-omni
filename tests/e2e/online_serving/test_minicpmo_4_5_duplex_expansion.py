# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Nightly lifecycle coverage for the MiniCPM-o 4.5 native-duplex API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    SOFT_INTERRUPT_SHA256,
    deploy_max_sessions,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
    validated_soft_interrupt_wav,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_lifecycle_probes,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_server_vad import run_server_vad_interrupt
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_soft_interrupt import (
    run_soft_interrupt,
)
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.omni]


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_admission_and_expiry_reaper(omni_server, tmp_path: Path) -> None:
    args = multi_session_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "admission_expiry",
        response_required=False,
    )
    args.sessions = 1
    args.disconnect_session_index = None
    args.takeover_session_index = None
    args.expire_session_index = 0
    args.verify_admission_limit = deploy_max_sessions()
    result = asyncio.run(run_lifecycle_probes(args))

    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
    assert result["expiry"]["ok"] is True
    assert result["expiry"]["error_code"] == "session_resume_expired"
    assert result["admission"]["ok"] is True
    assert result["admission"]["overflow_error_code"] == "resource_exhausted"


# CUDA-only for now: this contract asserts the model speaks while the user is
# still talking, which only holds when the duplex pipeline sustains real-time
# throughput. The current NPU stack runs several times slower than real time,
# so it never reaches a mid-stream decision point.
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_soft_interrupt(omni_server, tmp_path: Path) -> None:
    input_wav = validated_soft_interrupt_wav()
    result = asyncio.run(
        run_soft_interrupt(
            SimpleNamespace(
                url=realtime_url(omni_server),
                model=omni_server.model,
                input_wav=str(input_wav),
                ref_audio=str(resolve_ref_audio()),
                output_dir=str(tmp_path / "soft_interrupt"),
                summary_output=None,
                chunk_ms=200,
                timeout_s=180.0,
                require_audio=True,
                no_realtime_pacing=False,
                validation_mode="response-required",
                min_responses=2,
                min_audio_deltas_per_response=2,
                input_sha256=SOFT_INTERRUPT_SHA256,
                expect_followup_response_substring=None,
            )
        )
    )

    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
    assert result["error_count"] == 0
    assert result["response_lifecycle_ok"] is True
    assert result["response_audio_contract_ok"] is True
    assert result["followup_response_transcript_ok"] is True


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_server_vad_hard_interrupt(omni_server) -> None:
    result = asyncio.run(
        run_server_vad_interrupt(
            SimpleNamespace(
                url=realtime_url(omni_server),
                model=omni_server.model,
                input_wav=str(validated_input_wav()),
                interrupt_wav=str(validated_soft_interrupt_wav()),
                ref_audio=str(resolve_ref_audio()),
                chunk_ms=200,
                timeout_s=180.0,
            )
        )
    )
    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
