# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-weight regression: force a final one-token prefill chunk, then decode."""

import json
import os
from pathlib import Path

import pytest
import soundfile as sf
import torch
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.repo_utils import hf_api

pytestmark = [pytest.mark.advanced_model, pytest.mark.tts]

# Retain the existing localized and aggregate ROCm waveform bounds.
_ROCM_MAX_AUDIO_ABS_DIFF = 5.0e-2
_ROCM_MIN_AUDIO_SNR_DB = 30.0


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.asyncio
async def test_one_token_prefill_tail_matches_unchunked_audio(tmp_path, monkeypatch):
    # Greedy sampling alone does not make different prefill shapes numerically
    # identical. Use invariant kernels for the exact codes/waveform comparison.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    model = str(Path(os.environ.get("MODEL_PREFIX", "")) / "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    # Resolve the weights once, then use the same local snapshot for both
    # engines. A Hub repo ID makes each stage query remote metadata again,
    # which can hit the shared CI token's API rate limit during initialization.
    if not Path(model).is_dir():
        model = hf_api().snapshot_download(repo_id=model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = Qwen3TTSConfig.from_pretrained(model)
    info = {
        "task_type": ["CustomVoice"],
        "text": ["Hello, this is a test of speech generation."],
        "language": ["English"],
        "speaker": ["vivian"],
        "max_new_tokens": [128],
    }
    prompt_len = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        info,
        task_type="CustomVoice",
        tokenize_prompt=lambda text: tokenizer(text)["input_ids"],
        codec_language_id=config.talker_config.codec_language_id,
        spk_is_dialect=config.talker_config.spk_is_dialect,
    )
    assert prompt_len > 2
    prompt = {"prompt_token_ids": [0] * prompt_len, "additional_information": info}
    observations = []
    audios = []
    code2wav_updates: dict[str, bool | str] = {"enforce_eager": True, "async_scheduling": False}
    if current_omni_platform.is_rocm():
        # Independent BF16 decoders can exceed the waveform bounds for identical
        # codec tokens. Use FP32 here to isolate prefill parity from that drift;
        # the talker keeps its deployment dtype and exact-token assertions.
        code2wav_updates["dtype"] = "float32"
    for chunked in (False, True):
        deploy = modify_stage_config(
            get_deploy_config_path("qwen3_tts.yaml"),
            updates={
                "async_chunk": False,
                "stages": {
                    0: {
                        "enable_chunked_prefill": chunked,
                        "enable_prefix_caching": False,
                        "max_num_batched_tokens": prompt_len - 1 if chunked else 2048,
                        "max_model_len": 2048,
                        "max_num_seqs": 1,
                        "enforce_eager": True,
                        "async_scheduling": False,
                        "default_sampling_params.temperature": 0.0,
                        "default_sampling_params.max_tokens": 128,
                        "subtalker_sampling_params": {"do_sample": False},
                    },
                    1: code2wav_updates,
                },
            },
        )
        engine = AsyncOmni(
            model=model,
            deploy_config=deploy,
            worker_extension_cls=("tests.e2e.features.prefill_phase.worker_extension.PhaseContractWorkerExtension"),
            stage_init_timeout=600,
            init_timeout=900,
        )
        try:
            assert await engine.collective_rpc("start_phase_contract_probe", stage_ids=[0]) == [[True]]
            audio_parts = []
            async for output in engine.generate(prompt, request_id=f"phase-{chunked}"):
                if output.final_output_type == "audio":
                    multimodal = output.outputs[0].multimodal_output
                    assert (torch.as_tensor(multimodal["sr"]) == 24000).all()
                    audio = multimodal["audio"]
                    audio_parts.extend(audio if isinstance(audio, list) else [audio])
            assert audio_parts, "pipeline returned no audio"
            waveform = torch.cat([part.detach().cpu().reshape(-1) for part in audio_parts])
            assert waveform.numel() > 0 and torch.isfinite(waveform).all()
            assert waveform.abs().max() > 0
            sf.write(tmp_path / f"chunked-{chunked}.wav", waveform.float().numpy(), 24000)
            audios.append(waveform)
            probe = await engine.collective_rpc("get_phase_contract_probe", stage_ids=[0])
            observations.append(probe[0][0])
        finally:
            engine.shutdown()

    baseline, chunked_probe = observations
    same_audio_shape = audios[1].shape == audios[0].shape
    if same_audio_shape:
        audio_delta = audios[1].float() - audios[0].float()
        audio_noise_power = audio_delta.square().mean()
        audio_signal_power = audios[0].float().square().mean()
        audio_snr_db = float(
            10
            * torch.log10(audio_signal_power / audio_noise_power.clamp_min(torch.finfo(audio_noise_power.dtype).tiny))
        )
        max_audio_difference = float(audio_delta.abs().max())
    else:
        audio_snr_db = None
        max_audio_difference = None
    evidence = {
        "prompt_len": prompt_len,
        "code2wav_dtype_override": code2wav_updates.get("dtype"),
        "observations": observations,
        "audio_samples": audios[0].numel(),
        "chunked_audio_samples": audios[1].numel(),
        "max_audio_difference": max_audio_difference,
        "audio_snr_db": audio_snr_db,
        "rocm_max_audio_difference_limit": _ROCM_MAX_AUDIO_ABS_DIFF,
        "rocm_min_audio_snr_db": _ROCM_MIN_AUDIO_SNR_DB,
    }
    evidence_path = tmp_path / "phase_contract.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    print(f"Phase contract evidence: {evidence_path}")

    # Preserve both observations even when a parity assertion fails in CI.
    assert [prompt_len, 0, prompt_len, True] in baseline["events"]
    assert [prompt_len, 0, prompt_len - 1, True] in chunked_probe["events"]
    assert [prompt_len, prompt_len - 1, 1, True] in chunked_probe["events"]
    assert baseline["codes"] and chunked_probe["codes"]
    assert chunked_probe["codes"] == baseline["codes"]
    assert same_audio_shape
    if current_omni_platform.is_rocm():
        # Exact codes are the prefill contract. Retain both waveform bounds
        # with FP32 decoding so localized and aggregate audio regressions fail.
        assert max_audio_difference is not None
        assert max_audio_difference < _ROCM_MAX_AUDIO_ABS_DIFF, (
            f"ROCm waveform max difference {max_audio_difference:.8f} exceeds "
            f"{_ROCM_MAX_AUDIO_ABS_DIFF:.8f} despite exact generated codes"
        )
        assert audio_snr_db is not None
        assert audio_snr_db > _ROCM_MIN_AUDIO_SNR_DB, (
            f"ROCm waveform SNR {audio_snr_db:.2f} dB is not above "
            f"{_ROCM_MIN_AUDIO_SNR_DB:.2f} dB despite exact generated codes"
        )
    else:
        torch.testing.assert_close(audios[1], audios[0], rtol=1e-4, atol=1e-5)
