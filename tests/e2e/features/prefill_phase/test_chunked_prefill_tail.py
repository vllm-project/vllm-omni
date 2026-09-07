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

pytestmark = [pytest.mark.advanced_model, pytest.mark.tts]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_one_token_prefill_tail_matches_unchunked_audio(tmp_path, monkeypatch):
    # Greedy sampling alone does not make different prefill shapes numerically
    # identical. Use invariant kernels for the exact codes/waveform comparison.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    model = str(Path(os.environ.get("MODEL_PREFIX", "")) / "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
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
                    1: {"enforce_eager": True, "async_scheduling": False},
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
    evidence = {
        "prompt_len": prompt_len,
        "observations": observations,
        "audio_samples": audios[0].numel(),
        "chunked_audio_samples": audios[1].numel(),
        "max_audio_difference": (
            float((audios[1] - audios[0]).abs().max()) if audios[1].shape == audios[0].shape else None
        ),
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
    torch.testing.assert_close(audios[1], audios[0], rtol=1e-4, atol=1e-5)
