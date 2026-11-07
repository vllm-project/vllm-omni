# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-weight Phonon generation and compiled CUDA codec validation."""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import FlowSampleRunner, MimiDecodeRunner
from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder
from vllm_omni.model_executor.models.phonon.phonon_talker import PhononFlowLM

pytestmark = [pytest.mark.advanced_model, pytest.mark.tts, pytest.mark.slow]


@pytest.mark.cpu
def test_official_checkpoint_streaming_smoke():
    checkpoint = os.environ.get("PHONON_CHECKPOINT_DIR") or "gradium/phonon-31m-en"
    import sentencepiece
    from safetensors.torch import load_file
    from transformers.utils.hub import cached_file

    from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder
    from vllm_omni.model_executor.models.phonon.phonon_talker import PhononFlowLM
    from vllm_omni.model_executor.models.phonon.prompt import build_prompt

    config = json.loads(Path(cached_file(checkpoint, "config.json")).read_text())
    weights = load_file(cached_file(checkpoint, "model.safetensors"))
    lm = PhononFlowLM(config).eval()
    lm.load_official_weights(weights)
    decoder = MimiStreamingDecoder(config["mimi"], 1).eval()
    decoder.load_official_weights(weights)
    voice = load_file(cached_file(checkpoint, "default-voice.safetensors"))["speaker_wavs"]
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=cached_file(checkpoint, "tokenizer.model"))
    slots = torch.tensor([0])
    caches = [[torch.empty(0, 8, 64), torch.empty(0, 8, 64)] for _ in lm.transformer.layers]
    torch.manual_seed(53)
    with torch.inference_mode():
        prompt = build_prompt(lm, tokenizer.encode("Hello world."), voice)
        hidden = lm.transformer(prompt, torch.arange(prompt.shape[0]), caches)
        audio = []
        latents = []
        countdown = None
        for step in range(128):
            latent, eos = lm.sample(hidden[-1:], torch.randn(1, 32) * 0.8**0.5)
            denormalized = (latent * lm.emb_std + lm.emb_mean).unsqueeze(-1)
            latents.append(denormalized)
            pcm = decoder(denormalized, slots, torch.tensor([step * decoder.upsample_ratio]))
            assert pcm.shape == (1, 1, 1920)
            assert torch.isfinite(pcm).all() and torch.isfinite(eos).all()
            audio.append(pcm.flatten())
            if countdown is None and eos.item() > config["eos_threshold"]:
                countdown = 3
            if countdown == 0:
                break
            if countdown is not None:
                countdown -= 1
            feedback = lm.input_linear(latent) + lm.conditioning()
            hidden = lm.transformer(feedback, torch.tensor([prompt.shape[0] + step]), caches)
        assert countdown == 0, "Smoke request did not finish its EOS tail"
        waveform = torch.cat(audio)
        assert waveform.square().mean() > 0
        decoder.reset_slots(slots)
        complete = decoder(torch.cat(latents, -1), slots, torch.tensor([0])).flatten()
        torch.testing.assert_close(complete, waveform, atol=2e-5, rtol=2e-4)
        decoder.reset_slots(slots)
        expected = decoder(latents[0], slots, torch.tensor([0]))
        expected_state = {name: value.clone() for name, value in decoder.named_buffers() if "scratch" not in name}
        decoder.reset_slots(slots)
        compiled = torch.compile(decoder.forward, backend="eager", fullgraph=True, dynamic=False)
        actual = compiled(latents[0], slots, torch.tensor([0]))
        torch.testing.assert_close(actual, expected)
        for name, value in decoder.named_buffers():
            if "scratch" not in name:
                torch.testing.assert_close(value, expected_state[name])
        decoder.reset_slots(slots)
        reference = [
            decoder(latent, slots, torch.tensor([step * decoder.upsample_ratio]))
            for step, latent in enumerate(latents[:2])
        ]
        for name, buffer in decoder.named_buffers():
            if name.endswith(("history", "key_pool", "value_pool")):
                buffer.fill_(float("nan"))
        for step, latent in enumerate(latents[:2]):
            reused = compiled(latent, slots, torch.tensor([step * decoder.upsample_ratio]))
            torch.testing.assert_close(reused, reference[step])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_stream_handoff_and_slot_reset():
    torch.manual_seed(53)
    # Allow one PCM16 quantization step for compiled/eager waveform rounding.
    waveform_atol = 1.0 / 32768
    checkpoint = os.environ.get("PHONON_CHECKPOINT_DIR") or "gradium/phonon-31m-en"
    from safetensors.torch import load_file
    from transformers.utils.hub import cached_file

    config = json.loads(Path(cached_file(checkpoint, "config.json")).read_text())
    weights = load_file(cached_file(checkpoint, "model.safetensors"))
    eager = MimiStreamingDecoder(config["mimi"], 2).cuda().eval()
    captured = MimiStreamingDecoder(config["mimi"], 2).cuda().eval()
    eager.load_official_weights(weights)
    captured.load_official_weights(weights)
    lm = PhononFlowLM(config).cuda().eval()
    lm.load_official_weights(weights)
    flow = FlowSampleRunner(lm)
    runner = MimiDecodeRunner(captured)
    slot = torch.tensor([1], device="cuda")
    first = None
    with torch.inference_mode():
        for step in range(4):
            hidden, noise = (
                torch.randn(1, 512, device="cuda"),
                torch.randn(1, 32, device="cuda"),
            )
            latent, eos = flow.sample(hidden, noise)
            reference, expected_eos = lm.sample(hidden, noise)
            torch.testing.assert_close(latent, reference)
            torch.testing.assert_close(eos, expected_eos)
            codec_latent = (latent * lm.emb_std + lm.emb_mean).unsqueeze(-1)
            positions = torch.tensor([step * eager.upsample_ratio], device="cuda")
            expected = eager(codec_latent, slot, positions)
            actual, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, expected, atol=waveform_atol, rtol=2e-4)
            if first is None:
                first = actual
                saved = first.clone()
        torch.testing.assert_close(first, saved)
        tail_valid = torch.ones(1, dtype=torch.bool, device="cuda")
        tail_positions = positions + eager.upsample_ratio
        old_tail, tail_event = runner.decode(codec_latent, slot, tail_positions)
        torch.cuda.current_stream().wait_event(tail_event)
        saved_tail = old_tail.clone()

        # Compare reuse with the same compiled graph to isolate stale-state handling.
        runner.reset_slots(1)
        positions.zero_()
        cleared_outputs = []
        for step in range(2):
            reference, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            cleared_outputs.append(reference)
            positions.add_(eager.upsample_ratio)
        with torch.cuda.stream(runner.stream):
            for name, buffer in captured.named_buffers():
                if name.endswith(("history", "key_pool", "value_pool")):
                    buffer.fill_(float("nan"))
        positions.zero_()
        for reference in cleared_outputs:
            actual, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, reference, atol=waveform_atol, rtol=2e-4)
            positions.add_(eager.upsample_ratio)

        # Deliver the old request's final PCM after slot reset and a new decode.
        from vllm_omni.model_executor.models.phonon.model_state import PhononAudioOutput

        copy_stream = torch.cuda.Stream()
        snapshot = PhononAudioOutput(old_tail, tail_valid, [0], 1, tail_event)

        def copy_tensor(tensor):
            host = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            tensor.record_stream(copy_stream)
            return host

        with torch.cuda.stream(copy_stream):
            host_snapshot = snapshot.to_cpu(copy_stream, copy_tensor)
            copied = torch.cuda.Event()
            copied.record(copy_stream)
        copied.synchronize()
        delivered = host_snapshot.get_output()[0]["model_outputs"]
        torch.testing.assert_close(delivered, saved_tail.cpu().flatten(), atol=0, rtol=0)
