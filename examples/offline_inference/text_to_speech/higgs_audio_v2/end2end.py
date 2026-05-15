# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end offline higgs-audio v2 inference (Stage 0 -> Stage 1) for vllm-omni.

This example exercises the vllm-omni higgs_audio_v2 path on a clean machine
following the layout from ``add-tts-model`` SKILL.md. Until the Stage-0
talker integrates the AR hot loop (UPSTREAM_TRACE.md + reference fixtures),
the example exposes two modes:

  --mode hf_reference   Run the upstream HF reference (downloads boson-ai
                        checkpoints; writes a 24 kHz WAV). Use this for
                        smoke-testing on a clean machine.
  --mode stage1_only    Load a saved fixture (``tests/fixtures/higgs_audio_v2/
                        reference_<slug>.pt``) and replay its [8, T] code
                        tensor through the vllm-omni Stage-1 decoder
                        (``HiggsAudioV2Code2Wav``). This validates AC-4 end
                        to end without requiring the 3B Stage-0 talker.

Both modes share command-line flags. ``--mode stage1_only`` requires a
fixture and a path to the local audio-tokenizer checkpoint dir
(``--audio-tokenizer-dir``, defaulting to the cached
``bosonai/higgs-audio-v2-tokenizer/audio_tokenizer/`` location).

Usage examples:

  # End-to-end via the upstream HF reference (downloads 5.8B model)
  python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
      --mode hf_reference --text "Hello world." --output-wav hello.wav

  # Stage-1 only: replay fixture codes through vllm-omni HiggsAudioV2Code2Wav
  python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
      --mode stage1_only \
      --fixture tests/fixtures/higgs_audio_v2/reference_hello_world.pt \
      --audio-tokenizer-dir ~/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/<rev>/audio_tokenizer \
      --output-wav stage1_replay.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _save_wav(path: str, pcm: torch.Tensor, sample_rate: int) -> None:
    import soundfile as sf

    if pcm.dtype != torch.int16:
        pcm_float = pcm.detach().to(torch.float32)
        if pcm_float.abs().max() <= 1.0:
            pcm_int16 = (pcm_float.clamp_(-1.0, 1.0) * 32767.0).round().to(torch.int16)
        else:
            pcm_int16 = pcm_float.to(torch.int16)
    else:
        pcm_int16 = pcm
    pcm_int16 = pcm_int16.reshape(-1).cpu().numpy()
    sf.write(path, pcm_int16, sample_rate, subtype="PCM_16")
    print(f"[end2end] wrote {path} ({pcm_int16.shape[0]} samples @ {sample_rate} Hz)")


def run_hf_reference(args: argparse.Namespace) -> int:
    script = Path(__file__).with_name("reference_hf.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("reference_hf", script)
    if spec is None or spec.loader is None:
        print(f"could not load {script}", file=sys.stderr)
        return 1
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    cap = module.capture_prompt(processor, model, args.text, args.max_new_tokens)
    _save_wav(args.output_wav, cap.reference_pcm, sample_rate=24000)
    return 0


def _resolve_audio_tokenizer_dir(explicit: str | None) -> Path:
    """Locate the boson-ai audio_tokenizer subdir for Stage-1 weight loading.

    Resolution order:
    1. Explicit ``--audio-tokenizer-dir`` argument.
    2. ``HIGGS_AUDIO_V2_TOKENIZER_DIR`` env var.
    3. HF hub snapshot under ``$HF_HOME`` (or ``~/.cache/huggingface``) for
       ``models--bosonai--higgs-audio-v2-tokenizer``; we resolve via
       ``huggingface_hub.snapshot_download`` which auto-uses any cached copy.

    Raises ``FileNotFoundError`` if no path can be resolved.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if (path / "config.json").exists():
            return path
        if (path / "audio_tokenizer" / "config.json").exists():
            return path / "audio_tokenizer"
        raise FileNotFoundError(
            f"--audio-tokenizer-dir {explicit!r} does not contain a config.json "
            "or an audio_tokenizer subdir"
        )
    env_dir = os.environ.get("HIGGS_AUDIO_V2_TOKENIZER_DIR")
    if env_dir:
        return _resolve_audio_tokenizer_dir(env_dir)

    # Try HF cache via snapshot_download (no-op when already cached).
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download("bosonai/higgs-audio-v2-tokenizer"))
    candidate = snapshot / "audio_tokenizer"
    if candidate.exists():
        return candidate
    if (snapshot / "config.json").exists():
        return snapshot
    raise FileNotFoundError(
        f"Could not locate audio_tokenizer files under {snapshot}"
    )


def run_stage1_only(args: argparse.Namespace) -> int:
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"fixture not found: {fixture_path}", file=sys.stderr)
        return 1
    blob = torch.load(fixture_path, weights_only=False)
    # ``audio_codes`` in the fixture is canonical [1, num_codebooks=8, T] since
    # round 2. Tolerate the legacy [1, T, 8] layout for older fixtures.
    codes = blob["audio_codes"].long()
    if codes.ndim != 3:
        print(f"unexpected audio_codes shape: {tuple(codes.shape)}", file=sys.stderr)
        return 1
    if codes.shape[1] != 8 and codes.shape[2] == 8:
        codes = codes.transpose(1, 2).contiguous()
    if codes.shape[1] != 8:
        print(
            f"audio_codes second dim must be num_codebooks=8 after normalization; got "
            f"{tuple(codes.shape)}",
            file=sys.stderr,
        )
        return 1

    try:
        audio_tokenizer_dir = _resolve_audio_tokenizer_dir(args.audio_tokenizer_dir)
    except (FileNotFoundError, OSError) as exc:
        print(f"could not resolve audio tokenizer dir: {exc}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HiggsAudioV2Config()
    code2wav = HiggsAudioV2Code2Wav(config).to(device)
    # ``audio_tokenizer_subdir`` defaults to "" (root layout), so pass the
    # tokenizer dir directly. If the caller has overridden the config to use a
    # nested layout, the load_weights helper joins on the subdir name.
    code2wav.load_weights(model_dir=str(audio_tokenizer_dir), device=device)
    pcm = code2wav(codes.to(device))  # [B, 1, T*960]
    pcm = pcm.squeeze(0).squeeze(0)
    _save_wav(args.output_wav, pcm, sample_rate=int(config.sample_rate))

    if args.compare_with_reference:
        ref = blob["reference_pcm"].to(torch.float32) / 32767.0
        out = pcm.to(torch.float32).clamp_(-1.0, 1.0).cpu()
        min_len = min(int(ref.shape[0]), int(out.shape[0]))
        rms = ((ref[:min_len] - out[:min_len]) ** 2).mean().sqrt().item()
        print(f"[end2end] Stage-1 vs HF reference PCM RMS (normalized float): {rms:.3e}")
        if rms > 1e-4:
            print(
                f"[end2end] WARNING: RMS {rms:.3e} exceeds AC-4 threshold 1e-4",
                file=sys.stderr,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("hf_reference", "stage1_only"), default="stage1_only")
    parser.add_argument("--text", default="Hello world.")
    parser.add_argument("--model-id", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--fixture",
        type=str,
        default="tests/fixtures/higgs_audio_v2/reference_hello_world.pt",
        help="(stage1_only) path to a captured reference fixture .pt file",
    )
    parser.add_argument(
        "--audio-tokenizer-dir",
        type=str,
        default=None,
        help="(stage1_only) path to the audio_tokenizer subdir of the boson-ai checkpoint",
    )
    parser.add_argument("--output-wav", type=str, default="higgs_audio_v2_end2end.wav")
    parser.add_argument(
        "--compare-with-reference",
        action="store_true",
        help="(stage1_only) print AC-4-style RMS error between Stage-1 and the saved reference PCM",
    )
    args = parser.parse_args()

    if args.mode == "hf_reference":
        return run_hf_reference(args)
    return run_stage1_only(args)


if __name__ == "__main__":
    sys.exit(main())
