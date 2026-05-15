# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture an AUTHORITATIVE upstream PCM reference via boson-ai's own ServeEngine.

The fixtures committed at ``tests/fixtures/higgs_audio_v2/reference_*.pt``
include a ``reference_pcm`` tensor that was originally rendered by the
``transformers`` ``AutoProcessor.batch_decode`` path. That path loads the
audio_tokenizer from the SAME model.safetensors blob as the LM, but the
state-dict layout (``model.layers.*.*`` etc.) doesn't match the codec's
``acoustic_encoder`` / ``acoustic_decoder`` / ``quantizer.vq.*`` layout the
codec expects. The codec ends up randomly initialized and the decoded PCM
is silent noise (abs.max ~= 706, abs.mean ~= 398.7 regardless of prompt).

This script is the recipe for capturing a REAL upstream PCM reference using
boson-ai's own runtime path. It requires the upstream
``boson_multimodal`` package (or a vendored copy) to be importable; until
that lands the script bails out with a clear error and the AC-4 RMS test
stays ``xfail`` per the documented blocker.

Usage:

    pip install boson_multimodal  # or pull from https://github.com/boson-ai/higgs-audio

    python examples/offline_inference/text_to_speech/higgs_audio_v2/reference_serve_engine.py \\
        --prompts "Hello world." \\
        --output-dir tests/fixtures/higgs_audio_v2 \\
        --output-suffix _serve_engine

The output is written next to the existing reference_*.pt files with the
``--output-suffix`` appended to the slug (so the original fixtures remain
inspectable). Each .pt is structured to be a drop-in for the AC-4 positive
test once it flips from xfail to a real assertion.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch


def _slugify(text: str) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:48] or "prompt"


def _resolve_serve_engine():
    """Try to import boson_multimodal.serve.ServeEngine; otherwise None."""
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

        return HiggsAudioServeEngine
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompts", nargs="+", default=["Hello world."])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/higgs_audio_v2"),
    )
    parser.add_argument(
        "--output-suffix",
        default="_serve_engine",
        help="Appended to each reference_<slug>.pt before the extension so existing fixtures aren't clobbered.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ServeEngine = _resolve_serve_engine()
    if ServeEngine is None:
        print(
            "[ref_serve_engine] boson_multimodal.serve.serve_engine.HiggsAudioServeEngine not importable.\n"
            "Install via `pip install boson_multimodal` (or follow the upstream README at\n"
            "https://github.com/boson-ai/higgs-audio for the supported install path) and re-run.\n"
            "Until this script can run, the AC-4 positive RMS test stays xfail per the documented blocker.",
            file=sys.stderr,
        )
        return 2

    engine = ServeEngine(
        model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for prompt in args.prompts:
        # Boson's ServeEngine API: build a single-speaker ChatMLSample and call generate().
        from boson_multimodal.data_types import ChatMLSample, Message

        sample = ChatMLSample(
            messages=[
                Message(
                    role="system",
                    content="Generate audio following instruction.",
                ),
                Message(role="user", content=prompt),
            ],
        )
        out = engine.generate(
            chat_ml_sample=sample,
            max_new_tokens=int(args.max_new_tokens),
            temperature=0.0,
            top_k=1,  # greedy
        )
        # ``out.audio`` is the upstream-authoritative PCM at out.sampling_rate.
        audio = torch.as_tensor(out.audio).to(torch.float32).clamp_(-1.0, 1.0)
        pcm_int16 = (audio * 32767.0).round().to(torch.int16)
        slug = _slugify(prompt)
        out_path = args.output_dir / f"reference_{slug}{args.output_suffix}.pt"
        torch.save(
            {
                "prompt_text": prompt,
                "reference_pcm": pcm_int16,
                "sampling_rate": int(out.sampling_rate),
                "captured_via": "boson_multimodal.serve.serve_engine.HiggsAudioServeEngine",
            },
            out_path,
        )
        print(
            f"[ref_serve_engine] wrote {out_path} (samples={pcm_int16.shape[0]}, "
            f"sr={out.sampling_rate}, abs.max={int(pcm_int16.abs().max())})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
