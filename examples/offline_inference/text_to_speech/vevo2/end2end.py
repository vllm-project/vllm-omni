# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Vevo2 via vLLM-Omni.

Vevo2 (https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2)
is a unified controllable AR + flow-matching model for speech and singing
voice generation. This MVP integration runs zero-shot text-to-speech with
optional voice cloning via a timbre reference. SVS / voice-conversion /
melody-control modes are deferred to follow-up PRs (see #3391).

Output is mono WAV at 24 kHz.

Prerequisites
-------------
1. Clone Amphion and put it on PYTHONPATH; Vevo2 is not yet on PyPI:

       git clone https://github.com/open-mmlab/Amphion.git
       export PYTHONPATH=$PWD/Amphion:$PYTHONPATH
       pip install -r Amphion/models/svc/vevo2/requirements.txt

2. Download the checkpoint (CC BY-NC-ND 4.0 -- non-commercial only):

       hf download RMSnow/Vevo2 --local-dir ./ckpts/Vevo2

3. Initialise it once. The published repo ships no root ``config.json``, no
   root weight file and no root tokenizer files, so it cannot be loaded as
   downloaded. This script writes them and is idempotent, so it is safe to
   re-run (and repairs a half-initialised checkpoint in place):

       python examples/offline_inference/text_to_speech/vevo2/init_vevo2_checkpoint.py ./ckpts/Vevo2

Usage
-----
    python end2end.py \\
        --model ./ckpts/Vevo2 \\
        --text "Hello, this is Vevo2 speaking." \\
        --ref-audio /path/to/ref.wav \\
        --ref-text "Transcript of the reference clip."
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Prevent multiprocessing from re-importing CUDA in the wrong context.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402

# The bare ``RMSnow/Vevo2`` HF repo ships no root ``config.json`` and so
# cannot be loaded by id; the model must first be downloaded and prepared
# with ``init_vevo2_checkpoint.py`` (see the Prerequisites above). Default
# to that initialized local checkout instead of a path that always fails.
DEFAULT_MODEL = "./ckpts/Vevo2"


def build_request(
    text: str,
    ref_audio_path: str,
    ref_text: str = "",
    timbre_ref_path: str | None = None,
    seed: int | None = None,
    top_k: int = 25,
    top_p: float = 0.8,
    temperature: float = 1.0,
    flow_matching_steps: int = 32,
) -> dict:
    """Build an Omni request payload for Vevo2.

    The reference audio is forwarded as ``prompt_audio_path`` (a filesystem
    path); upstream's ``inference_ar_and_fm`` requires
    ``timbre_ref_wav_path`` to be a path on disk. ``ref_text`` is the
    transcription of the reference clip and conditions the AR LM's
    prosody.
    """
    additional: dict = {
        "text": [text],
        "prompt_audio_path": [str(ref_audio_path)],
        "ref_text": [ref_text],
        "top_k": [top_k],
        "top_p": [top_p],
        "temperature": [temperature],
        "flow_matching_steps": [flow_matching_steps],
    }
    if timbre_ref_path is not None:
        additional["timbre_ref_path"] = [str(timbre_ref_path)]
    if seed is not None:
        additional["seed"] = [seed]

    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 24000) -> None:
    """Write the model's mono waveform to ``path``."""
    audio_np = waveform.float().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    sf.write(path, audio_np, sample_rate)
    print(f"  Saved {path} ({audio_np.shape}, {sample_rate} Hz)")


def main(args) -> None:
    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing: {args.text!r}")
    print(f"  ref_audio:   {args.ref_audio}")
    if args.timbre_ref:
        print(f"  timbre_ref:  {args.timbre_ref}")
    if args.ref_text:
        print(f"  ref_text:    {args.ref_text!r}")
    inputs = build_request(
        text=args.text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text or "",
        timbre_ref_path=args.timbre_ref,
        seed=args.seed,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        flow_matching_steps=args.flow_matching_steps,
    )

    # Each item yielded by ``Omni.generate`` is itself an ``OmniRequestOutput``
    # (it subclasses ``RequestOutput``), so the generation content is read off
    # the object directly -- same shape as
    # ``examples/offline_inference/text_to_speech/gepard/end2end.py``.
    for i, req_output in enumerate(omni.generate(inputs, sampling_params)):
        for j, out in enumerate(req_output.outputs):
            mm = out.multimodal_output
            if mm is None:
                print(f"  [req {i}] No audio output.")
                continue
            # Vevo2 yields delta audio chunks; the engine concatenates
            # them at finish into a single ``audio`` tensor (key set by
            # ``_consolidate_multimodal_tensors`` in
            # vllm_omni/engine/output_processor.py). Defensively handle
            # the pre-consolidation list form too.
            audio = mm.get("audio")
            if audio is None:
                audio = mm.get("model_outputs")
            if audio is None:
                print(f"  [req {i}] No waveform in multimodal_output.")
                continue
            if isinstance(audio, list):
                non_empty = [c for c in audio if hasattr(c, "numel") and c.numel() > 0]
                if not non_empty:
                    print(f"  [req {i}] Empty waveform list.")
                    continue
                audio = torch.cat([c.reshape(-1) for c in non_empty], dim=0)
            sr_tensor = mm.get("sr")
            if isinstance(sr_tensor, list) and sr_tensor:
                sr_tensor = sr_tensor[-1]
            sr = int(sr_tensor.item()) if sr_tensor is not None and hasattr(sr_tensor, "item") else 24000
            out_path = str(output_dir / f"output_{i}_{j}.wav")
            save_audio(audio, out_path, sr)

    print("Done.")


def parse_args():
    parser = FlexibleArgumentParser(description="Vevo2 offline inference")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Local checkpoint path prepared by init_vevo2_checkpoint.py "
            "(default: ./ckpts/Vevo2). The bare RMSnow/Vevo2 repo id cannot "
            "be loaded directly -- see the Prerequisites in this file's docstring."
        ),
    )
    parser.add_argument(
        "--text",
        default="Hello, this is Vevo2 speaking.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Path to reference audio (used as both style and timbre reference unless --timbre-ref is set).",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript of --ref-audio (recommended; conditions the AR LM's prosody).",
    )
    parser.add_argument(
        "--timbre-ref",
        default=None,
        help="Optional separate timbre reference audio. Defaults to --ref-audio.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--flow-matching-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "vevo2_output",
        ),
        help="Directory for WAV outputs (default: ~/.cache/vevo2_output).",
    )
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Path to a deploy YAML; leave unset to auto-load vllm_omni/deploy/vevo2.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
