# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline PersonaPlex inference through the vllm-omni fullduplex backend.

This is the runnable end-to-end proof for the integration: it drives a user WAV
through :class:`PersonaPlexEngine` + :class:`PersonaPlexSession` (our wrappers
around the Moshi/Mimi lockstep loop) and writes the agent's audio + inner-monologue
text. It mirrors ``python -m moshi.offline`` but goes through our ``FrameStepper``
seam, so a green run validates that the session/adapter plumbing is faithful.

Prereqs: HF access to ``nvidia/personaplex-7b-v1`` (``export HF_TOKEN=...``). The
stack is moshi-free (native temporal + depformer + streaming Mimi via
``transformers``); decoding is greedy.

Example:
    python examples/offline_inference/personaplex/personaplex_offline.py \\
        --input-wav input_assistant.wav --output-wav out.wav \\
        --voice-prompt NATF2.pt --persona "You are a wise and friendly teacher."
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from vllm_omni.experimental.fullduplex.personaplex import (
    PersonaPlexConfig,
    PersonaPlexEngine,
    PersonaPlexSession,
)
from vllm_omni.experimental.fullduplex.personaplex.config import DEFAULT_PERSONA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wav", required=True, help="User-side input WAV")
    parser.add_argument("--output-wav", required=True, help="Agent audio output WAV")
    parser.add_argument("--output-text", default=None, help="Optional JSON for agent text")
    parser.add_argument("--voice-prompt", default="NATF2.pt", help="Voice clone (bundled .pt or .wav)")
    parser.add_argument("--persona", default=DEFAULT_PERSONA, help="System role text")
    parser.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    import sphn  # local import: only the runnable driver needs the audio I/O dep

    config = PersonaPlexConfig(
        hf_repo=args.hf_repo,
        voice_prompt=args.voice_prompt,
        persona=args.persona,
        device=args.device,
        cpu_offload=args.cpu_offload,
    )

    engine = PersonaPlexEngine(config).load()
    session = PersonaPlexSession(engine, config)
    session.open()

    # Load exactly like the reference (moshi.lm.load_audio): native read then
    # explicit resample, so greedy runs stay bit-faithful to `python -m moshi.offline`.
    user_pcm, src_sr = sphn.read(args.input_wav)
    user_pcm = sphn.resample(user_pcm, src_sample_rate=src_sr, dst_sample_rate=config.sample_rate)
    audio, text = session.run(np.asarray(user_pcm[0], dtype=np.float32))

    sphn.write_wav(args.output_wav, audio, config.sample_rate)
    print(f"wrote {audio.shape[0] / config.sample_rate:.2f}s of agent audio to {args.output_wav}")
    print(f"agent text: {text!r}")
    if args.output_text:
        with open(args.output_text, "w") as f:
            json.dump({"text": text}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
