# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B offline end-to-end text-to-music.

One song is one or two engine requests against the single-stage pipeline:
``cot=full|melody`` first asks the model to write an ABC score (the abc phase,
tokens only), then the semantic phase generates codec tokens and the model
solves the acoustic ODE + decodes 48 kHz stereo as the request finishes.
``cot=off`` and a user-supplied ``--abc-file`` skip the first request.
Prefix caching is disabled; each request prefills its own prompt.

Example:
    python end2end.py --model /models/YuE2-3B --vae /models/YuE2-Vae \
        --style "gentle lo-fi hip-hop, 85 BPM, F minor" \
        --lyrics "[Verse]\nHumming on a quiet street\n[Chorus]\nStay a while" \
        --cot full --seed 831001 --max-frames 200 --output song.wav
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.yue2.yue2 import (
    ABC_SAMPLING,
    CONTEXT,
    KEY_MAX_AUDIO_FRAMES,
    KEY_MAX_HOLD_STEPS,
    KEY_MIN_TOKENS,
    KEY_PENALTY_WINDOW,
    KEY_PHASE,
    KEY_PREFIX_IDS,
    KEY_REPETITION_PENALTY,
    KEY_SEED,
    KEY_SKIP_SYNTHESIS,
    KEY_TEMPERATURE,
    KEY_TOP_K,
    KEY_TOP_P,
    SEMANTIC_SAMPLING,
    STOP_TOKEN_IDS,
)
from vllm_omni.tokenizers.yue2_prompt import (
    abc_ids_from_generated,
    abc_prefix_ids,
    semantic_frames,
    semantic_prefix_ids,
)
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer


def sampling_params(engine, *, phase, seed, max_frames, prompt_ids):
    """Deep-copy the stage defaults, then set the yue2_* keys for one phase."""
    params = copy.deepcopy(engine.resolve_sampling_params_list(None))[0]
    preset = ABC_SAMPLING if phase == "abc" else SEMANTIC_SAMPLING
    params.extra_args = {
        **(params.extra_args or {}),
        KEY_PHASE: phase,
        KEY_SEED: seed,
        KEY_TEMPERATURE: preset["temperature"],
        KEY_TOP_P: preset["top_p"],
        KEY_TOP_K: preset["top_k"],
        KEY_REPETITION_PENALTY: preset["repetition_penalty"],
        KEY_PENALTY_WINDOW: preset["penalty_window"],
        KEY_MIN_TOKENS: preset["min_tokens"],
        KEY_MAX_AUDIO_FRAMES: max_frames,
        KEY_SKIP_SYNTHESIS: phase == "abc",
        # NAR conditioning needs the full prompt ids.
        KEY_PREFIX_IDS: list(prompt_ids),
    }
    if phase == "abc":
        params.max_tokens = preset["max_tokens"]
    else:
        # Acoustic completion no longer consumes AR tokens.
        params.max_tokens = min(max_frames + 2, CONTEXT - len(prompt_ids))
        params.extra_args[KEY_MAX_HOLD_STEPS] = 0
    params.stop_token_ids = list(STOP_TOKEN_IDS)
    params.detokenize = False
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="m-a-p/YuE2-3B checkpoint dir or repo id")
    parser.add_argument("--revision", default=None, help="main model and tokenizer revision")
    parser.add_argument("--vae", default=None, help="m-a-p/YuE2-Vae directory or repo id")
    parser.add_argument("--vae-revision", default=None, help="independent VAE revision")
    parser.add_argument("--deploy-config", default=None, help="deployment YAML")
    parser.add_argument("--style", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--cot", choices=["off", "melody", "full"], default="full")
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--max-frames", type=int, default=200, help="semantic frame budget (25 frames = 1 s)")
    parser.add_argument("--abc-file", default=None, help="external ABC score (requires cot=melody/full)")
    parser.add_argument("--output", default="yue2_song.wav")
    parser.add_argument(
        "--dump-tokens",
        default=None,
        metavar="PATH",
        help="write prompt + generated token ids to this json (default: <output>.tokens.json)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="override the deploy yaml's per-stage gpu_memory_utilization "
        "(necessary on cards shared with other tenants: the yaml fraction is "
        "of the WHOLE card, not of the free remainder)",
    )
    args = parser.parse_args()

    vae_overrides = {}
    if args.vae_revision:
        vae_overrides["yue2_vae_revision"] = args.vae_revision
    if args.vae:
        vae_overrides["yue2_vae_path"] = args.vae
    engine_kwargs: dict = {
        "revision": args.revision,
        "stage_overrides": {"0": {"hf_overrides": vae_overrides}},
    }
    if args.deploy_config:
        engine_kwargs["deploy_config"] = args.deploy_config
    if args.gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    engine = Omni(model=args.model, **engine_kwargs)
    vllm_config = engine.engine.stage_vllm_configs[0]
    config = vllm_config.model_config
    tokenizer = YuE2TextTokenizer.from_pretrained(
        config.model,
        revision=config.revision,
        token=config.hf_token,
        cache_dir=vllm_config.load_config.download_dir,
    )

    abc_ids = None
    if args.abc_file is not None:
        if args.cot == "off":
            raise SystemExit("--abc-file requires --cot melody|full")
        abc_ids = tokenizer.encode(Path(args.abc_file).read_text())
        print(f"Using supplied ABC score ({len(abc_ids)} tokens)")
    elif args.cot != "off":
        prompt_ids = abc_prefix_ids(tokenizer.encode, args.style, args.lyrics, args.cot)
        prompt = {"prompt_token_ids": prompt_ids}
        params = sampling_params(engine, phase="abc", seed=args.seed, max_frames=args.max_frames, prompt_ids=prompt_ids)
        outputs = engine.generate([prompt], [params])
        generated = list(outputs[0].outputs[0].token_ids)
        abc_ids = abc_ids_from_generated(generated)
        print(f"Generated ABC score ({len(abc_ids)} tokens):\n{tokenizer.decode(abc_ids)}\n")

    prompt_ids = semantic_prefix_ids(tokenizer.encode, args.style, args.lyrics, args.cot, abc_ids=abc_ids)
    prompt = {"prompt_token_ids": prompt_ids}
    params = sampling_params(
        engine, phase="semantic", seed=args.seed, max_frames=args.max_frames, prompt_ids=prompt_ids
    )
    outputs = engine.generate([prompt], [params])
    output = outputs[0].outputs[0]
    generated_ids = list(output.token_ids)
    if args.dump_tokens:
        import json

        payload = {
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated_ids,
            # Codec frames only: without the end token and the hold tokens
            # the request emits while its song is synthesized.
            "semantic_frames": semantic_frames(generated_ids),
            "seed": args.seed,
            "cot": args.cot,
            "max_frames": args.max_frames,
        }
        dump_path = args.dump_tokens if args.dump_tokens.endswith(".json") else args.output + ".tokens.json"
        Path(dump_path).write_text(json.dumps(payload))
        print(f"Dumped {len(generated_ids)} generated token ids to {dump_path}")
    mm = output.multimodal_output or {}
    audio = mm.get("audio")
    if audio is None and "model_outputs" in mm:
        audio = mm["model_outputs"]
    if audio is None:
        raise SystemExit("No audio in the model output; check server logs")
    sr = int(mm.get("sr").item()) if mm.get("sr") is not None else 48000
    truncated = False
    meta = mm.get("meta") or {}
    if "truncated" in meta:
        truncated = bool(torch.as_tensor(meta["truncated"]).reshape(-1)[-1].item())

    waveform = audio.float()
    # soundfile wants [frames, channels] float in [-1, 1]; torchaudio has no
    # wheel matching vllm 0.29.0's torch pin, so the driver uses libsndfile.
    sf.write(args.output, waveform.T.contiguous(), sr)
    print(
        f"Saved {args.output}: {waveform.shape[-1] / sr:.1f}s @ {sr} Hz stereo, "
        f"truncated={truncated}, generated {len(output.token_ids)} semantic tokens"
    )


if __name__ == "__main__":
    main()
