# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build native song inputs; no engine calls or inference orchestration."""

import math

from vllm import SamplingParams

from vllm_omni.tokenizers.yue2_prompt import INSTRUCTIONS, abc_prefix_ids, semantic_prefix_ids

from .yue2 import (
    ABC_END,
    ABC_SAMPLING,
    ABC_START,
    CONTEXT,
    DEFAULT_SEED,
    EOD,
    MUSIC_START,
    SEMANTIC_SAMPLING,
    STOP_TOKEN_IDS,
)

AUTO_ABC = "yue2_auto_abc"
SEMANTIC_ARGS = "yue2_semantic_args"


def phase_params(prefix, phase, seed, budget):
    preset = ABC_SAMPLING if phase == "abc" else SEMANTIC_SAMPLING
    return SamplingParams(
        max_tokens=budget + (2 if phase == "semantic" else 0),
        detokenize=False,
        stop_token_ids=list(STOP_TOKEN_IDS),
        extra_args={
            "yue2_phase": phase,
            "yue2_seed": seed,
            "yue2_prefix_ids": list(prefix),
            "yue2_skip_synthesis": phase == "abc",
            "yue2_max_audio_frames": budget,
            "yue2_max_hold_steps": 0,
            **{
                f"yue2_{key}": preset[key]
                for key in ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window", "min_tokens")
            },
        },
    )


def prepare_song(
    encode,
    style,
    lyrics,
    *,
    cot="off",
    abc=None,
    seed=DEFAULT_SEED,
    max_frames=200,
    max_abc_tokens=4096,
    max_model_len=CONTEXT,
    cfg_scale=1.0,
):
    """Return (token prompt, SamplingParams) for one native YuE2 request.

    Missing ABC with cot=melody/full enables automatic ABC in the same native
    request. The semantic phase restarts its generator at the same seed, as
    in the upstream two-call example. External ABC bypasses generation.
    """
    if cot not in ("off", "melody", "full"):
        raise ValueError("cot must be off, melody or full")
    if not isinstance(style, str) or not style.strip() or not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("YuE2 requires nonempty style and lyrics")
    if abc is not None and (cot == "off" or not isinstance(abc, str) or not abc.strip()):
        raise ValueError("A nonempty ABC score requires cot=melody/full")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    for value, maximum, name in (
        (max_frames, SEMANTIC_SAMPLING["max_tokens"], "max_frames"),
        (max_abc_tokens, ABC_SAMPLING["max_tokens"], "max_abc_tokens"),
        (max_model_len, CONTEXT, "max_model_len"),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    if (
        isinstance(cfg_scale, bool)
        or not isinstance(cfg_scale, (int, float))
        or not math.isfinite(cfg_scale)
        or not 0 <= cfg_scale <= 20
    ):
        raise ValueError("cfg_scale must be finite and in [0,20]")
    automatic = cot != "off" and abc is None
    prefix = (
        abc_prefix_ids(encode, style, lyrics, cot)
        if automatic
        else semantic_prefix_ids(encode, style, lyrics, cot, None if abc is None else encode(abc))
    )
    required = len(prefix) + (max_abc_tokens + 2 if automatic else 0) + max_frames + 2
    if required > max_model_len:
        raise ValueError(f"YuE2 song budget needs {required} tokens, exceeding context {max_model_len}")
    params = phase_params(prefix, "abc" if automatic else "semantic", seed, max_abc_tokens if automatic else max_frames)
    if automatic:
        params.extra_args[AUTO_ABC] = True
        params.extra_args[SEMANTIC_ARGS] = phase_params([], "semantic", seed, max_frames).extra_args
    if cfg_scale != 1:
        semantic = params.extra_args[SEMANTIC_ARGS] if automatic else params.extra_args
        negative = [EOD] + encode(INSTRUCTIONS[cot])
        negative += [MUSIC_START] if cot == "off" else [ABC_START]
        if cot != "off" and not automatic:
            negative += encode(abc) + [ABC_END, MUSIC_START]
        semantic.update(yue2_cfg_scale=float(cfg_scale), yue2_negative_prefix=negative, yue2_cfg_needs_abc=automatic)
    return {"prompt_token_ids": prefix}, params
