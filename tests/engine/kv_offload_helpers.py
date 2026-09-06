# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pieces for the LMCache KV + hidden-state offload consistency tests.

Kept model-agnostic so each model gets its own test module with its own device
layout and memory budget.
"""

_FACTS = [
    "Mercury is the closest planet to the Sun and completes an orbit in about eighty-eight Earth days.",
    "The Pacific is the largest ocean on Earth and covers roughly a third of the planet's surface.",
    "Gold has the chemical symbol Au, taken from aurum, its name in Latin.",
    "A regular hexagon has six equal sides and tiles the plane without leaving any gaps.",
    "The Nile flows northward through Egypt and empties into the Mediterranean Sea.",
    "Diamond is the hardest natural mineral and sits at the top of the Mohs scale.",
    "Venus rotates in the opposite direction to most planets, so the Sun there rises in the west.",
    "The Amazon carries more water than any other river and drains a basin shared by several countries.",
    "Helium is lighter than air, which is why a balloon filled with it floats upward.",
    "An octave spans eight notes and doubles the frequency between its first and last pitch.",
    "Antarctica is the driest continent and most of it receives less precipitation than a desert.",
    "Copper conducts electricity better than iron, which is why household wiring uses it.",
    "The Sahara is the largest hot desert and stretches across much of northern Africa.",
    "Water boils at one hundred degrees Celsius at sea level, and lower where the pressure drops.",
    "A leap year has three hundred and sixty-six days because February gains an extra one.",
    "Everest is the highest mountain above sea level and sits on the border of Nepal and Tibet.",
    "Silver is the most reflective metal, which makes it useful for mirrors and telescopes.",
    "The Dead Sea lies below sea level and is salty enough that swimmers float without effort.",
    "Bamboo is the fastest growing plant and some species gain most of a metre in a day.",
    "Neon glows red in a discharge tube, which is where the classic sign colour comes from.",
    "The Baltic is the least salty sea because so many rivers empty fresh water into it.",
    "Graphite and diamond are both pure carbon and differ only in how their atoms are arranged.",
    "Jupiter has the shortest day of the planets and turns once in under ten hours.",
    "Mount Fuji is the highest peak in Japan and last erupted in the early eighteenth century.",
]

# Long enough to span several LMCache chunks so a cache hit actually skips
# prefill; shared by every prompt so later requests hit the cached prefix.
# The facts are deliberately unrelated: near-identical candidates leave the
# greedy argmax nearly tied, so any float-level difference flips the answer.
SHARED_PREFIX = " ".join(f"Fact {i}: {fact}" for i, fact in enumerate(_FACTS))

GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42, "max_tokens": 48}


def prompts(n: int = 3) -> list[dict]:
    return [
        {
            "prompt": (
                f"<|im_start|>user\n{SHARED_PREFIX}\nRepeat Fact {i} above word for "
                f"word.<|im_end|>\n<|im_start|>assistant\n"
            )
        }
        for i in range(n)
    ]


def stage_overrides(
    *,
    lmcache: bool,
    prefix_caching: bool,
    hidden_states: bool = True,
    thinker_extra: dict | None = None,
    downstream_extra: dict[str, dict] | None = None,
    lmcache_extra: dict | None = None,
) -> dict:
    """Patch the thinker; keep the default talker/code2wav stages so audio runs.

    ``thinker_extra`` and ``downstream_extra`` carry the per-model device layout
    and memory budget. The talker defaults to temperature 0.9, which amplifies
    any float-level difference into a different audio sequence, so every stage
    is pinned to greedy here.
    """
    # async_chunk and enforce_eager are left to the caller: Qwen3-Omni hands off
    # to the talker through an async-chunk-specific processor, so forcing them
    # breaks the handoff.
    thinker: dict = {
        "max_num_seqs": 4,
        "enable_prefix_caching": prefix_caching,
        "default_sampling_params": dict(GREEDY),
        **(thinker_extra or {}),
    }
    if lmcache:
        lmcache_config: dict = {"config_file": "", **(lmcache_extra or {})}
        if not hidden_states:
            lmcache_config["enable_hidden_state_cache"] = False
        thinker["omni_kv_config"] = {"kv_store_config": {"lmcache_config": lmcache_config}}

    overrides: dict = {"0": thinker}
    for stage_id, extra in (downstream_extra or {}).items():
        overrides[stage_id] = {"default_sampling_params": dict(GREEDY), **extra}
    return overrides


def collect(omni, request_prompts) -> dict[str, dict]:
    """Run one round, keyed by prompt -- request ids are per-engine and would
    pair one prompt's baseline output with another's cached output."""
    by_request: dict[str, dict] = {}
    prompt_of: dict[str, str] = {}
    for out in omni.generate(request_prompts, omni.default_sampling_params_list):
        entry = by_request.setdefault(out.request_id, {})
        if out.final_output_type == "text":
            entry["text"] = out.outputs[0].text
            if getattr(out, "prompt", None):
                prompt_of[out.request_id] = out.prompt
        elif out.final_output_type == "audio":
            audio = out.outputs[0].multimodal_output["audio"]
            entry["audio"] = audio.detach().cpu().float()

    results: dict[str, dict] = {}
    for rid, entry in by_request.items():
        key = prompt_of.get(rid)
        assert key is not None, f"no text output (and therefore no prompt) for request {rid}"
        results[key] = entry
    return results


def run_rounds(
    *, model: str, overrides: dict, rounds: int, init_timeout: int = 900, num_prompts: int = 3
) -> list[dict[str, dict]]:
    """Build an engine, run ``rounds`` identical rounds, return each one."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=model,
        stage_overrides=overrides,
        trust_remote_code=True,
        stage_init_timeout=init_timeout,
        batch_timeout=5,
        init_timeout=init_timeout,
    )
    try:
        request_prompts = prompts(num_prompts)
        return [collect(omni, request_prompts) for _ in range(rounds)]
    finally:
        omni.close()


def run(*, model: str, overrides: dict, rounds: int, init_timeout: int = 900) -> dict[str, dict]:
    """Build an engine, run ``rounds`` identical rounds, return the last one."""
    return run_rounds(model=model, overrides=overrides, rounds=rounds, init_timeout=init_timeout)[-1]


def audio_len(entry: dict) -> int:
    audio = entry.get("audio")
    return 0 if audio is None else int(audio.numel())


def compare(
    baseline: dict[str, dict],
    cached: dict[str, dict],
    *,
    expect_audio: bool = True,
    assert_waveform: bool = True,
) -> list[str]:
    """Return a problem per prompt.

    ``assert_waveform`` requires the waveforms to match sample for sample. That
    holds between two engines compared at the same round, which is what makes
    the audio check meaningful rather than a length comparison. It does not hold
    between an engine's first and second round: the first inference after
    startup carries JIT and autotuning state that the rest do not, and the
    talker's autoregressive decode amplifies it into a different waveform even
    when the text is identical.

    ``expect_audio=False`` inverts the audio check for the KV-only configuration:
    a cache hit skips the thinker's prefill, so with the hidden-state store off
    the talker is handed nothing and silence is the correct outcome. Asserting it
    rather than tolerating it is what makes that arm evidence for the
    hidden-state offload being load-bearing. It holds only when every request is
    served from the cache, which is the case when the same prompts are repeated.
    """
    problems = []
    for i, prompt in enumerate(sorted(baseline)):
        want, got = baseline[prompt], cached[prompt]
        if got["text"] != want["text"]:
            problems.append(f"prompt {i}: text differs\n  baseline={want['text']!r}\n  cached={got['text']!r}")

        want_len, got_len = audio_len(want), audio_len(got)
        if not expect_audio:
            if got_len:
                problems.append(f"prompt {i}: expected no audio without the hidden-state store, got {got_len} samples")
            continue
        if want_len and not got_len:
            problems.append(f"prompt {i}: baseline produced {want_len} audio samples, offload produced none")
        elif want_len != got_len:
            problems.append(f"prompt {i}: audio length {want_len} -> {got_len}")
        elif want_len:
            delta = (got["audio"] - want["audio"]).abs().max().item()
            print(f"prompt {i}: audio max|delta| = {delta:.3e}")
            if assert_waveform and delta != 0:
                problems.append(f"prompt {i}: audio differs, max|delta| = {delta:.3e}")
    return problems
