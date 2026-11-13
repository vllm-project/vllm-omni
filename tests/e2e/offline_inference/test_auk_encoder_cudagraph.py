# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK encoder stage: FULL CUDA graph replay matches eager, request by request.

``auk.yaml`` captures the encoder prefill as FULL CUDA graphs padded to a few
token buckets. A replay reuses static input buffers and the graph pool, so a
request could in principle read conditioning left behind by an earlier, longer
prompt. This test runs the same request sequence through an eager encoder and
through the graphed one and compares the text condition stage 0 hands to
stage 1 (the fused thinker hidden states) for every request. The sequence
alternates prompts of different lengths inside one capture bucket, crosses
into a larger bucket, and then returns to the first prompt.

Like ``test_auk.py`` it needs an assembled checkpoint directory:

    VLLM_OMNI_AUK_MODEL_DIR=/path/to/auk-omni python -m pytest tests/e2e/offline_inference/test_auk_encoder_cudagraph.py
"""

from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.stage_input_processors import auk as auk_input_processor
from vllm_omni.model_extras.auk import auk_prompt, auk_sampling_params

MODEL_DIR_ENV = "VLLM_OMNI_AUK_MODEL_DIR"

# Text-only instruct prompts: the token count, and so the capture bucket,
# follows the text length. SHORT_A and SHORT_B differ in length but share a
# bucket; LONG lands in a larger one. The test checks these properties on the
# recorded condition lengths, so a tokenizer change fails loudly instead of
# silently shrinking the coverage.
_TEMPLATE = 'Generate speech based on the following description: "{desc}". The content to speak is: "{text}".'
SHORT_A = _TEMPLATE.format(
    desc="A calm young woman speaking warmly.",
    text="Welcome back, how was your day? I kept some soup warm for you, and the kettle is on if you want tea.",
)
SHORT_B = _TEMPLATE.format(
    desc="An older man with a deep, slightly raspy voice, speaking slowly.",
    text=(
        "The train to the coast leaves at seven, so please be on time. Bring a warm coat, "
        "some cash for the ferry, and the blue umbrella from the hall, because the forecast says rain."
    ),
)
LONG = _TEMPLATE.format(
    desc=(
        "A cheerful radio host in her thirties, bright and energetic, with clear articulation, "
        "a slight smile in her voice, and a brisk but relaxed pace, recorded close to the microphone"
    ),
    text=(
        "Good morning and welcome to the weekend edition. Today we are talking about small gardens, "
        "the plants that thrive on a sunny balcony, and the neighbours who turned an empty lot into "
        "a place where the whole street now meets on Saturday afternoons to trade seedlings and stories. "
        "Stay with us, because after the news we will open the phone lines and hear from you."
    ),
)
SEQUENCE = [SHORT_A, SHORT_B, SHORT_A, LONG, SHORT_B, SHORT_A]

# The DiT output is not inspected; keep stage 1 as cheap as it goes (it also
# runs eager in both configs, see _deploy).
GEN_SECONDS = 1.0
NFE = 1

# bf16 hidden states. Graph replay pads the prefill to the bucket size but runs
# the same kernels on the real tokens, so the two paths agree to bf16 rounding.
# Measured on H20: cosine >= 0.99998 and max-abs error <= 2.9e-3 of the
# condition's max magnitude for every request; the bounds leave about 3x
# headroom for other GPUs.
MIN_COSINE = 0.9999
MAX_REL_ABS = 1e-2

_model_dir = os.environ.get(MODEL_DIR_ENV)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skipif(not _model_dir, reason=f"set {MODEL_DIR_ENV} to an assembled AuK directory"),
]


def _deploy(tmp_path: Path, *, graph_encoder: bool) -> str:
    """``auk.yaml`` with stage 1 eager, and the encoder stage eager or graphed as shipped.

    Only the encoder differs between the two runs. The conditions compared here
    are recorded before stage 1 runs, so its DiT compile, graph capture and
    codec bucket warm-up would only add startup time; eager stage 1 skips them
    and still takes the real stage handoff.
    """
    with open(get_deploy_config_path("auk.yaml")) as f:
        deploy = yaml.safe_load(f)
    for stage in deploy["stages"]:
        if stage["stage_id"] == 1:
            stage["enforce_eager"] = True
        elif not graph_encoder:
            stage["enforce_eager"] = True
            stage.pop("compilation_config", None)
    path = tmp_path / f"auk_{'graph' if graph_encoder else 'eager'}_encoder.yaml"
    path.write_text(yaml.safe_dump(deploy))
    return str(path)


def _graph_capture_sizes() -> list[int]:
    with open(get_deploy_config_path("auk.yaml")) as f:
        deploy = yaml.safe_load(f)
    stage = next(s for s in deploy["stages"] if s["stage_id"] == 0)
    assert stage.get("enforce_eager") is False, "auk.yaml no longer graphs the encoder; update this test"
    return sorted(stage["compilation_config"]["cudagraph_capture_sizes"])


def _run_sequence(deploy_config: str, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Run SEQUENCE one request at a time and return each request's text condition."""
    conditions: list[torch.Tensor] = []
    original = auk_input_processor.encoder2dit

    def recording_encoder2dit(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        result = original(*args, **kwargs)
        if result is not None:
            conditions.append(result["prompt_embeds"].detach().to("cpu", torch.float32).clone())
        return result

    # The orchestrator runs in this process and resolves the stage input
    # function by module attribute at startup, so the patch must precede Omni.
    with monkeypatch.context() as patch:
        patch.setattr(auk_input_processor, "encoder2dit", recording_encoder2dit)
        with OmniRunner(str(Path(_model_dir).resolve()), deploy_config=deploy_config) as runner:
            for text in SEQUENCE:
                prompt = auk_prompt(text, None, gen_seconds=GEN_SECONDS)
                outputs = runner.omni.generate(prompt, auk_sampling_params(nfe=NFE, cfg=2.0, seed=0))
                assert outputs, "no outputs returned"
    assert len(conditions) == len(SEQUENCE), f"recorded {len(conditions)} conditions for {len(SEQUENCE)} requests"
    return conditions


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_encoder_graph_replay_matches_eager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_sizes = _graph_capture_sizes()
    eager = _run_sequence(_deploy(tmp_path, graph_encoder=False), monkeypatch)
    graphed = _run_sequence(_deploy(tmp_path, graph_encoder=True), monkeypatch)

    lengths = {text: cond.shape[0] for text, cond in zip(SEQUENCE, eager)}
    for i, (ref, got) in enumerate(zip(eager, graphed)):
        assert got.shape == ref.shape, f"request {i}: shape {tuple(got.shape)} != eager {tuple(ref.shape)}"
        assert torch.isfinite(got).all(), f"request {i}: non-finite condition"
        cosine = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        rel_abs = ((got - ref).abs().max() / ref.abs().max()).item()
        assert cosine >= MIN_COSINE and rel_abs <= MAX_REL_ABS, (
            f"request {i} ({lengths[SEQUENCE[i]]} tokens): cosine {cosine:.6f}, max-abs error {rel_abs:.2e} "
            "of the condition's magnitude"
        )

    # The sequence must exercise what it claims to: two different lengths in
    # one bucket, a second bucket, and every prompt inside the graphed range.
    assert max(lengths.values()) <= capture_sizes[-1], f"{lengths} exceeds the largest capture size"
    bucket = {text: capture_sizes[bisect.bisect_left(capture_sizes, n)] for text, n in lengths.items()}
    assert lengths[SHORT_A] != lengths[SHORT_B] and bucket[SHORT_A] == bucket[SHORT_B], (lengths, bucket)
    assert bucket[LONG] > bucket[SHORT_A], (lengths, bucket)

    # Repeats of a prompt reproduce its first condition exactly, eager or graphed:
    # nothing from the requests in between leaks into the replay.
    for run in (eager, graphed):
        first: dict[str, torch.Tensor] = {}
        for text, cond in zip(SEQUENCE, run):
            if text in first:
                torch.testing.assert_close(cond, first[text], atol=0, rtol=0)
            else:
                first[text] = cond
