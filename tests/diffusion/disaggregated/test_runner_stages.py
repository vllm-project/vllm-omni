# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral runner tests for disaggregated stages (#4948 DiffusionV2Atoms).

These import the real ``DiffusionModelRunner`` and drive it with fake pipelines
and lightweight tensors, so they require torch + the vllm_omni runtime. They are
marked ``needs_runtime`` and skipped where those deps are unavailable (e.g. the
Windows dev host); run them on the XPU node inside the vllm-omni-xpu container:

    pytest tests/diffusion/disaggregated/test_runner_stages.py -m needs_runtime
"""

from __future__ import annotations

import types

import pytest

# The full runner path needs torch + the vllm_omni runtime. Any failure to
# import them (absent, or a broken partial install as on the Windows dev host)
# skips the whole module rather than erroring collection.
try:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.interface import StageBoundary, StagePayload
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.stage_roles import DECODE, DENOISE, ENCODE, StageComponentSpec
    from vllm_omni.diffusion.worker.diffusion_model_runner import (
        STAGE_PAYLOAD_OUTPUT_KEY,
        STAGE_PAYLOAD_PROMPT_KEY,
        DiffusionModelRunner,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
except Exception as exc:  # ImportError, or torch DLL load failure
    pytest.skip(f"vllm_omni runtime unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.needs_runtime

_CARRIER_KEY = "fake_carrier"


class FakeDisaggregatedPipeline:
    """Minimal pipeline implementing the DiffusionV2Atoms disaggregation contract.

    Records which atoms ran so tests can assert a stage never crosses into
    another stage's work (no DiT on encode/decode, no encode on denoise, etc.).
    Everything is stashed on a private carrier on ``state.extra`` and marshalled
    through pack/unpack_stage_state, mirroring the DreamZero shape.
    """

    supports_disaggregated_execution = True
    supports_step_execution = False

    def __init__(self):
        self.calls: list[str] = []
        self.enable_diffusion_pipeline_profiler = False

    # --- DiffusionV2Atoms (encode chain) ---
    def init_state(self, state):
        self.calls.append("init_state")
        state.extra.pop(_CARRIER_KEY, None)
        return state

    def check_inputs(self, state):
        self.calls.append("check_inputs")
        return state

    def encode(self, state):
        self.calls.append("encode")
        state.extra[_CARRIER_KEY] = {"prompt_embeds": torch.zeros(1, 4)}
        return state

    def prepare(self, state):
        self.calls.append("prepare")
        return state

    # --- denoise (whole-request atom) ---
    def diffuse(self, state):
        self.calls.append("diffuse")
        carrier = state.extra[_CARRIER_KEY]
        carrier["latents"] = torch.ones(1, 16)
        return state

    # --- decode chain ---
    def decode(self, state):
        self.calls.append("decode")
        return state

    def postprocess(self, state):
        self.calls.append("postprocess")
        return DiffusionOutput(output={"image": torch.zeros(3, 8, 8)})

    # --- payload marshalling ---
    def pack_stage_state(self, state, boundary):
        self.calls.append(f"pack:{boundary.value}")
        carrier = state.extra.get(_CARRIER_KEY, {})
        return StagePayload.create(
            request_id=state.request_id,
            boundary=boundary,
            scalar_fields={"session_id": "sess"},
            private_tensor_fields={k: v for k, v in carrier.items() if v is not None},
        )

    def unpack_stage_state(self, payload, state):
        self.calls.append("unpack")
        state.extra[_CARRIER_KEY] = dict(payload.private_tensor_fields)
        return state

    # --- step-only stubs (unused: request-mode pipeline) ---
    def build_step_batch(self, states, *, cached_batch=None):
        raise NotImplementedError

    def build_step_attention_metadata(self, input_batch):
        return None

    def denoise_step(self, input_batch):
        raise NotImplementedError

    def step_scheduler(self, state, noise_pred):
        raise NotImplementedError

    @classmethod
    def required_components_for_stage(cls, model_stage):
        if model_stage == ENCODE:
            return StageComponentSpec(tokenizer=True, text_encoder=True, vae_encoder=True)
        if model_stage == DENOISE:
            return StageComponentSpec(dit=True, scheduler=True)
        if model_stage == DECODE:
            return StageComponentSpec(vae_decoder=True)
        return StageComponentSpec()


def _make_runner(model_stage):
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = types.SimpleNamespace(
        model_stage=model_stage,
        model_class_name="FakeDisaggregatedPipeline",
        parallel_config=types.SimpleNamespace(use_hsdp=False),
        stage_id=0,
    )
    runner.device = torch.device("cpu")
    runner.vllm_config = None
    runner.pipeline = FakeDisaggregatedPipeline()
    runner.state_cache = {}
    return runner


def _request(prompt):
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(seed=0),
        request_id="req-1",
    )


def _encode_payload():
    return StagePayload.create(
        request_id="req-1",
        boundary=StageBoundary.ENCODE_TO_DIT,
        scalar_fields={"session_id": "sess"},
        private_tensor_fields={"prompt_embeds": torch.zeros(1, 4)},
    )


def _denoise_payload():
    return StagePayload.create(
        request_id="req-1",
        boundary=StageBoundary.DIT_TO_DECODE,
        scalar_fields={"session_id": "sess"},
        private_tensor_fields={"latents": torch.ones(1, 16)},
    )


def test_encode_stage_runs_only_encode_atoms():
    runner = _make_runner(ENCODE)
    out = runner.execute_encode_stage(_request({"prompt": "hi"}))
    calls = runner.pipeline.calls
    assert calls[:4] == ["init_state", "check_inputs", "encode", "prepare"]
    assert "diffuse" not in calls and "decode" not in calls
    payload = out.custom_output[STAGE_PAYLOAD_OUTPUT_KEY]
    assert payload.boundary is StageBoundary.ENCODE_TO_DIT
    # state released
    assert "req-1" not in runner.state_cache


def test_denoise_stage_unpacks_payload_and_does_not_encode():
    runner = _make_runner(DENOISE)
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: _encode_payload()}})
    out = runner.execute_denoise_stage(req)
    calls = runner.pipeline.calls
    assert "unpack" in calls and "diffuse" in calls
    assert "check_inputs" not in calls and "encode" not in calls
    payload = out.custom_output[STAGE_PAYLOAD_OUTPUT_KEY]
    assert payload.boundary is StageBoundary.DIT_TO_DECODE


def test_decode_stage_runs_only_decode():
    runner = _make_runner(DECODE)
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: _denoise_payload()}})
    out = runner.execute_decode_stage(req)
    calls = runner.pipeline.calls
    assert "unpack" in calls and "decode" in calls and "postprocess" in calls
    assert "diffuse" not in calls
    assert out.output is not None and "image" in out.output


def test_denoise_rejects_wrong_boundary():
    runner = _make_runner(DENOISE)
    # A DIT_TO_DECODE payload arriving at the denoise stage must be rejected.
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: _denoise_payload()}})
    with pytest.raises(Exception, match="boundary"):
        runner.execute_denoise_stage(req)


def test_missing_payload_raises_actionable_error():
    runner = _make_runner(DENOISE)
    req = _request({"prompt": "no payload here"})
    with pytest.raises(Exception, match="without a StagePayload"):
        runner.execute_denoise_stage(req)


# --- RFC #4590 B.1: request-identity cross-check ----------------------------


def _payload_for(request_id, boundary=None):
    return StagePayload.create(
        request_id=request_id,
        boundary=boundary or StageBoundary.ENCODE_TO_DIT,
        scalar_fields={"session_id": "sess"},
        private_tensor_fields={"prompt_embeds": torch.zeros(1, 4)},
    )


def test_denoise_rejects_cross_request_payload():
    """A payload built for request A delivered to request B must be rejected
    before any model/KV work (RFC #4590 B.1)."""
    runner = _make_runner(DENOISE)
    # request_id is "req-1" (see _request); route a payload stamped "other-req".
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: _payload_for("other-req")}})
    with pytest.raises(Exception, match="cross-request|other-req"):
        runner.execute_denoise_stage(req)
    # The pipeline never ran diffuse — rejection happened before model execution.
    assert "diffuse" not in runner.pipeline.calls


def test_decode_rejects_cross_request_payload():
    runner = _make_runner(DECODE)
    req = _request(
        {"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: _payload_for("other-req", StageBoundary.DIT_TO_DECODE)}}
    )
    with pytest.raises(Exception, match="cross-request|other-req"):
        runner.execute_decode_stage(req)
    assert "postprocess" not in runner.pipeline.calls


def test_matching_request_id_is_accepted():
    """Sanity: the same-request payload (the normal case) is NOT rejected."""
    runner = _make_runner(DENOISE)
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: _payload_for("req-1")}})
    # req-1 matches; denoise proceeds and packs a DIT_TO_DECODE payload.
    out = runner.execute_denoise_stage(req)
    assert out.custom_output[STAGE_PAYLOAD_OUTPUT_KEY].boundary is StageBoundary.DIT_TO_DECODE


def test_malformed_dict_payload_raises_stage_payload_error():
    """A partial/plain-dict payload (no request_id) surfaces as StagePayloadError,
    not a raw KeyError, when rehydrated on the consumer side (RFC #4590 B.1)."""
    from vllm_omni.diffusion.stage_payload import StagePayloadError

    runner = _make_runner(DENOISE)
    # A dict missing required keys, as if a corrupted wire payload arrived.
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: {"session_id": "sess"}}})
    with pytest.raises(StagePayloadError):
        runner.execute_denoise_stage(req)


# --- RFC #4590 B.2: disaggregated stages cannot enter the monolithic batch ---


class _NewReq:
    def __init__(self, req):
        self.req = req


def _scheduler_output(reqs):
    return types.SimpleNamespace(scheduled_new_reqs=[_NewReq(r) for r in reqs])


@pytest.mark.parametrize("stage", [ENCODE, DENOISE, DECODE])
def test_disaggregated_stage_rejects_monolithic_batch_path(stage):
    """execute_model_batch runs the monolithic pipeline.forward(batch); a
    disaggregated role must be rejected there rather than silently composing all
    phases in one process (RFC #4590 B.2)."""
    runner = _make_runner(stage)
    sched = _scheduler_output([_request({"prompt": "hi"})])
    with pytest.raises(RuntimeError, match="disaggregated|batch"):
        runner.execute_model_batch(sched, runner.od_config)


def test_monolithic_stage_allows_batch_path():
    """A monolithic ('diffusion') runner is NOT blocked by the disaggregated
    batch guard — it fails later for an unrelated reason (no real pipeline), not
    with the disaggregated-role RuntimeError."""
    runner = _make_runner("diffusion")
    assert runner.is_disaggregated_stage is False
    sched = _scheduler_output([_request({"prompt": "hi"})])
    # The guard must NOT fire for the monolithic role. Any error raised here must
    # not be the disaggregated-batch rejection.
    try:
        runner.execute_model_batch(sched, runner.od_config)
    except RuntimeError as exc:
        assert "disaggregated" not in str(exc)
    except Exception:
        pass  # other failures (fake pipeline) are fine — the guard didn't fire
