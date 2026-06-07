#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""π0 e2e: LeRobot parity (in-process) + OpenPI online serving (websocket).

This file holds two tests:
  * ``test_pi0_vllm_omni_vs_lerobot`` (below) — the bit-for-bit parity oracle.
  * ``test_pi0_openpi_online`` (bottom) — boots ``vllm serve --deploy-config
    pi0.yaml`` and drives the real OpenPI websocket, asserting ``[50, 32]``
    actions. Needs a GPU; skipped in CI. Mirrors
    ``tests/e2e/online_serving/test_dreamzero_expansion.py``.

--- LeRobot parity ---
Verifies that vllm-omni's ``Pi0ForActionPrediction`` produces bit-for-bit
matching action chunks with LeRobot's ``PI0Policy`` when fed:
  * the same weights (``lerobot/pi0_base``)
  * the same pre-processed inputs (images, masks, tokens, state)
  * the same initial noise tensor

π0 is a flow-matching model (Euler-integrated ODE from t=1 → t=0 with a fixed
``num_steps``), so the output is deterministic once the noise is fixed and
``torch.allclose`` on the final action chunk is a valid oracle. This is the
authoritative correctness oracle for the vllm-omni π0 port (max|Δ| < 1e-4).

Run in a SEPARATE ``lerobot[pi]`` venv (avoids dep conflict with the vllm-omni
env), with the vllm-omni ``pi0`` package importable::

    PI0_PARITY_MODEL_PATH=lerobot/pi0_base \
    PI0_PARITY_DEVICE=cpu PI0_PARITY_DTYPE=float32 \
        python -m pytest tests/pi0/test_pi0_e2e.py::test_pi0_vllm_omni_vs_lerobot -v -s

Skipped automatically when LeRobot is not installed (e.g. the vllm-omni env / CI).

Environment variables:
  PI0_PARITY_DEVICE       cpu | cuda  (default: cpu)
  PI0_PARITY_DTYPE        float32 | bfloat16  (default: float32)
  PI0_PARITY_ATOL         absolute tolerance (default: 1e-4)
  PI0_PARITY_FROM_PRETRAINED  "1" real weights, "0" random (default: 1)
  PI0_PARITY_BATCH_SIZE   default: 2
  PI0_PARITY_NUM_STEPS    flow-matching steps (default: 10)
  PI0_PARITY_MODEL_PATH   Local path or HF repo id of the pi0_base checkpoint in
                          LeRobot format. Default: "lerobot/pi0_base" (HF download).
"""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch

# ─── Skip rules ───────────────────────────────────────────────────────
# Skip the whole file in CI (needs real weights). The LeRobot-comparison test is
# additionally gated on lerobot being importable (run it in a lerobot venv at
# transformers 5.3.0); the pipeline e2e test is gated only on a checkpoint, so it
# can run in the vllm-omni env where lerobot is absent.
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Parity test requires real weights + LeRobot; not meant for CI",
)


# ─── Config ───────────────────────────────────────────────────────────
DEVICE = os.environ.get("PI0_PARITY_DEVICE", "cpu")
DTYPE_STR = os.environ.get("PI0_PARITY_DTYPE", "float32")
ATOL = float(os.environ.get("PI0_PARITY_ATOL", "1e-4"))
FROM_PRETRAINED = os.environ.get("PI0_PARITY_FROM_PRETRAINED", "1") == "1"
BATCH_SIZE = int(os.environ.get("PI0_PARITY_BATCH_SIZE", "2"))
NUM_STEPS = int(os.environ.get("PI0_PARITY_NUM_STEPS", "10"))
MODEL_PATH = os.environ.get("PI0_PARITY_MODEL_PATH", "lerobot/pi0_base")

# Must match LeRobot defaults for ``lerobot/pi0_base``.
ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 48


# ─── Online-serving e2e wiring (guarded) ──────────────────────────────
# The serving stack (tests.helpers.runtime + the OpenPI websocket client) is
# only needed by test_pi0_openpi_online. Guard the imports so the LeRobot parity
# test can still be collected / run as a script (``python tests/pi0/test_pi0_e2e.py``)
# in a minimal lerobot venv that lacks the full vllm-omni serving deps.
try:
    from tests.helpers.mark import hardware_test
    from tests.helpers.runtime import OmniServerParams, get_open_port
    from tests.pi0 import openpi_client_helper

    _HAS_SERVING_DEPS = True
except Exception:  # noqa: BLE001 - serving stack absent (e.g. lerobot parity venv)
    _HAS_SERVING_DEPS = False

    def hardware_test(**_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap

    OmniServerParams = None  # type: ignore[assignment]
    get_open_port = None  # type: ignore[assignment]
    openpi_client_helper = None  # type: ignore[assignment]

# Server model for the online e2e: a local pi0_base dir via PI0_E2E_CKPT, else
# the HF repo id (downloaded on first boot).
E2E_MODEL = os.environ.get("PI0_E2E_CKPT", "lerobot/pi0_base")
test_params: list = []
if _HAS_SERVING_DEPS:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"
    test_params = [
        OmniServerParams(
            model=E2E_MODEL,
            port=8092,
            server_args=[
                "--deploy-config",
                "vllm_omni/deploy/pi0.yaml",
                "--served-model-name",
                "pi0",
                "--enforce-eager",
                "--disable-log-stats",
            ],
            env_dict={
                "ATTENTION_BACKEND": "torch",
                "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "MASTER_PORT": str(get_open_port()),
            },
        )
    ]


def _resolve_checkpoint_dir() -> str:
    """Return a local dir containing the pi0_base checkpoint (download if needed)."""
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")


# ─── Dummy dataset stats (identity) ──────────────────────────────────
def _dummy_dataset_stats() -> dict:
    return {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM),
            "std": torch.ones(STATE_DIM),
            "q01": torch.zeros(STATE_DIM),
            "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
            "q01": torch.zeros(ACTION_DIM),
            "q99": torch.ones(ACTION_DIM),
        },
        "images": {
            cam: {
                "mean": torch.zeros(3, 224, 224),
                "std": torch.ones(3, 224, 224),
                "q01": torch.zeros(3, 224, 224),
                "q99": torch.ones(3, 224, 224),
            }
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
    }


def _create_dummy_batch(batch_size: int = BATCH_SIZE, device: str = DEVICE) -> dict:
    """Reproducible dummy inputs — identical across both implementations."""
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt = "Pick up the red block and place it in the bin"
    return {
        "observation.state": torch.randn(
            batch_size, STATE_DIM, generator=g, dtype=torch.float32
        ).to(device),
        "action": torch.randn(
            batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32
        ).to(device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device),
        "observation.images.left_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device),
        "observation.images.right_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device),
        "task": [prompt for _ in range(batch_size)],
    }


# ─── LeRobot instantiation ────────────────────────────────────────────
def _instantiate_lerobot():
    from lerobot.policies.pi0 import PI0Config, PI0Policy
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

    if FROM_PRETRAINED:
        policy = PI0Policy.from_pretrained(MODEL_PATH, strict=True)
    else:
        config = PI0Config(
            max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM, dtype=DTYPE_STR
        )
        policy = PI0Policy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    policy.eval()

    pre, post = make_pi0_pre_post_processors(
        config=policy.config, dataset_stats=_dummy_dataset_stats()
    )
    return policy, pre, post


# ─── vllm-omni instantiation ──────────────────────────────────────────
def _instantiate_vllm_omni():
    """Build the vllm-omni π0 model in isolation (no pipeline, no engine)."""
    from vllm_omni.diffusion.models.pi0 import Pi0Config, Pi0ForActionPrediction

    cfg = Pi0Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=NUM_STEPS,
        dtype=DTYPE_STR,
    )
    model = Pi0ForActionPrediction(cfg)
    model.to(DEVICE).eval()

    if FROM_PRETRAINED:
        _load_lerobot_weights(model)
    return model


def _load_lerobot_weights(model):
    """Feed the ``lerobot/pi0_base`` safetensors into the vllm-omni model.

    ``Pi0ForActionPrediction.load_weights`` handles the leading ``model.``
    prefix and the flat→nested / lm_head→embed_tokens remaps, so we can pass
    the raw checkpoint dict.
    """
    import safetensors.torch

    cache_dir = _resolve_checkpoint_dir()
    path = os.path.join(cache_dir, "model.safetensors")
    state = safetensors.torch.load_file(path)
    model.load_weights(list(state.items()))


# ─── Helpers to extract LeRobot's pre-processed inputs ────────────────
def _extract_lerobot_model_inputs(lerobot_policy, processed_batch):
    """Mimic what ``PI0Policy.predict_action_chunk`` feeds into
    ``self.model.sample_actions``. We use these *exact* tensors for vllm-omni
    so any divergence must come from the core model, not preprocessing.
    """
    images, img_masks = lerobot_policy._preprocess_images(processed_batch)
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    lang_tokens = processed_batch[OBS_LANGUAGE_TOKENS]
    lang_masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = lerobot_policy.prepare_state(processed_batch)
    return images, img_masks, lang_tokens, lang_masks, state


# ─── Shared fixed-noise sampler ───────────────────────────────────────
def _make_fixed_noise(batch_size: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(
        batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32
    ).to(device)


# ─── Main test ────────────────────────────────────────────────────────
@pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed (run in a lerobot venv).")
def test_pi0_vllm_omni_vs_lerobot():
    print("\n[parity] Instantiating LeRobot…")
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()

    print("[parity] Instantiating vllm-omni…")
    omni_model = _instantiate_vllm_omni()

    print("[parity] Preparing shared inputs…")
    raw_batch = _create_dummy_batch()
    processed_batch = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks, state = _extract_lerobot_model_inputs(
        lerobot_policy, processed_batch
    )
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    print(f"[parity] state.shape={state.shape}  lang_tokens.shape={lang_tokens.shape}")
    print(f"[parity] images[0].shape={images[0].shape} (num_cams={len(images)})")
    print(f"[parity] noise.shape={noise.shape}  dtype={noise.dtype}")

    # ── LeRobot forward ──
    print("[parity] Running LeRobot sample_actions…")
    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=NUM_STEPS,
        )
    print(
        f"[parity] LeRobot actions: shape={lerobot_actions.shape} "
        f"mean={lerobot_actions.mean().item():.6f} std={lerobot_actions.std().item():.6f}"
    )

    # ── vllm-omni forward ──
    print("[parity] Running vllm-omni sample_actions…")
    with torch.no_grad():
        omni_actions = omni_model.sample_actions(
            images=images,
            image_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            num_steps=NUM_STEPS,
        )
    print(
        f"[parity] vllm-omni actions: shape={omni_actions.shape} "
        f"mean={omni_actions.mean().item():.6f} std={omni_actions.std().item():.6f}"
    )

    # ── Compare ──
    diff = (lerobot_actions.float() - omni_actions.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[parity] |Δ| max={max_diff:.2e}  mean={mean_diff:.2e}  atol={ATOL:.1e}")
    close = torch.allclose(lerobot_actions.float(), omni_actions.float(), atol=ATOL)
    print(f"[parity] torch.allclose(atol={ATOL}): {close}")

    if not close:
        print("\n[parity] ⚠️  Outputs diverge — running per-stage diagnostics…")
        _diagnose_divergence(
            lerobot_policy.model, omni_model,
            images, img_masks, lang_tokens, lang_masks, state, noise,
        )

    assert close, (
        f"vllm-omni vs LeRobot actions differ beyond atol={ATOL}. "
        f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
    )


# ─── Per-stage divergence diagnostics ─────────────────────────────────
@torch.no_grad()
def _diagnose_divergence(
    lerobot_flow_model, omni_model,
    images, img_masks, lang_tokens, lang_masks, state, noise,
):
    """Localize a numerical mismatch to a specific pipeline stage:
      1. Prefix embeddings (SigLIP / embed_tokens / projector) — image vs lang.
      2. Prefix KV cache layer 0 — PaliGemma LM attention.
      3. A single denoise_step velocity at t=1.0 — action expert forward.
    """
    from vllm_omni.diffusion.models.pi0.modeling_pi0 import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    # ── Stage 1: prefix embeddings ──
    lr_prefix_embs, lr_prefix_pad, lr_prefix_att = lerobot_flow_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    sg_prefix_embs, sg_prefix_pad, sg_prefix_att = omni_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    total_diff = (lr_prefix_embs.float() - sg_prefix_embs.float()).abs().max().item()
    print(
        f"[diag] prefix_embs max |Δ| = {total_diff:.2e}   "
        f"(shape={tuple(sg_prefix_embs.shape)})"
    )
    print(f"[diag] prefix_pad_masks equal: {torch.equal(lr_prefix_pad, sg_prefix_pad)}")
    print(
        f"[diag] prefix_att_masks equal: "
        f"{torch.equal(lr_prefix_att.bool(), sg_prefix_att.bool())}"
    )

    num_cams = len(images)
    img_len = 256 * num_cams
    lang_len = lr_prefix_embs.shape[1] - img_len
    img_diff = (
        lr_prefix_embs[:, :img_len].float() - sg_prefix_embs[:, :img_len].float()
    ).abs().max().item()
    lang_diff = (
        lr_prefix_embs[:, img_len:].float() - sg_prefix_embs[:, img_len:].float()
    ).abs().max().item()
    print(f"[diag]   image slice [:{img_len}] max|Δ| = {img_diff:.2e}   (num_cams={num_cams})")
    print(f"[diag]   lang slice  [{img_len}:] max|Δ| = {lang_diff:.2e}   (len={lang_len})")

    def _stats(name, t):
        t = t.float()
        print(
            f"[diag] {name} prefix_embs: mean={t.mean().item():+.4f} "
            f"std={t.std().item():.4f} min={t.min().item():+.2f} "
            f"max={t.max().item():+.2f}"
        )

    _stats("LeRobot ", lr_prefix_embs)
    _stats("vllm-omni", sg_prefix_embs)

    # ── Stage 2: prefix KV cache ──
    prefix_att_2d = make_att_2d_masks(sg_prefix_pad, sg_prefix_att)
    prefix_pos = torch.cumsum(sg_prefix_pad, dim=1) - 1
    prefix_att_4d = prepare_attention_masks_4d(prefix_att_2d)

    lerobot_flow_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, lr_kv = lerobot_flow_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_pos,
        past_key_values=None, inputs_embeds=[lr_prefix_embs, None], use_cache=True,
    )
    _, sg_kv = omni_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_pos,
        past_key_values=None, inputs_embeds=[sg_prefix_embs, None], use_cache=True,
    )

    def _layer0_kv(cache):
        if isinstance(cache, list):
            return cache[0]
        layer0 = cache.layers[0]
        return layer0.keys, layer0.values

    try:
        lr_k, lr_v = _layer0_kv(lr_kv)
        sg_k, sg_v = _layer0_kv(sg_kv)
        dk = (lr_k.float() - sg_k.float()).abs().max().item()
        dv = (lr_v.float() - sg_v.float()).abs().max().item()
        print(f"[diag] prefix KV layer0  K max|Δ|={dk:.2e}  V max|Δ|={dv:.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"[diag] could not extract prefix KV for comparison: {e}")

    # ── Stage 3: a single denoise_step at t=1.0 ──
    bsize = state.shape[0]
    t = torch.ones(bsize, dtype=torch.float32, device=state.device)
    lr_vt = lerobot_flow_model.denoise_step(state, sg_prefix_pad, lr_kv, noise, t)
    sg_vt = omni_model.denoise_step(state, sg_prefix_pad, sg_kv, noise, t)
    print(
        f"[diag] denoise_step(t=1) v_t max|Δ| = "
        f"{(lr_vt.float() - sg_vt.float()).abs().max().item():.2e}"
    )


# ─── End-to-end online serving (OpenPI websocket) ─────────────────────
#
# Boots ``vllm serve --omni --deploy-config pi0.yaml`` and drives the real
# OpenPI websocket (``/v1/realtime/robot/openpi``) — the same wire path a robot
# uses (handshake metadata → send observation → receive action chunk). Mirrors
# ``tests/e2e/online_serving/test_dreamzero_expansion.py``. Needs a GPU + a
# pi0_base checkpoint; skipped in CI (module-level ``pytestmark``) and when the
# serving stack / OpenPI client deps are unavailable.
@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.skipif(not _HAS_SERVING_DEPS, reason="serving stack / OpenPI deps unavailable")
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_pi0_openpi_online(omni_server):
    try:
        openpi_client_helper.require_dependencies()
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    result = openpi_client_helper.run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        prompt="pick up the red block and place it in the bin",
        session_id="pi0-online-e2e",
        num_steps=2,
    )

    # Asserts every returned chunk is [50, 32] + finite, and the handshake
    # metadata matches pi0.yaml's policy_server_config.
    openpi_client_helper.validate_session_result(result)

    metadata = result["metadata"]
    assert tuple(metadata["image_resolution"]) == (224, 224)
    assert metadata["needs_wrist_camera"] is True
    assert metadata["needs_session_id"] is False
    assert metadata["action_space"] == "joint_position"


if __name__ == "__main__":
    test_pi0_vllm_omni_vs_lerobot()
