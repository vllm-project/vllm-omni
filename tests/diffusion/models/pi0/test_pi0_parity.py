#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""π0 LeRobot parity (in-process): the bit-for-bit correctness oracle.

Verifies that vllm-omni's ``Pi0ForActionPrediction`` produces bit-for-bit
matching action chunks with LeRobot's ``PI0Policy`` when fed:
  * the same weights (``lerobot/pi0_base``)
  * the same pre-processed inputs (images, masks, tokens, state)
  * the same initial noise tensor

π0 is a flow-matching model (Euler-integrated ODE from t=1 → t=0 with a fixed
``num_steps``), so the output is deterministic once the noise is fixed and
``torch.allclose`` on the final action chunk is a valid oracle. This is the
authoritative correctness oracle for the vllm-omni π0 port (max|Δ| < 1e-4).

The OpenPI websocket online-serving e2e lives separately in
``tests/e2e/online_serving/test_pi0_expansion.py``.

Run in a SEPARATE ``lerobot[pi]`` venv (avoids dep conflict with the vllm-omni
env), with the vllm-omni ``pi0`` package importable::

    python -m pytest tests/diffusion/models/pi0/test_pi0_parity.py -v -s

Skipped automatically when LeRobot is not installed (e.g. the vllm-omni env).
The run is CPU/float32 with fixed defaults; the only override is
``PI0_PARITY_MODEL_PATH`` (a local pi0_base dir in LeRobot format, to skip the
HF download; defaults to ``lerobot/pi0_base``).
"""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch

# local_model: needs real weights + a lerobot venv (transformers 5.3.0), so it
# runs locally rather than in the ready-CI (see docs/contributing/ci/CI_5levels.md).
# Additionally gated on lerobot being importable.
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]


# ─── Config (fixed; matches LeRobot defaults for ``lerobot/pi0_base``) ──
DEVICE = os.environ.get("PI_PARITY_DEVICE", "cpu")
DTYPE_STR = "float32"
ATOL = 1e-4
BF16_ATOL = 5e-2
NUM_STEPS = 10
BATCH_SIZE = 2
ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 48

# The only knob: point at a local pi0_base dir (LeRobot format) to skip the HF
# download; defaults to the HF repo id.
MODEL_PATH = os.environ.get("PI0_PARITY_MODEL_PATH", "lerobot/pi0_base")
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


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


def _create_dummy_batch(batch_size: int = BATCH_SIZE, device: str = DEVICE, num_views: int = 3) -> dict:
    """Reproducible dummy inputs — identical across both implementations."""
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt = "Pick up the red block and place it in the bin"
    batch = {
        "observation.state": torch.randn(batch_size, STATE_DIM, generator=g, dtype=torch.float32).to(device),
        "action": torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32).to(device),
        "task": [prompt for _ in range(batch_size)],
    }
    for camera in CAMERAS[:num_views]:
        batch[f"observation.images.{camera}"] = torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device)
    return batch


# ─── LeRobot instantiation ────────────────────────────────────────────
def _instantiate_lerobot(dtype: str = DTYPE_STR):
    from lerobot.policies.pi0 import PI0Policy
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

    config = PI0Policy.config_class.from_pretrained(MODEL_PATH)
    config.dtype = dtype
    config.device = DEVICE
    policy = PI0Policy.from_pretrained(MODEL_PATH, config=config, strict=True)
    policy.to(DEVICE)
    policy.config.device = DEVICE
    policy.eval()

    reference_q_dtype = policy.model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj.weight.dtype
    assert reference_q_dtype == getattr(torch, dtype), (
        f"LeRobot reference requested {dtype}, but layer-0 Q projection uses {reference_q_dtype}."
    )

    pre, post = make_pi0_pre_post_processors(config=policy.config, dataset_stats=_dummy_dataset_stats())
    return policy, pre, post


# ─── vllm-omni instantiation ──────────────────────────────────────────
def _instantiate_vllm_omni(dtype: str = DTYPE_STR):
    """Build the vllm-omni π0 model in isolation (no pipeline, no engine)."""
    from vllm_omni.diffusion.models.pi.common import inference_dtype
    from vllm_omni.diffusion.models.pi0 import Pi0Config, Pi0ForActionPrediction

    cfg = Pi0Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=NUM_STEPS,
        dtype=dtype,
    )
    model = Pi0ForActionPrediction(cfg)
    inference_dtype.apply_pi_inference_dtype(model, getattr(torch, dtype))
    model.to(device=DEVICE).eval()
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
    return torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32).to(device)


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
    images, img_masks, lang_tokens, lang_masks, state = _extract_lerobot_model_inputs(lerobot_policy, processed_batch)
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    print(f"[parity] state.shape={state.shape}  lang_tokens.shape={lang_tokens.shape}")
    print(f"[parity] images[0].shape={images[0].shape} (num_cams={len(images)})")
    print(f"[parity] noise.shape={noise.shape}  dtype={noise.dtype}")

    # ── LeRobot forward ──
    print("[parity] Running LeRobot sample_actions…")
    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=NUM_STEPS,
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
            lerobot_policy.model,
            omni_model,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
        )

    assert close, (
        f"vllm-omni vs LeRobot actions differ beyond atol={ATOL}. max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
    )


# ─── Per-stage divergence diagnostics ─────────────────────────────────
@pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed (run in a lerobot venv).")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pi0 BF16 reference parity requires CUDA.")
@pytest.mark.parametrize("num_views", [1, 2, 3])
def test_pi0_bfloat16_matches_lerobot_bfloat16(num_views):
    """Compare the shared mixed-BF16 implementation under fixed noise."""
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot(dtype="bfloat16")
    omni_model = _instantiate_vllm_omni(dtype="bfloat16")

    raw_batch = _create_dummy_batch(num_views=num_views)
    processed_batch = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks, state = _extract_lerobot_model_inputs(lerobot_policy, processed_batch)
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=NUM_STEPS,
        )
        omni_actions = omni_model.sample_actions(
            images=images,
            image_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            num_steps=NUM_STEPS,
        )

    diff = (lerobot_actions.float() - omni_actions.float()).abs()
    print(f"[parity] bfloat16 views={num_views} |Δ| max={diff.max().item():.2e} mean={diff.mean().item():.2e}")
    assert torch.allclose(lerobot_actions.float(), omni_actions.float(), atol=BF16_ATOL), (
        f"bfloat16 actions differ beyond atol={BF16_ATOL}; max_diff={diff.max().item():.2e}"
    )


@torch.no_grad()
def _diagnose_divergence(
    lerobot_flow_model,
    omni_model,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise,
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
    from vllm_omni.diffusion.models.pi.common import backbone

    sg_prefix_embs, sg_prefix_pad, sg_prefix_att = backbone.embed_multimodal_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        paligemma=omni_model.paligemma_with_expert.paligemma,
    )
    total_diff = (lr_prefix_embs.float() - sg_prefix_embs.float()).abs().max().item()
    diagnostics = {"prefix_max_abs": total_diff}
    print(f"[diag] prefix_embs max |Δ| = {total_diff:.2e}   (shape={tuple(sg_prefix_embs.shape)})")
    print(f"[diag] prefix_pad_masks equal: {torch.equal(lr_prefix_pad, sg_prefix_pad)}")
    print(f"[diag] prefix_att_masks equal: {torch.equal(lr_prefix_att.bool(), sg_prefix_att.bool())}")

    num_cams = len(images)
    img_len = 256 * num_cams
    lang_len = lr_prefix_embs.shape[1] - img_len
    img_diff = (lr_prefix_embs[:, :img_len].float() - sg_prefix_embs[:, :img_len].float()).abs().max().item()
    lang_diff = (lr_prefix_embs[:, img_len:].float() - sg_prefix_embs[:, img_len:].float()).abs().max().item()
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
        attention_mask=prefix_att_4d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[lr_prefix_embs, None],
        use_cache=True,
    )
    _, sg_kv = omni_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[sg_prefix_embs, None],
        use_cache=True,
    )

    def _layer0_kv(cache):
        if isinstance(cache, list):
            return cache[0]
        layer0 = cache.layers[0]
        return layer0.keys, layer0.values

    try:
        lr_k, lr_v = _layer0_kv(lr_kv)
        sg_k, sg_v = _layer0_kv(sg_kv)
        key_delta = (lr_k.float() - sg_k.float()).abs()
        value_delta = (lr_v.float() - sg_v.float()).abs()
        valid = sg_prefix_pad[:, None, :, None].expand_as(key_delta)
        masked = ~valid
        dk = key_delta.max().item()
        dv = value_delta.max().item()
        dk_valid = key_delta.masked_select(valid).max().item()
        dv_valid = value_delta.masked_select(valid).max().item()
        dk_masked = key_delta.masked_select(masked).max().item() if masked.any() else 0.0
        dv_masked = value_delta.masked_select(masked).max().item() if masked.any() else 0.0
        print(f"[diag] prefix KV layer0  K max|Δ|={dk:.2e}  V max|Δ|={dv:.2e}")
        print(
            f"[diag]   valid tokens  K max|Δ|={dk_valid:.2e}  V max|Δ|={dv_valid:.2e}; "
            f"masked tokens K max|Δ|={dk_masked:.2e}  V max|Δ|={dv_masked:.2e}"
        )
        diagnostics["layer0_kv"] = {
            "key_max_abs": dk,
            "value_max_abs": dv,
            "valid_key_max_abs": dk_valid,
            "valid_value_max_abs": dv_valid,
            "masked_key_max_abs": dk_masked,
            "masked_value_max_abs": dv_masked,
        }
    except Exception as e:  # noqa: BLE001
        print(f"[diag] could not extract prefix KV for comparison: {e}")

    # ── Stage 3: a single denoise_step at t=1.0 ──
    bsize = state.shape[0]
    t = torch.ones(bsize, dtype=torch.float32, device=state.device)
    lr_vt = lerobot_flow_model.denoise_step(state, sg_prefix_pad, lr_kv, noise, t)
    sg_vt = omni_model.denoise_step(state, sg_prefix_pad, sg_kv, noise, t)
    denoise_max_abs = (lr_vt.float() - sg_vt.float()).abs().max().item()
    diagnostics["denoise_t1_max_abs"] = denoise_max_abs
    print(f"[diag] denoise_step(t=1) v_t max|Δ| = {denoise_max_abs:.2e}")
    return diagnostics


if __name__ == "__main__":
    test_pi0_vllm_omni_vs_lerobot()
