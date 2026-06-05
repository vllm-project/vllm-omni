#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity test: vllm-omni π0 vs LeRobot π0.

Verifies that vllm-omni's ``Pi0ForActionPrediction`` produces bit-for-bit
matching action chunks with LeRobot's ``PI0Policy`` when fed:
  * the same weights (``lerobot/pi0_base``)
  * the same pre-processed inputs (images, masks, tokens, state)
  * the same initial noise tensor

π0 is a flow-matching model (Euler-integrated ODE from t=1 → t=0 with a fixed
``num_steps``), so the output is deterministic once the noise is fixed and
``torch.allclose`` on the final action chunk is a valid oracle. This is the
authoritative correctness oracle for the SGLang→vllm-omni port (max|Δ| < 1e-4).

Run in a SEPARATE ``lerobot[pi]`` venv (avoids dep conflict with the vllm-omni
env), with the vllm-omni ``pi0`` package importable::

    PI0_PARITY_MODEL_PATH=lerobot/pi0_base \
    PI0_PARITY_DEVICE=cpu PI0_PARITY_DTYPE=float32 \
        python -m pytest tests/pi0/test_pi0_parity.py -v -s

Skipped automatically when LeRobot is not installed (e.g. the vllm-omni env / CI).

Environment variables:
  PI0_PARITY_DEVICE       cpu | cuda  (default: cpu)
  PI0_PARITY_DTYPE        float32 | bfloat16  (default: float32)
  PI0_PARITY_ATOL         absolute tolerance (default: 1e-4)
  PI0_PARITY_FROM_PRETRAINED  "1" real weights, "0" random (default: 1)
  PI0_PARITY_BATCH_SIZE   default: 2
  PI0_PARITY_NUM_STEPS    flow-matching steps (default: 10)
  PI0_PARITY_MODEL_PATH   Local path or HF repo id of the pi0_base checkpoint IN
                          LEROBOT FORMAT (consumed by PI0Policy.from_pretrained,
                          which rejects SGLang-only keys model_type/architectures/
                          auto_map). Default: "lerobot/pi0_base" (HF download).
                          DO NOT point at a SGLang-config dir like /data08/models/pi0.
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

    # Friendly check: a SGLang-flavor config.json (top-level model_type: pi0)
    # makes PI0Policy.from_pretrained blow up deep inside draccus.
    if FROM_PRETRAINED and os.path.isdir(MODEL_PATH):
        cfg_path = os.path.join(MODEL_PATH, "config.json")
        if os.path.exists(cfg_path):
            try:
                import json as _json

                with open(cfg_path) as _f:
                    _cfg = _json.load(_f)
                if _cfg.get("model_type") == "pi0":
                    raise RuntimeError(
                        f"PI0_PARITY_MODEL_PATH={MODEL_PATH!r} looks like a "
                        "SGLang-config π0 dir (config.json has 'model_type': "
                        "'pi0'); LeRobot's PI0Policy.from_pretrained rejects "
                        "SGLang-only keys. Unset it (defaults to the HF repo "
                        "lerobot/pi0_base) or point it at a LeRobot-format dir."
                    )
            except (OSError, ValueError):
                pass

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


# ─── End-to-end serving path (Pi0Pipeline.forward) ────────────────────
#
# Exercises the real serving entry point — Pi0Pipeline.forward(req): raw
# robot_obs → build_model_inputs → sample_actions → DiffusionOutput{actions}
# (the same path the OpenPI websocket server drives, minus the wire transport).
# Runs in the vllm-omni env on the production transformers; gated on a real
# checkpoint via PI0_E2E_CKPT. Not a bit-for-bit LeRobot comparison (LeRobot's
# reference is only valid at transformers 5.3.0); a shape + finiteness + spread
# sanity that guards the pipeline wiring (config resolve, weight load, pre/post).
E2E_CKPT = os.environ.get("PI0_E2E_CKPT")


@pytest.mark.skipif(not E2E_CKPT, reason="Set PI0_E2E_CKPT to a pi0_base dir for the e2e path.")
def test_pi0_pipeline_e2e():
    import numpy as np

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi0 import Pi0Pipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    od = OmniDiffusionConfig(model=E2E_CKPT, model_class_name="Pi0Pipeline", dtype="float32")
    pipe = Pi0Pipeline(od_config=od)

    obs = {
        "observation.images.base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(STATE_DIM, dtype=np.float32),
        "prompt": "pick up the red block",
    }
    sp = OmniDiffusionSamplingParams(extra_args={"robot_obs": obs, "session_id": "e2e", "reset": True})
    req = OmniDiffusionRequest(prompts=["pick up the red block"], sampling_params=sp, request_id="e2e-0")

    out = pipe.forward(req)
    assert out.error is None, out.error
    actions = np.asarray(out.output["actions"], dtype=np.float32)
    assert actions.shape == (ACTION_HORIZON, ACTION_DIM), actions.shape
    assert np.isfinite(actions).all()
    assert float(actions.std()) > 1e-3, "action output collapsed"

    # Dummy-warmup path returns zeros without touching the obs.
    sp0 = OmniDiffusionSamplingParams(extra_args={})
    req0 = OmniDiffusionRequest(prompts=["dummy run"], sampling_params=sp0, request_id="e2e-warmup")
    out0 = pipe.forward(req0)
    assert np.asarray(out0.output["actions"]).shape == (ACTION_HORIZON, ACTION_DIM)
    assert not np.asarray(out0.output["actions"]).any()


if __name__ == "__main__":
    test_pi0_vllm_omni_vs_lerobot()
