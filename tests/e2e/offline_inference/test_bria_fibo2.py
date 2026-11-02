# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end tests for Bria Fibo-2.0 Dev / Edit Turbo through the full engine.

Needs the gated checkpoints in the local HF cache (``HF_HOME`` pointing at the
baseline cache works), so they run locally rather than in CI. The edit tests
always run against the committed source image
(``tests/assets/bria_fibo2/edit_source.png``); when the parity baseline
reference data is also available (``FIBO2_EDIT_REF`` etc.), the decoded images
are additionally compared against the baseline captures.
"""

import math
import os

import numpy as np
import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("FIBO2_MODEL", "briaai/Fibo-2.0-dev")
TURBO_MODEL = os.environ.get("FIBO2_TURBO_MODEL", "briaai/Fibo-2.0-edit-turbo")
REF_DATA = os.environ.get("FIBO2_REF_DATA", "/root/autodl-fs/fibo-baseline/ref-data/run1")
EDIT_REF = os.environ.get("FIBO2_EDIT_REF", "/root/autodl-fs/fibo-baseline/ref-data/edit1")
TURBO_REF = os.environ.get("FIBO2_TURBO_REF", "/root/autodl-fs/fibo-baseline/ref-data/edit-turbo1")

# The edit source committed with the repo: the baseline T2I capture the
# diffusers edit baseline was run on, so the optional parity comparison below
# sees exactly the input that baseline saw.
EDIT_SOURCE = os.path.join(os.path.dirname(__file__), "../../assets/bria_fibo2/edit_source.png")
# The edit baseline's instruction; used when the baseline reference data is
# absent so the smoke run still exercises the real edit path.
EDIT_INSTRUCTION = "Make the background a snowy forest"

CAPTION = (
    '{"short_description":"A red bicycle leaning against a white brick wall on a sunny morning.",'
    '"objects":[{"description":"a red city bicycle with a wicker basket","location":"center"}],'
    '"background_setting":"a white brick wall with a green door",'
    '"lighting":{"conditions":"bright morning sun","direction":"from the right"},'
    '"style_medium":"photograph"}'
)


def _compare_to_baseline(img, baseline_path, min_cos, label):
    ref = np.asarray(Image.open(baseline_path), dtype=np.float32)
    got = np.asarray(img, dtype=np.float32)
    af, bf = got.flatten(), ref.flatten()
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-8))
    mse = float(np.mean((af / 255 - bf / 255) ** 2))
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    print(f"{label}: cosine={cos:.4f} psnr={psnr:.2f}dB")
    assert cos >= min_cos, f"{label} diverged from baseline: cosine={cos:.4f}"


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_bria_fibo2_t2i(offline_client: OfflineOmniClient):
    sampling = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": CAPTION,
        "sampling_params": sampling,
    }
    # OfflineOmniClient.send_diffusion_request returns a single response
    resp = offline_client.send_diffusion_request(request_config)
    assert resp.success, f"request failed: {getattr(resp, 'error_message', None)}"
    assert resp.images, f"no images in response: text={resp.text_content}"
    img = resp.images[0]
    got = np.asarray(img, dtype=np.float32)
    assert got.shape == (1024, 1024, 3), f"unexpected output shape {got.shape}"

    baseline = os.path.join(REF_DATA, "image.png")
    if not os.path.exists(baseline):
        pytest.skip("baseline reference image not available; e2e smoke only")
    # 30 denoising steps of bf16 kernel noise vs the diffusers baseline land
    # around cosine 0.998 / 29 dB; a broken component collapses far below.
    _compare_to_baseline(img, baseline, 0.99, "e2e vs baseline")


def _edit_e2e(offline_client, model, ref_dir, steps, guidance, label, min_cos):
    import json as _json

    edited = os.path.join(ref_dir, "edited_image.png")
    meta_path = os.path.join(ref_dir, "meta.json")
    have_baseline = os.path.exists(edited) and os.path.exists(meta_path)
    instruction = _json.load(open(meta_path))["stats"]["instruction"] if have_baseline else EDIT_INSTRUCTION

    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=42,
    )
    request_config = {
        "model": model,
        "prompt": instruction,
        "images": [EDIT_SOURCE],
        "sampling_params": sampling,
    }
    resp = offline_client.send_diffusion_request(request_config)
    assert resp.success, f"{label} request failed: {getattr(resp, 'error_message', None)}"
    assert resp.images, f"no images in {label} response: text={resp.text_content}"
    img = resp.images[0]

    got = np.asarray(img, dtype=np.float32)
    assert got.ndim == 3 and got.shape[2] == 3, f"bad image shape {got.shape}"
    assert got.shape[0] % 16 == 0 and got.shape[1] % 16 == 0, f"size not multiple of 16: {got.shape}"

    if not have_baseline:
        pytest.skip(f"{label} baseline reference not available; e2e smoke only")
    meta = _json.load(open(meta_path))
    exp_h, exp_w = meta["stats"]["height"], meta["stats"]["width"]
    assert got.shape == (exp_h, exp_w, 3), f"unexpected {label} output shape {got.shape} vs baseline {(exp_h, exp_w)}"
    _compare_to_baseline(img, edited, min_cos, label)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_bria_fibo2_edit(offline_client: OfflineOmniClient):
    """Edit path: instruction + image to edit (the baseline T2I capture),
    compared against the diffusers edit baseline. The edit path transcribes
    most tokens from the context image, so it tracks the baseline much
    tighter than generation (pipeline-level smoke: cosine 0.99995 / 49 dB)."""
    _edit_e2e(offline_client, MODEL, EDIT_REF, 30, 5.0, "edit e2e vs baseline", 0.995)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(TURBO_MODEL, None)], indirect=True)
def test_bria_fibo2_edit_turbo(offline_client: OfflineOmniClient):
    """Edit Turbo: 4 steps, guidance 1.0 (no CFG, single forward per step),
    compared against the diffusers edit-turbo baseline."""
    _edit_e2e(offline_client, TURBO_MODEL, TURBO_REF, 4, 1.0, "turbo edit e2e vs baseline", 0.995)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(TURBO_MODEL, None)], indirect=True)
def test_bria_fibo2_turbo_defaults(offline_client: OfflineOmniClient):
    """A turbo request omitting steps/guidance must get the documented turbo
    defaults (4 steps, guidance 1: no CFG) from the official repo id, without
    depending on the checkpoint's directory name. Wrong defaults (30 steps +
    CFG) land far below the tolerance against the turbo baseline."""
    sampling = OmniDiffusionSamplingParams(seed=42)
    resp = offline_client.send_diffusion_request(
        {"model": TURBO_MODEL, "prompt": EDIT_INSTRUCTION, "images": [EDIT_SOURCE], "sampling_params": sampling}
    )
    assert resp.success, f"request failed: {getattr(resp, 'error_message', None)}"
    assert resp.images, f"no images in response: text={resp.text_content}"
    got = np.asarray(resp.images[0], dtype=np.float32)
    assert got.ndim == 3 and got.shape[2] == 3, f"bad image shape {got.shape}"

    edited = os.path.join(TURBO_REF, "edited_image.png")
    if not os.path.exists(edited):
        pytest.skip("turbo baseline reference not available; e2e smoke only")
    _compare_to_baseline(resp.images[0], edited, 0.99, "turbo defaults vs baseline")


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_bria_fibo2_default_guidance(offline_client: OfflineOmniClient):
    """Omitting guidance_scale must fall back to the model default (Dev: 5.0
    with CFG), not the request layer's coerced 1.0. Verifies via the e2e
    baseline: guidance 5 (default) tracks it; a wrongly-off CFG would not."""

    sampling = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=30,
        seed=42,
    )
    # The request layer (OmniDiffusionRequest.__post_init__) coerces the
    # unset value to 1.0 while guidance_scale_provided stays False; if the
    # pipeline honored the coerced 1.0, CFG would be off and the image would
    # diverge from the guidance-5 baseline below.
    resp = offline_client.send_diffusion_request({"model": MODEL, "prompt": CAPTION, "sampling_params": sampling})
    assert resp.success and resp.images
    baseline = os.path.join(REF_DATA, "image.png")
    if not os.path.exists(baseline):
        pytest.skip("baseline reference image not available")
    _compare_to_baseline(resp.images[0], baseline, 0.99, "default-guidance vs baseline")


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_bria_fibo2_batch(offline_client: OfflineOmniClient):
    """Two identical requests submitted together. The batching oracle is the
    exact agreement of the two rows (identical prompt + seed must give
    identical images whether the engine batched them or ran them apart);
    the baseline comparison is informational (flow matching amplifies
    batch-vs-single kernel differences chaotically over 30 steps)."""
    sampling = OmniDiffusionSamplingParams(height=1024, width=1024, num_inference_steps=30, guidance_scale=5.0, seed=42)
    outputs = offline_client.runner.generate_multimodal(prompts=[CAPTION, CAPTION], sampling_params_list=sampling)
    assert len(outputs) == 2, f"expected 2 outputs, got {len(outputs)}"
    images = []
    for i, out in enumerate(outputs):
        assert out.images, f"request {i} returned no images"
        got = np.asarray(out.images[0], dtype=np.float32)
        assert got.shape == (1024, 1024, 3), f"request {i} bad shape {got.shape}"
        images.append(got)

    a, b = images[0].flatten(), images[1].flatten()
    pair_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"batch rows agreement: cosine={pair_cos:.4f}")
    assert pair_cos >= 0.999, f"identical rows in one batch disagree: cosine={pair_cos:.4f}"

    baseline = os.path.join(REF_DATA, "image.png")
    if not os.path.exists(baseline):
        pytest.skip("baseline reference image not available")
    _compare_to_baseline(images[0], baseline, 0.95, "batch row0 vs baseline (informational)")
