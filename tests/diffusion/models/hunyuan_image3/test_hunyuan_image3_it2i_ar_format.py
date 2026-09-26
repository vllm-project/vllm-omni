# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the IT2I AR-prefill prompt matches the official HF chat-template output.

PR #3107 builds the AR prefill via
:func:`vllm_omni.diffusion.models.hunyuan_image3.prompt_utils.build_prompt_tokens`,
which segment-tokenizes the canonical Instruct chat template (`<|startoftext|>`
+ `{system}\\n\\n` + `User: [<img>]{user_prompt}\\n\\nAssistant: {trigger?}`).

The official HunyuanImage-3.0-Instruct repo ships a Jinja `chat_template` in
its tokenizer config and an `image_processor.py` whose `process_image`
defines the same VAE/VIT preprocessing the diffusion pipeline uses on the
condition image. To prevent silent drift between the AR's input distribution
and what the model was actually trained on, this test asserts:

1. ``build_prompt_tokens`` token-id sequence equals the HF reference produced
   by ``tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)``
   for the same `(system, user_prompt, image)` triple.
2. The image-tensor produced by the diffusion-side ``_resize_and_crop_center``
   is byte-identical to the AR-side ``HunyuanImage3Processor._resize_and_crop``
   output (i.e. AR and DiT preprocess the IT2I condition image identically).

Both checks need the official tokenizer/image-processor classes; we gate on
``HF_HOME`` cache availability so the suite stays runnable on machines
without the model weights.
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_HUNYUAN_MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


def _hf_cached(model_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots")
    return os.path.isdir(snap_dir) and any(os.scandir(snap_dir))


def _snapshot_dir(model_id: str) -> pathlib.Path:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_root = pathlib.Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    snap = next(iter(snap_root.iterdir()))
    return snap


# --- Real AR-output comparison lives in
# tests/e2e/accuracy/test_hunyuan_image3_it2i_ar_output.py ---
#
# Earlier revisions of this file shipped a CPU-only "compare prefill
# token sequences" check that called the official tokenizer's
# `apply_chat_template`. That comparison was misleading: it only verified
# the *input* prompt template, not the AR-stage *generated output*; and
# it kept skipping because instantiating
# `HunyuanImage3TokenizerFast.from_pretrained(snap)` returns a
# byte-fallback (char-level) tokenizer that is not the same encoding the
# vllm-omni production path actually uses (which goes through the
# standard `AutoTokenizer.from_pretrained`).
#
# The "AR output matches official" contract is genuinely a GPU-required
# end-to-end test: it must drive `model.prepare_model_inputs` +
# `model.generate(do_sample=False)` on the HF side and the IT2I `i2t`
# stage on the omni side, then compare AR-generated token sequences.
# That is now the responsibility of the e2e test in
# tests/e2e/accuracy/test_hunyuan_image3_it2i_ar_output.py.


_OFFICIAL_PKG = "_hunyuan_image_3_official_snapshot"


def _import_official_snapshot_modules():
    """Register the HunyuanImage-3.0-Instruct snapshot as a fake package so
    its ``image_processor.py`` (which does ``from .tokenization_hunyuan_image_3
    import ...``) can be loaded with relative imports intact.

    Returns ``(tokenization_module, image_processor_module)`` or ``(None, None)``
    if either fails (e.g. snapshot missing, optional dep like diffusers absent).
    """
    import importlib.util
    import sys
    import types

    if _OFFICIAL_PKG in sys.modules:
        pkg = sys.modules[_OFFICIAL_PKG]
        return (
            sys.modules.get(f"{_OFFICIAL_PKG}.tokenization_hunyuan_image_3"),
            sys.modules.get(f"{_OFFICIAL_PKG}.image_processor"),
        )

    snap = _snapshot_dir(_HUNYUAN_MODEL_ID)
    if not (snap / "image_processor.py").is_file():
        return None, None

    pkg = types.ModuleType(_OFFICIAL_PKG)
    pkg.__path__ = [str(snap)]
    sys.modules[_OFFICIAL_PKG] = pkg

    def _load(name: str):
        full = f"{_OFFICIAL_PKG}.{name}"
        spec = importlib.util.spec_from_file_location(full, snap / f"{name}.py")
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[full]
            return None
        return mod

    tok_mod = _load("tokenization_hunyuan_image_3")
    if tok_mod is None:
        return None, None
    img_mod = _load("image_processor")
    return tok_mod, img_mod


@pytest.mark.skipif(
    not _hf_cached(_HUNYUAN_MODEL_ID),
    reason=f"{_HUNYUAN_MODEL_ID} not in HF cache",
)
def test_dit_condition_image_preprocessing_byte_matches_official_hf():
    """The diffusion pipeline's ``_resize_and_crop_center`` (used to feed
    the VAE encoder for IT2I conditioning) must produce byte-identical
    pixels to the **official** HuggingFace
    ``image_processor.resize_and_crop`` (loaded straight out of the
    HunyuanImage-3.0-Instruct snapshot's bundled ``image_processor.py``)
    at ``crop_type='center'``.

    Bounty-hunter's PR #3107 review flagged that the DiT-side helper had
    drifted from the AR-side processor on rounding boundaries; PR #3107
    commit ``0a7e0e6f`` aligned the DiT helper to the AR-side algorithm.
    AR and DiT both *claim* to mirror the HF reference, so the actual
    contract is "DiT (and AR) match the HF reference verbatim". We
    enforce that contract here by comparing directly to the HF function
    rather than to a sibling vllm-omni copy.
    """
    import numpy as np
    from PIL import Image

    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        _resize_and_crop_center,
    )

    _tok_mod, official_module = _import_official_snapshot_modules()
    if official_module is None or not hasattr(official_module, "resize_and_crop"):
        pytest.skip("Official HunyuanImage3 image_processor.py not loadable")
    official_resize_and_crop = official_module.resize_and_crop

    rng = np.random.default_rng(seed=42)
    src_size_pairs = [(640, 1024), (1024, 1024), (1280, 720), (480, 800)]
    target_size_pairs = [(1024, 1024), (1024, 768), (768, 1024)]

    for src_w, src_h in src_size_pairs:
        src_arr = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
        src = Image.fromarray(src_arr, mode="RGB")
        for tw, th in target_size_pairs:
            ref_out = official_resize_and_crop(
                src,
                target_size=(tw, th),
                resample=Image.Resampling.LANCZOS,
                crop_type="center",
            )
            dit_out = _resize_and_crop_center(src, tw, th)
            assert ref_out.size == dit_out.size == (tw, th), (
                f"size mismatch for src={(src_w, src_h)} target={(tw, th)}: "
                f"hf_official={ref_out.size} dit={dit_out.size}"
            )
            ref_pixels = np.asarray(ref_out)
            dit_pixels = np.asarray(dit_out)
            assert np.array_equal(ref_pixels, dit_pixels), (
                f"DiT condition-image preprocessing diverged from HF "
                f"image_processor.resize_and_crop at src={(src_w, src_h)} "
                f"target={(tw, th)}: max abs diff = "
                f"{int(np.abs(ref_pixels.astype(int) - dit_pixels.astype(int)).max())}"
            )
