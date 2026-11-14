# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Online-serving test for stacking two real LoRA adapters on one request.

Serves the public ``krea/Krea-2-Turbo`` checkpoint and posts a single
``/v1/images/generations`` request whose ``lora`` field is a **list** of two
adapters — the officially published ``krea/Krea-2-LoRA-vintagetarot`` and
``krea/Krea-2-LoRA-retroanime`` style LoRAs (real trained weights, raw
diffusers-format checkpoints).

The upstream repos ship only ``<name>.safetensors`` without a PEFT
``adapter_config.json`` and without the ``adapter_model.safetensors`` file name
the vLLM loader expects, so the fixture below materializes a local adapter
directory per LoRA: weights are copied byte-identical, only the config is
synthesized (rank read from the tensors, ``lora_alpha = r`` mirroring the
official darkbrush repackaging used by ``test_krea2_expansion.py``).

The test confirms:
- each adapter alone visibly changes the output (serving-path sanity),
- the stacked request succeeds and differs from baseline and from either
  single-adapter output (both adapters contribute),
- stacking order does not change the output (the manager composes adapters
  under an order-independent key).
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

MODEL = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
# Official Krea 2 style LoRAs (real trained adapters, diffusers format).
LORA_A_REPO = os.environ.get("KREA2_LORA_A", "krea/Krea-2-LoRA-vintagetarot")
LORA_B_REPO = os.environ.get("KREA2_LORA_B", "krea/Krea-2-LoRA-retroanime")

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})

PROMPT = "a fox in the snow"


def _materialize_adapter(repo_id: str, dest: Path) -> Path:
    """Download a raw diffusers LoRA and repackage it as a PEFT adapter dir.

    Only the metadata is synthesized; the safetensors payload is used as-is.
    ``hf_hub_download(local_dir=...)`` keeps every write inside ``dest`` so the
    test never touches the (possibly read-only or full) shared HF cache.
    """
    dest.mkdir(parents=True, exist_ok=True)
    filename = f"{repo_id.rsplit('-', 1)[-1]}.safetensors"
    downloaded = Path(hf_hub_download(repo_id, filename, local_dir=str(dest)))
    weights_path = dest / "adapter_model.safetensors"
    if not weights_path.exists():
        downloaded.rename(weights_path)

    with safe_open(str(weights_path), framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(".lora_A.weight")]
        rank = f.get_slice(keys[0]).get_shape()[0]
    target_modules = sorted({k[: -len(".lora_A.weight")] for k in keys})
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL,
        "r": rank,
        "lora_alpha": rank,
        "bias": "none",
        "target_modules": target_modules,
    }
    (dest / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return dest


@pytest.fixture(scope="module")
def stacked_lora_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("krea2_stacked_loras")
    return (
        _materialize_adapter(LORA_A_REPO, root / "vintagetarot"),
        _materialize_adapter(LORA_B_REPO, root / "retroanime"),
    )


def _lora_payload(lora: dict | list[dict] | None = None) -> dict:
    payload = {
        "prompt": PROMPT,
        "n": 1,
        "size": "512x512",
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
        "seed": 42,
    }
    if lora is not None:
        payload["lora"] = lora
    return payload


def _post_image(server: OmniServer, payload: dict) -> np.ndarray:
    url = f"http://{server.host}:{server.port}/v1/images/generations"
    resp = requests.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=900)
    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return np.asarray(img.convert("RGB"), dtype=np.int16)


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            # Default adapter cache (max_cpu_loras=1): stacked requests must
            # work without raising the cache because sources are loaded
            # transiently for composition.
            OmniServerParams(model=MODEL, server_args=["--enforce-eager"]),
            id="single_card_multi_lora",
            marks=SINGLE_CARD_FEATURE_MARKS,
        )
    ],
    indirect=True,
)
def test_krea2_stacked_loras(omni_server: OmniServer, stacked_lora_dirs: tuple[Path, Path]):
    """Stack two real Krea 2 LoRAs in one request via ``lora: [{...}, {...}]``."""
    lora_a_dir, lora_b_dir = stacked_lora_dirs
    lora_a = {"name": "vintagetarot", "path": str(lora_a_dir), "scale": 1.0}
    lora_b = {"name": "retroanime", "path": str(lora_b_dir), "scale": 1.0}

    baseline = _post_image(omni_server, _lora_payload())
    img_a = _post_image(omni_server, _lora_payload(lora_a))
    img_b = _post_image(omni_server, _lora_payload(lora_b))
    stacked = _post_image(omni_server, _lora_payload([lora_a, lora_b]))
    stacked_reversed = _post_image(omni_server, _lora_payload([lora_b, lora_a]))

    def diff(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.abs(x - y).mean())

    # Each adapter alone has a visible effect over the serving path.
    assert diff(baseline, img_a) > 0.5, f"LoRA A had no visible effect: diff={diff(baseline, img_a)}"
    assert diff(baseline, img_b) > 0.5, f"LoRA B had no visible effect: diff={diff(baseline, img_b)}"

    # The stacked request must reflect both adapters: it matches neither the
    # base output nor either single-adapter output.
    assert diff(baseline, stacked) > 0.5, f"stacked LoRAs had no visible effect: diff={diff(baseline, stacked)}"
    assert diff(img_a, stacked) > 0.5, f"stacked output equals LoRA A only: diff={diff(img_a, stacked)}"
    assert diff(img_b, stacked) > 0.5, f"stacked output equals LoRA B only: diff={diff(img_b, stacked)}"

    # The composed adapter is keyed order-independently: [A, B] and [B, A]
    # activate the same composed weights for the same seed.
    diff_order = diff(stacked, stacked_reversed)
    assert diff_order < 5.0, f"stacked LoRA output depends on adapter order: diff={diff_order:.2f}"
