# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end online-serving smoke test for Alpamayo-1.5 (chat completions API).

Launches ``vllm-omni serve`` and posts a multi-camera + ego-history chat request,
exercising the full online stack: the baked extended tokenizer, the Qwen3-VL
image preprocess, ``vllm_xargs`` -> ``extra_args`` propagation (robot_obs
delivery), server-side history fusion, AR chain-of-thought generation and the
inline flow-matching trigger.

Scope note: this asserts the server returns a 200 with non-empty chain-of-thought
reasoning AND that the predicted trajectory survives in the response body under
``message.multimodal_output["actions"]`` (preserved by
``OmniChatCompletionResponse``'s ``model_serializer``, since the base
``ChatCompletionResponseChoice.message`` schema would otherwise drop the extra
field). The numerical minADE-vs-GT check lives in the offline test
(``tests/e2e/offline_inference/test_alpamayo.py``), which reads
``ro.multimodal_output["actions"]`` directly.

Equivalent to running the example:
    vllm-omni serve "$ALPAMAYO_WEIGHTS" --omni --tokenizer "$ALPAMAYO_TOKENIZER_DIR" ...
    python3 examples/online_serving/alpamayo/http_client.py

Marked ``local_model``: Alpamayo-1.5 weights are gated and the multi-camera clip
``.pkl`` is not in the repo, so this only runs locally with both provided
(``ALPAMAYO_CLIP_PKL`` set), never in the merge CI lane.
"""

import base64
import io
import json
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("ALPAMAYO_MODEL", "nvidia/Alpamayo-1.5-10B")
VLM_BASE = os.environ.get("ALPAMAYO_VLM_BASE", "Qwen/Qwen3-VL-8B-Instruct")
STAGE_CONFIG = get_deploy_config_path("alpamayo1_5.yaml")
CLIP_PKL = os.environ.get("ALPAMAYO_CLIP_PKL")
N_SAMPLES = int(os.environ.get("ALPAMAYO_N_SAMPLES", "4"))
# Baked extended tokenizer (Qwen3-VL base + Alpamayo traj tokens). The chat
# endpoint renders prompts with its own tokenizer, so it must know the trajectory
# tokens (e.g. the <|traj_history|> placeholders) — hence a disk-baked tokenizer.
TOKENIZER_DIR = os.environ.get("ALPAMAYO_TOKENIZER_DIR", "/tmp/alpamayo_online_e2e_tokenizer")

_SYSTEM = "You are a driving assistant that generates safe and accurate actions."
_INSTRUCTION = "output the chain-of-thought reasoning of the driving process, then output the future trajectory"
_VISION_PH = "<|vision_start|><|image_pad|><|vision_end|>"
_HISTORY_BLOCK = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
_CAMERA_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
# Passthrough chat template: collapse message content back to the raw prompt
# string we already formatted (no extra role markers).
_CHAT_TEMPLATE = "{% for m in messages %}{% for c in m.content %}{% if c.type == 'text' %}{{ c.text }}{% endif %}{% endfor %}{% endfor %}"

_SERVER_ARGS = [
    "--tokenizer",
    TOKENIZER_DIR,
    "--trust-remote-code",
    "--trust-request-chat-template",
    "--dtype",
    "bfloat16",
    "--enforce-eager",
    "--max-model-len",
    "32768",
    "--gpu-memory-utilization",
    "0.6",
    "--limit-mm-per-prompt",
    '{"image": 16}',
]

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        server_args=_SERVER_ARGS,
        stage_init_timeout=600,
    ),
]

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.omni,
    pytest.mark.skipif(
        not CLIP_PKL,
        reason="Set ALPAMAYO_CLIP_PKL to a clip .pkl (multi-cam frames + ego history).",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _bake_tokenizer():
    """Bake the extended tokenizer to TOKENIZER_DIR before the server starts."""
    if not CLIP_PKL:
        yield
        return
    from transformers import AutoProcessor

    from vllm_omni.model_executor.models.alpamayo.processing import add_alpamayo_tokens

    processor = AutoProcessor.from_pretrained(VLM_BASE, trust_remote_code=True)
    add_alpamayo_tokens(processor.tokenizer)
    processor.save_pretrained(TOKENIZER_DIR)
    yield


def _encode_image(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_prompt(camera_indices, num_frames_per_camera: int) -> str:
    parts = []
    for cam_id in camera_indices:
        parts.append(f"{_CAMERA_NAMES.get(int(cam_id), f'Camera {cam_id}')}: ")
        for frame_idx in range(num_frames_per_camera):
            parts.append(f"frame {frame_idx} {_VISION_PH}")
    cam_block = "".join(parts)
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{cam_block}{_HISTORY_BLOCK}{_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>assistant\n<|cot_start|>"
    )


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_alpamayo_online_smoke(omni_server: OmniServer, openai_client) -> None:
    """Server accepts a multi-cam + robot_obs request and returns CoT reasoning."""
    import pandas as pd
    from PIL import Image

    data = pd.read_pickle(CLIP_PKL)
    frames = data["image_frames"].flatten(0, 1)
    pil_images = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]
    cam_ids = data["camera_indices"].tolist()
    num_frames_per_camera = int(frames.shape[0] // len(cam_ids))
    prompt = _build_prompt(cam_ids, num_frames_per_camera)

    hx = data["ego_history_xyz"]
    hr = data["ego_history_rot"]
    if hx.ndim == 3:
        hx = hx.unsqueeze(0)
        hr = hr.unsqueeze(0)
    robot_obs = {"ego_history_xyz": hx.tolist(), "ego_history_rot": hr.tolist()}

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(i)}"}} for i in pil_images
    ]
    user_content.append({"type": "text", "text": prompt})

    # Raw HTTP (not the typed OpenAI client) so the response's non-standard
    # ``multimodal_output`` field survives — mirrors examples/.../http_client.py.
    payload = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 400,
        "temperature": 0.6,
        "top_p": 0.98,
        "chat_template": _CHAT_TEMPLATE,
        # vllm_xargs values must be flat primitives -> robot_obs as JSON string.
        "vllm_xargs": {"robot_obs": json.dumps(robot_obs), "n_samples": N_SAMPLES},
    }
    resp = openai_client.send_chat_completions_http_request({"json": payload})[0]
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.json_body}"

    choices = resp.json_body["choices"]
    assert choices, "No choices in response"
    message = choices[0]["message"]
    text = (message.get("reasoning") or "") + (message.get("content") or "")
    assert text.strip(), "Empty chain-of-thought response"

    # The sampled trajectory rides on multimodal_output["actions"] — preserved
    # through serialization by OmniChatCompletionResponse's model_serializer.
    mm = message.get("multimodal_output")
    assert isinstance(mm, dict) and "actions" in mm, f"actions missing from response: keys={list(message)}"
    actions = mm["actions"]  # nested list, shape (n_samples, n_waypoints=64, action_dim=2)
    assert len(actions) == N_SAMPLES, f"expected {N_SAMPLES} samples, got {len(actions)}"
    assert len(actions[0]) == 64 and len(actions[0][0]) == 2, "unexpected action trajectory shape"
