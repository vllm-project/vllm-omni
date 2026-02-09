import re

from .types import Modality, ModelMode, Spec


def _bagel_payload_preprocessor(payload: dict) -> dict:
    try:
        for message in payload["messages"]:
            for content in message["content"]:
                if content["type"] == "text":
                    content["text"] = "<|im_start|>" + content["text"] + "<|im_end|>"
    except (KeyError, TypeError):
        raise RuntimeError("Internal Error: malformatted BAGEL payload")
    extra_body = payload.pop("extra_body", {})
    return {**payload, **extra_body}


def _qwen25_payload_preprocessor(payload: dict) -> dict:
    if payload["messages"][0]["role"] != "system":
        payload["messages"] = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                ),
            },
            *payload["messages"],
        ]
    return payload


_MODEL_PIPELINE_SPECS: dict[str, Spec] = {
    r"BAGEL-7B-MoT": {
        "stages": [
            "diffusion"  # The vLLM-Omni interface treats it as a single-stage diffusion model
        ],
        "modes": [
            {
                "mode": ModelMode.COMPREHENSION,
                "input_modalities": [Modality.TEXT, Modality.IMAGE],
            }
        ],
        "payload_preprocessor": _bagel_payload_preprocessor,
    },
    r"Qwen2.5-Omni*": {
        "stages": ["autoregression", "autoregression", "autoregression"],
        "payload_preprocessor": _qwen25_payload_preprocessor,
        "modes": [
            {
                "mode": ModelMode.COMPREHENSION,
                "input_modalities": [
                    Modality.TEXT,
                    Modality.IMAGE,
                    Modality.VIDEO,
                    Modality.AUDIO,
                ],
            }
        ],
    },
    r"Qwen3-Omni*": {
        "stages": ["autoregression", "autoregression", "autoregression"],
        "modes": [
            {
                "mode": ModelMode.COMPREHENSION,
                "input_modalities": [
                    Modality.TEXT,
                    Modality.IMAGE,
                    Modality.VIDEO,
                    Modality.AUDIO,
                ],
            }
        ],
    },
}
# Convert dict keys to regex patterns
MODEL_PIPELINE_SPECS: dict[re.Pattern, Spec] = {}
for k, v in _MODEL_PIPELINE_SPECS.items():
    MODEL_PIPELINE_SPECS[re.compile(k)] = v
del _MODEL_PIPELINE_SPECS


def lookup_model_spec(model: str) -> tuple[Spec | None, str | None]:
    try:
        last_component = model.rstrip("/").rsplit("/", 1)[-1]
    except IndexError:
        last_component = model
    for pattern, spec in MODEL_PIPELINE_SPECS.items():
        if pattern.search(last_component):
            return spec, pattern.pattern
    return None, None


# ============== DEMONSTRATION ==============

if __name__ == "__main__":
    test_paths = [
        "Qwen/Qwen2.5-Omni-7B",
        "MyModels/Qwen2.5-Omni-3B",
        "/root/home/Qwen2.5-Omni-7B",
        "Qwen/Qwen3-Omni",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Custom/Path/UnknownModel-Instruct",
        "Not/Matching/Anything",
    ]

    test_payload = {"messages": [{"role": "user", "content": "prompt"}]}

    print("Testing registry lookups:\n")
    for path in test_paths:
        spec, _ = lookup_model_spec(path)
        if spec:
            if preprocessor := spec.get("payload_preprocessor"):
                result = preprocessor(test_payload)
                print(f"✓ {path:<40} → {result}")
            else:
                print(f"✓ {path:<40} → No preprocessor")
        else:
            print(f"✗ {path:<40} → No match")
