# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROTOCOL = "cwm_causal/v1"
BUNDLE_SCHEMA = "h3_omni_native/v1"
INVOKER_RENDER_PIPELINE_CLASS_NAME = "InvokerRenderPipeline"
VIDEO_SIZE = (1344, 768)
FPS = 24
PACKET_FRAMES = 17
TAIL_FRAMES = 5


def is_invoker_render_bundle(model: str) -> bool:
    manifest_path = Path(model) / "native-runtime.json"
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open() as source:
            manifest = json.load(source)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(manifest, Mapping):
        return False
    recipe = manifest.get("recipe")
    return (
        manifest.get("schema") == BUNDLE_SCHEMA and isinstance(recipe, Mapping) and recipe.get("protocol") == PROTOCOL
    )


def _file(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Missing Invoker-Render bundle file: {path}")
    return path


def _directory(value: str) -> str:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"Missing Invoker-Render component directory: {path}")
    return str(path)


@dataclass(frozen=True)
class InvokerRenderAssets:
    dit_dir: str
    text_encoder: str
    processor: str
    video_vae: str
    decoder_config: Path
    decoder_weights: Path
    system_prompt: str
    proxy_size: tuple[int, int]
    proxy_resize: str

    @classmethod
    def read(cls, model: str) -> "InvokerRenderAssets":
        if not is_invoker_render_bundle(model):
            raise ValueError("Invoker-Render requires a prepared cwm_causal/v1 bundle")
        with (Path(model) / "native-runtime.json").open() as source:
            manifest = json.load(source)
        recipe = manifest["recipe"]
        sampler = recipe["diffusion"]["sampling_timesteps"]["args"]
        if (
            recipe["conditioning_mode"] != "static_image_qwen_proxy_vae_v1"
            or recipe["position_policy"] != "cwm_frame_aligned_v1"
            or recipe["sink_latents"] != 1
            or recipe["recent_latents"] != 12
            or recipe["bootstrap_latents"] != 9
            or recipe["current_latent_schedule"] != [5]
            or recipe["segment_boundaries"] != [45, 70]
            or recipe["rope_reference_latents"] != 47
            or sampler != {"T": 1.0, "num_sampling_steps": 4, "shift": 12.0}
            or recipe["diffusion"]["sampler"]["class_name"] != "EulerSampler"
            or recipe["guidance"] != {"mode": "plain", "drop": "proxy", "scale": 1.0}
        ):
            raise ValueError("Invoker-Render bundle uses a different CWM sampling recipe")
        canvas = manifest["canvas"]
        proxy_size = (canvas["proxy_width"], canvas["proxy_height"])
        if (
            (canvas["width"], canvas["height"]) != VIDEO_SIZE
            or any(type(size) is not int or size <= 0 or size % 32 for size in proxy_size)
            or proxy_size[0] * 4 != proxy_size[1] * 7
        ):
            raise ValueError("Invalid Invoker-Render output or proxy canvas")
        cwm, codec = manifest["cwm"], manifest["codec"]
        resize = cwm["proxy_resize"]
        if (
            cwm["posterior"] != "mean"
            or (cwm["proxy_encode_width"], cwm["proxy_encode_height"]) != VIDEO_SIZE
            or resize not in ("nearest", "lanczos")
            or codec["encoder"] != "h3"
            or codec["decoder"] != "causal"
            or codec["window"] != 16
        ):
            raise ValueError("Invoker-Render requires full-size H3 mean encoding and the 16-token causal decoder")
        with _file(manifest["recipe_selection"]["source_artifact"]).open() as source:
            artifact = json.load(source)
        prompt = artifact["encoders"]["recipe"]["system_prompt"].strip()
        if not prompt:
            raise ValueError("Invoker-Render artifact must contain a frozen prompt")

        server = manifest["server_args"]
        index = _file(server["transformer_weights_path"])
        if index.name != "diffusion_pytorch_model.safetensors.index.json":
            raise ValueError("Invoker-Render requires the released Diffusers DiT shard index")
        with index.open() as source:
            shards = set(json.load(source)["weight_map"].values())
        if not shards or any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
            raise ValueError("Invalid Invoker-Render DiT shard index")
        for shard in shards:
            _file(str(index.parent / shard))
        _file(str(index.parent / "config.json"))
        if not (Path(model) / "transformer" / "config.json").is_file():
            raise ValueError("Invoker-Render bundle requires transformer/config.json")
        components = server["component_paths"]
        return cls(
            str(index.parent),
            _directory(components["text_encoder"]),
            _directory(components["processor"]),
            _directory(components["video_vae"]),
            _file(codec["decoder_config"]),
            _file(codec["decoder_weights"]),
            prompt,
            proxy_size,
            resize,
        )
