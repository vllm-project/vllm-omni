import os
import sys
import time
import json
import functools

import torch

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)

# Env switches
_ENABLE_PROFILING = os.getenv("QWEN_PROFILE_ENABLE", "").lower() in (
    "1", "true", "yes"
)

_PROFILE_PATH = os.getenv("QWEN_PROFILE_JSON", "")

print(
    f"[sitecustomize] loaded, pid={os.getpid()}, profiling={_ENABLE_PROFILING}"
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("QWEN_PROFILE_ENABLE", "1")
profile_json = os.path.join(BASE_DIR, "file.jsonl")
os.environ.setdefault("QWEN_PROFILE_JSON", profile_json)

print("[env setup]")
print(" QWEN_PROFILE_ENABLE =", os.environ["QWEN_PROFILE_ENABLE"])
print(" QWEN_PROFILE_JSON   =", os.environ["QWEN_PROFILE_JSON"])


# Timing helpers
def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def log_record(record: dict) -> None:
    if not _ENABLE_PROFILING or not _PROFILE_PATH:
        return

    record["ts"] = time.time()
    record["pid"] = os.getpid()

    os.makedirs(os.path.dirname(_PROFILE_PATH), exist_ok=True)
    with open(_PROFILE_PATH, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

# Monkey-patch helper
def patch_method(cls, method_name: str, wrapper_fn):
    if not _ENABLE_PROFILING:
        return
    if cls is None or not hasattr(cls, method_name):
        return

    orig = getattr(cls, method_name)
    if hasattr(orig, "__patched__"):
        return

    wrapped = wrapper_fn(orig)
    wrapped.__patched__ = True
    setattr(cls, method_name, wrapped)


# Patch logic
def _patch():
    if not _ENABLE_PROFILING:
        print("[sitecustomize] profiling disabled, skip patch")
        return

    print("[sitecustomize] start patching Qwen-Image hooks")

    # 1. Text Encoder: encode_prompt -> measure latency + tokens/s
    def wrap_encode_prompt(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, *args, **kwargs):
            prompt = None
            if args:
                prompt = args[0]
            else:
                prompt = kwargs.get("prompt")

            # Token counting (re-tokenize, negligible overhead)
            num_tokens = 0
            if prompt is not None and hasattr(self, "tokenizer"):
                try:
                    if isinstance(prompt, str):
                        toks = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                        )
                        num_tokens = toks.input_ids.shape[-1]
                    elif isinstance(prompt, list):
                        toks = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                        num_tokens = toks.input_ids.shape[-1] * len(prompt)
                except Exception:
                    num_tokens = 0

            t0 = sync_time()
            out = orig_fn(self, *args, **kwargs)
            dt = sync_time() - t0

            tokens_per_sec = num_tokens / dt if dt > 0 else 0.0

            log_record(
                {
                    "stage": "text_encoding",
                    "time": dt,
                    "num_tokens": num_tokens,
                    "tokens_per_second": tokens_per_sec,
                }
            )
            return out

        return wrapper

    patch_method(QwenImagePipeline, "encode_prompt", wrap_encode_prompt)

    # 2. transformer.forward 
    _denoise_state = {
        "step": 0,
        "total_time": 0.0,
    }

    def wrap_transformer_forward(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, *args, **kwargs):
            step = _denoise_state["step"]
            _denoise_state["step"] += 1

            t0 = sync_time()
            out = orig_fn(self, *args, **kwargs)
            dt = sync_time() - t0

            _denoise_state["total_time"] += dt

            log_record(
                {
                    "stage": "denoise_step",
                    "step": step,
                    "time": dt,
                }
            )
            return out

        return wrapper

    patch_method(QwenImageTransformer2DModel, "forward", wrap_transformer_forward)

    # Estimate the number of image tokens
    def _estimate_image_tokens(pipeline: QwenImagePipeline, req) -> dict:
        """
            Based on the height/width, vae_scale_factor of QwenImagePipeline, and the prompt/num_outputs_per_prompt in the request, estimate:
            ▪ The number of tokens per image

            ▪ The total number of images

            ▪ The total number of tokens

        """
        # infer batch size
        prompt = getattr(req, "prompt", None)
        if isinstance(prompt, str) or prompt is None:
            batch_size = 1
        elif isinstance(prompt, (list, tuple)):
            batch_size = len(prompt)
        else:
            batch_size = 1

        # prompt==>images
        num_images_per_prompt = getattr(req, "num_outputs_per_prompt", 1) or 1
        total_images = batch_size * num_images_per_prompt

        height = getattr(req, "height", None) or pipeline.default_sample_size * pipeline.vae_scale_factor
        width = getattr(req, "width", None) or pipeline.default_sample_size * pipeline.vae_scale_factor

        patch_size = pipeline.vae_scale_factor * 2

        h_tokens = int(height) // patch_size
        w_tokens = int(width) // patch_size
        tokens_per_image = max(h_tokens * w_tokens, 0)

        total_tokens = tokens_per_image * total_images

        return {
            "batch_size": batch_size,
            "num_images_per_prompt": num_images_per_prompt,
            "total_images": total_images,
            "tokens_per_image": tokens_per_image,
            "total_tokens": total_tokens,
            "height": int(height),
            "width": int(width),
        }

    # 3. Pipeline forward image token/s
    def wrap_pipeline_forward(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, req):
            _denoise_state["step"] = 0
            _denoise_state["total_time"] = 0.0

            t0 = sync_time()
            out = orig_fn(self, req)
            total_time = sync_time() - t0

            steps = _denoise_state["step"]
            denoise_total = _denoise_state["total_time"]
            avg_step = denoise_total / steps if steps > 0 else 0.0

            img_stats = _estimate_image_tokens(self, req)
            total_tokens = img_stats["total_tokens"] * steps


            tokens_per_sec_denoise = total_tokens / denoise_total if denoise_total > 0 else 0.0
            tokens_per_sec_pipeline = total_tokens / total_time if total_time > 0 else 0.0

            log_record(
                {
                    "stage": "diffusion",
                    "num_steps": steps,
                    "denoise_total": denoise_total,
                    "denoise_step_avg": avg_step,
                    "pipeline_forward": total_time,
                    "height": img_stats["height"],
                    "width": img_stats["width"],
                    "batch_size": img_stats["batch_size"],
                    "num_images_per_prompt": img_stats["num_images_per_prompt"],
                    "total_images": img_stats["total_images"],
                    "tokens_per_image": img_stats["tokens_per_image"],
                    "total_image_tokens": total_tokens,
                    "image_tokens_per_second_denoise": tokens_per_sec_denoise,
                    "image_tokens_per_second_pipeline": tokens_per_sec_pipeline,
                }
            )
            return out

        return wrapper

    patch_method(QwenImagePipeline, "forward", wrap_pipeline_forward)

    # 4. VAE Decode
    def wrap_vae_decode(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, *args, **kwargs):
            t0 = sync_time()
            out = orig_fn(self, *args, **kwargs)
            dt = sync_time() - t0

            log_record(
                {
                    "stage": "vae_decode",
                    "time": dt,
                }
            )
            return out

        return wrapper

    patch_method(AutoencoderKLQwenImage, "decode", wrap_vae_decode)

    print("[sitecustomize] patch done")

_patch()