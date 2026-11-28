# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Direct vLLM-Omni policy API client for the official LIBERO evaluator.

Images and captions are prepared by the evaluator. This module returns physical
10D actions; the evaluator retains ownership of rotation and gripper conversion.
No local model weights or OpenPI runtime are required.
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from vllm_omni.entrypoints.cosmos3.action_policy_common import DEFAULT_LIBERO_STATS, LiberoActionNormalizer
from vllm_omni.entrypoints.cosmos3.action_ready_stream import BASELINE_PROFILE, POLICY_PROFILE
from vllm_omni.entrypoints.cosmos3.libero_policy_inputs import prepare_libero_policy_inputs
from vllm_omni.entrypoints.cosmos3.moh_timing import model_infer_ms

logger = logging.getLogger(__name__)


# 父类
class VllmOmniActionTransport:
    """Transport prepared pixels/captions to vLLM and return normalized actions."""

    # 初始化推理客户端
    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.0,
        flow_shift: float = 5.0,
        seed: int = 0,
        timeout: float = 1800.0,
        poll_interval: float = 0.5,
        action_time_schedule: str = "has_euler",  # ours
        early_stop_actions: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        for name, value in (("guidance_scale", guidance_scale), ("flow_shift", flow_shift), ("timeout", timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if num_inference_steps != 10 or guidance_scale != 1:
            raise ValueError("The native LIBERO10 reproduction profile requires 10 steps and guidance_scale=1.")
        if not math.isfinite(poll_interval) or poll_interval < 0:
            raise ValueError("poll_interval must be nonnegative and finite.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.flow_shift = flow_shift
        self.seed = seed
        self.timeout = timeout
        self.poll_interval = poll_interval
        if action_time_schedule not in {"has_euler", "constant_euler"}:
            raise ValueError("action_time_schedule must be has_euler or constant_euler")
        self.action_time_schedule = action_time_schedule
        if early_stop_actions is not None and (
            type(early_stop_actions) is not int or not 1 <= early_stop_actions <= 16
        ):
            raise ValueError("early_stop_actions must be an integer between 1 and 16")
        self.early_stop_actions = early_stop_actions
        self.policy_profile = POLICY_PROFILE if action_time_schedule == "has_euler" else BASELINE_PROFILE
        self.session = session or requests.Session()
        if session is None:
            # Calls address the configured model server directly, including localhost.
            self.session.trust_env = False

    def check_health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # 获取模型和推理配置
    def get_info(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
        response.raise_for_status()
        models = [item["id"] for item in response.json()["data"]]
        if not models:
            raise RuntimeError("vLLM-Omni returned no served models.")
        if self.model is None:
            self.model = models[0]
        if self.model not in models:
            raise ValueError(f"Requested model {self.model!r} is not served; available models: {models!r}.")
        return {
            "backend": "vllm-online",
            "server_url": self.base_url,
            "model": self.model,
            "num_steps": self.num_inference_steps,
            "guidance": self.guidance_scale,
            "flow_shift": self.flow_shift,
            "seed": self.seed,
            "action_chunk_size": 16,
            "early_stop_actions": self.early_stop_actions,
            "raw_action_dim": 10,
            "fps": 20,
            "action_normalization": "quantile_rot",
            "action_time_schedule": self.action_time_schedule,
            "faster_alpha": 0.6,
            "faster_u0": 0.9,
            "faster_formula": "paper_v3",
            "policy_profile": self.policy_profile,
            "noise_rng": "numpy_randomstate_per_modality",
            "sampler_dtype": "float32",
            "measure_ttfa": False,
            "streaming": {"supported": False},
            "batch_execution": "sequential_online_jobs",
        }

    # 解析模型生成的动作
    @staticmethod
    def _decode_action(payload: Any, action_count: int = 16) -> np.ndarray:
        if not isinstance(payload, dict) or "data" not in payload or "shape" not in payload:
            raise ValueError("vLLM-Omni job must return an action object containing data and shape.")
        dtype = str(payload.get("dtype", "float32")).removeprefix("torch.")
        if dtype not in {"float", "float16", "float32", "float64", "bfloat16", "half", "double"}:
            raise ValueError(f"Expected floating-point actions, got dtype={dtype!r}.")
        actions = np.asarray(payload["data"], dtype=np.float32)
        if tuple(payload["shape"]) != actions.shape:
            raise ValueError("Action data does not match the declared shape.")
        if actions.shape == (1, action_count, 10):
            actions = actions[0]
        if actions.shape != (action_count, 10) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite [{action_count}, 10] actions, got {actions.shape}.")
        return actions

    # 推理
    def predict_prepared(
        self,
        image: np.ndarray,
        *,
        prompt: str,
        frame_size: list[int],
        image_size: int = 256,
        action_stream: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Submit one already-preprocessed native input without transforming it again."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected prepared uint8 RGB pixels.")
        if list(image.shape[:2]) != list(frame_size[:2]) or len(frame_size) != 4:
            raise ValueError("Prepared image and native frame_size must agree.")
        if image_size != 256 or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Expected a native JSON prompt and image_size=256.")
        if action_stream is not None and self.action_time_schedule != "has_euler":
            raise ValueError("Constant-time baseline does not support progressive streaming")
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        height, width = image.shape[:2]
        data = {
            "prompt": prompt,
            "size": f"{width}x{height}",
            "num_frames": "17",
            "fps": "20",
            "num_inference_steps": str(self.num_inference_steps),
            "guidance_scale": str(self.guidance_scale),
            "flow_shift": str(self.flow_shift),
            "seed": str(self.seed),
            "extra_params": json.dumps(
                {
                    "action_mode": "policy",
                    "return_video": False,
                    "domain_name": "libero",
                    "raw_action_dim": 10,
                    "action_chunk_size": 16,
                    "early_stop_actions": self.early_stop_actions,
                    "policy_profile": self.policy_profile,
                    "action_time_schedule": self.action_time_schedule,
                    "faster_alpha": 0.6,
                    "faster_u0": 0.9,
                    "faster_formula": "paper_v3",
                    "native_frame_size": frame_size,
                    "image_size": image_size,
                    "view_point": "concat_view" if width > height else "third_person_view",
                }
            ),
        }
        if self.model is not None:
            data["model"] = self.model
        if action_stream is not None:  # 流式配置
            extra = json.loads(data["extra_params"])
            extra["action_stream"] = action_stream
            data["extra_params"] = json.dumps(extra)
        job_start = time.perf_counter()
        response = self.session.post(  # 向vllm-omni提交推理任务
            f"{self.base_url}/v1/videos",
            data=data,
            files={"input_reference": ("libero.png", buffer.getvalue(), "image/png")},
            timeout=min(self.timeout, 60),
        )
        response.raise_for_status()
        job_id = response.json().get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("vLLM-Omni did not return a video job id.")
        job_url = f"{self.base_url}/v1/videos/{job_id}"
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for vLLM-Omni job {job_id}.")
                response = self.session.get(job_url, timeout=min(remaining, 30))
                response.raise_for_status()
                job = response.json()
                status = job.get("status")
                if status == "completed":
                    payload = job.get("action")
                    profile = payload.get("policy_profile") if isinstance(payload, dict) else None
                    if profile is None and isinstance(payload, dict):
                        if payload.get("action_mode") == f"policy:{self.policy_profile}":
                            profile = self.policy_profile
                    if profile != self.policy_profile:
                        raise RuntimeError(
                            "vLLM backend did not confirm the native policy profile. "
                            "Use the vLLM-Omni Cosmos3 HAS pipeline and restart the server."
                        )
                    if self.early_stop_actions is not None and (
                        payload.get("early_stop_actions") != self.early_stop_actions
                        or type(payload.get("denoise_steps")) is not int
                        or not 1 <= payload["denoise_steps"] <= self.num_inference_steps
                        or payload.get("requested_denoise_steps") != self.num_inference_steps
                    ):
                        raise RuntimeError(
                            "vLLM backend did not confirm early-stop sampling metadata; restart the worker"
                        )
                    actions = self._decode_action(job.get("action"), self.early_stop_actions or 16)
                    if metrics is not None:
                        worker_seconds = job.get("inference_time_s")
                        worker_ms = None
                        if (
                            isinstance(worker_seconds, int | float)
                            and math.isfinite(worker_seconds)
                            and worker_seconds >= 0
                        ):
                            worker_ms = float(worker_seconds) * 1000
                        metrics.update(
                            worker_trace_id=(payload.get("policy_timing") or {}).get("exec.trace_id"),
                            worker_pid=(payload.get("policy_timing") or {}).get("exec.pid"),
                            execution_stage_ms={
                                key: value
                                for key, value in (payload.get("policy_timing") or {}).items()
                                if key.startswith("exec.")
                            },
                            job_id=job_id,
                            worker_inference_ms=worker_ms,
                            policy_infer_ms=model_infer_ms(
                                payload.get("policy_timing"), payload.get("timing_contract")
                            ),
                            timing_contract=payload.get("timing_contract"),
                            worker_job_latency_ms=(time.perf_counter() - job_start) * 1000,
                            stage_durations=job.get("stage_durations", {}),
                            peak_memory_mb=job.get("peak_memory_mb"),
                            early_stop_actions=payload.get("early_stop_actions"),
                            denoise_steps=payload.get("denoise_steps"),
                            requested_denoise_steps=payload.get("requested_denoise_steps"),
                        )
                    return actions
                if status in {"failed", "cancelled"}:
                    raise RuntimeError(f"vLLM-Omni job {job_id} {status}: {job.get('error')}")
                if status not in {"queued", "in_progress"}:
                    raise RuntimeError(f"Unknown vLLM-Omni job status: {status!r}.")
                time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
        finally:
            try:
                response = self.session.delete(job_url, timeout=30)
                response.raise_for_status()
            except requests.RequestException:
                logger.warning("Could not remove vLLM-Omni job %s", job_id, exc_info=True)

    def close(self) -> None:
        self.session.close()


# 新增输入预处理和最终动作反归一化
class VllmOmniActionClient(VllmOmniActionTransport):
    """Compatibility client for direct evaluation; uses the shared policy transforms."""

    def __init__(self, base_url: str, *, action_stats_path: Path = DEFAULT_LIBERO_STATS, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)
        self.normalizer = LiberoActionNormalizer(action_stats_path)

    def predict(self, image: np.ndarray, *, prompt: str, domain_name: str, image_size: int) -> list[list[float]]:
        if domain_name != "libero":
            raise ValueError("This client uses LIBERO action statistics; domain_name must be libero.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A non-empty policy prompt is required.")
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a uint8 RGB image with shape [height, width, 3].")
        pixels, caption, frame_size = prepare_libero_policy_inputs(image, prompt, image_size)
        actions = self.predict_prepared(pixels, prompt=caption, frame_size=frame_size, image_size=image_size)
        return self.normalizer.denormalize(actions)
