# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS End-to-End Offline Inference Example.

GLM-TTS is a two-stage TTS system:
  - Stage 0 (AR): Llama-based model generates speech tokens from text
  - Stage 1 (DiT): Flow matching model converts speech tokens to audio

Usage:
    # Sync two-stage (default)
    python examples/offline_inference/text_to_speech/glm_tts/end2end.py \
        --model /path/to/GLM-TTS \
        --text "你好，这是一个语音合成测试。" \
        --ref-audio /path/to/reference.wav \
        --ref-text "参考音频的转录文本。" \
        --output-dir ./output

    # Async chunk mode (streaming DiT)
    python examples/offline_inference/text_to_speech/glm_tts/end2end.py \
        --model /path/to/GLM-TTS --async-chunk \
        --text "你好，这是一个语音合成测试。" \
        --ref-audio /path/to/reference.wav \
        --ref-text "参考音频的转录文本。" \
        --output-dir ./output
"""

import base64
import io
import logging
import os
import tempfile
import time
from typing import Any
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch
import yaml

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni
from vllm_omni.model_executor.models.glm_tts.glm_tts import build_glm_tts_prefill_metadata

logger = logging.getLogger(__name__)

DEFAULT_DEPLOY_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "vllm_omni",
    "deploy",
    "glm_tts.yaml",
)
SAMPLE_RATE = 24000


def _load_ref_audio(ref_audio: str) -> tuple[torch.Tensor, int]:
    """Load reference audio from file path, URL, or data URI."""
    if ref_audio.startswith(("http://", "https://")):
        with urlopen(ref_audio, timeout=60) as response:
            audio_obj: Any = io.BytesIO(response.read())
    elif ref_audio.startswith("data:"):
        _, _, encoded = ref_audio.partition(",")
        audio_obj = io.BytesIO(base64.b64decode(encoded))
    else:
        audio_obj = ref_audio
    wav_np, sr = sf.read(audio_obj, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    return torch.from_numpy(wav_np), int(sr)


def _concat_audio(audio_val: Any) -> np.ndarray:
    """Concatenate audio tensors from multimodal output."""
    if isinstance(audio_val, list):
        tensors = [torch.as_tensor(t).float().reshape(-1) for t in audio_val if t is not None]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).cpu().numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.float().cpu().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _extract_sample_rate(audio_mm: dict) -> int:
    """Extract sample rate from multimodal output dict."""
    sr_raw = audio_mm.get("sr", SAMPLE_RATE)
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else SAMPLE_RATE
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


def _modify_deploy_config(base_path: str, async_chunk: bool) -> str:
    """Build deploy config with explicit sync/async mode and eager execution.

    Mirrors the logic in ``tests/e2e/offline_inference/test_glm_tts.py``
    (``_get_deploy_config``) so that example runs match CI behavior.
    """
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["async_chunk"] = async_chunk
    for stage in cfg.get("stages", []):
        stage["enforce_eager"] = True
        if stage.get("stage_id") == 0:
            stage["async_scheduling"] = bool(async_chunk)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="glm_tts_")
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def main(args):
    """Run offline GLM-TTS inference."""
    os.makedirs(args.output_dir, exist_ok=True)
    base_deploy_config = args.deploy_config or DEFAULT_DEPLOY_CONFIG
    deploy_config_path = _modify_deploy_config(base_deploy_config, args.async_chunk)

    ref_audio_wav, ref_audio_sr = _load_ref_audio(args.ref_audio)
    if not args.ref_text:
        raise ValueError("GLM-TTS requires --ref-audio and --ref-text.")

    inputs = [
        {
            "prompt": args.text,
            "multi_modal_data": {
                "audio": (ref_audio_wav.float().cpu().numpy(), ref_audio_sr),
            },
            "modalities": ["audio"],
            "mm_processor_kwargs": {"prompt_text": args.ref_text},
            "additional_information": build_glm_tts_prefill_metadata(
                args.model,
                args.text,
                args.ref_text,
            ),
        }
    ]

    omni = Omni(
        model=args.model,
        stage_configs_path=deploy_config_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    outputs = omni.generate(inputs)
    elapsed = (time.perf_counter() - t_start) * 1000

    assert outputs, "No outputs returned"
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, "No audio output found"

    audio = _concat_audio(audio_mm["audio"])
    sr = _extract_sample_rate(audio_mm)
    out_path = os.path.join(args.output_dir, "output.wav")
    sf.write(out_path, audio, samplerate=sr, format="WAV")

    logger.info("Saved %s (%.2fs @ %dHz)", out_path, len(audio) / sr, sr)
    logger.info("Total inference: %.1f ms", elapsed)


def parse_args():
    parser = FlexibleArgumentParser(description="GLM-TTS Text-to-Speech Example")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--text", type=str, default="你好，这是一个语音合成测试。")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference WAV path/URL")
    parser.add_argument("--ref-text", type=str, required=True, help="Transcript of ref audio")
    parser.add_argument("--deploy-config", type=str, default=None)
    parser.add_argument(
        "--async-chunk",
        action="store_true",
        default=False,
        help="Enable async_chunk mode (streaming DiT). Default: sync two-stage.",
    )
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
