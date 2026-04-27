# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline smoke tests for MiniCPM-o 4.5 text and multimodal generation.

This covers the current non-async pipeline in ``minicpmo.yaml`` using the
same MiniCPM-specific TTS prompt shape as the existing Modal smoke script.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import hashlib
import json
import random
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pytest
import soundfile as sf

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.utils import hardware_test

MODEL = "openbmb/MiniCPM-o-4_5"
ARTIFACT_DIR_ENV = "MINICPMO45_E2E_OUTPUT_DIR"
REF_AUDIO_PATH_ENV = "MINICPMO45_REF_AUDIO_PATH"
SYNTHETIC_MEDIA_SEED_ENV = "MINICPMO45_SYNTHETIC_MEDIA_SEED"
ASYNC_LONG_AUDIO_MIN_SECONDS = 10.0
AUDIO_OUTPUT_SYSTEM_PROMPT = (
    "When audio output is requested, reply with speech only and follow any requested length constraints."
)


def get_stage_config() -> str:
    return str(
        Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "minicpmo.yaml"
    )


def get_async_chunk_stage_config() -> str:
    return str(
        Path(__file__).parent.parent.parent.parent
        / "vllm_omni"
        / "model_executor"
        / "stage_configs"
        / "minicpmo_async_chunk.yaml"
    )


@lru_cache(maxsize=1)
def _resolve_model_path() -> str:
    if os.path.isdir(MODEL):
        return MODEL

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL, resume_download=True)


def _normalize_token_ids(tokenized: Any) -> list[int]:
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def _load_ref_audio_payload_from_env() -> dict[str, Any] | None:
    ref_audio_path = os.environ.get(REF_AUDIO_PATH_ENV, "").strip()
    if not ref_audio_path:
        return None

    path = Path(ref_audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MiniCPM ref audio file not found: {path}")

    wav, sr = librosa.load(path, sr=None, mono=True)
    wav_np = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav_np.size == 0:
        raise ValueError(f"MiniCPM ref audio file is empty: {path}")

    return {
        "wav": wav_np.tolist(),
        "sr": int(sr),
    }


def _build_tts_prompt(
    model_path: str,
    text: str,
    *,
    ref_audio_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {
            "role": "system",
            "content": AUDIO_OUTPUT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            use_tts_template=True,
            enable_thinking=False,
        )
    except TypeError as e:
        raise RuntimeError(
            "MiniCPM tokenizer.apply_chat_template does not accept the TTS template kwargs needed for speech mode."
        ) from e

    prompt = {
        "prompt_token_ids": _normalize_token_ids(tokenized),
        "modalities": ["audio"],
    }
    if ref_audio_payload is not None:
        prompt["additional_information"] = {"ref_audio": ref_audio_payload}
    return prompt


def _build_multimodal_text_prompt(
    text: str,
    *,
    image: np.ndarray | None = None,
    audio: tuple[np.ndarray, int] | None = None,
    video: np.ndarray | None = None,
    modalities: list[str] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if system_prompt is None:
        system_prompt = (
            "You are MiniCPM, a helpful multimodal assistant that can understand text, image, audio, and video inputs."
        )

    user_content = ""
    multi_modal_data: dict[str, Any] = {}

    if audio is not None:
        user_content += "(<audio>./</audio>)"
        multi_modal_data["audio"] = audio

    if image is not None:
        user_content += "(<image>./</image>)"
        multi_modal_data["image"] = image

    if video is not None:
        user_content += "(<video>./</video>)"
        multi_modal_data["video"] = video

    user_content += text

    assistant_prefix = "<|im_start|>assistant\n"
    if modalities and "audio" in modalities:
        assistant_prefix += "<think>\n\n</think>\n\n<|tts_bos|>"

    prompt = {
        "prompt": (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"{assistant_prefix}"
        ),
        "modalities": modalities or ["text"],
    }
    if multi_modal_data:
        prompt["multi_modal_data"] = multi_modal_data
    return prompt


def _extract_text_and_audio(outputs: list[Any]) -> tuple[str | None, Any, int]:
    import torch

    text_output: str | None = None
    audio_tensor = None
    sample_rate = 24000

    for stage_output in outputs:
        final_output_type = getattr(stage_output, "final_output_type", None)
        request_output = getattr(stage_output, "request_output", None)
        if request_output is None:
            continue

        if final_output_type == "text" and getattr(request_output, "outputs", None):
            text_output = request_output.outputs[0].text
            continue

        if final_output_type != "audio":
            continue

        multimodal_output = getattr(request_output, "multimodal_output", None)
        if not multimodal_output and getattr(request_output, "outputs", None):
            multimodal_output = getattr(request_output.outputs[0], "multimodal_output", None)
        if not isinstance(multimodal_output, dict):
            continue

        audio_obj = multimodal_output.get("audio")
        sr_obj = multimodal_output.get("sr", sample_rate)

        if isinstance(sr_obj, list) and sr_obj:
            sr_obj = sr_obj[-1]
        if hasattr(sr_obj, "item"):
            sr_obj = sr_obj.item()
        sample_rate = int(sr_obj)

        if isinstance(audio_obj, list):
            tensor_parts = [part for part in audio_obj if isinstance(part, torch.Tensor)]
            if tensor_parts:
                audio_tensor = torch.cat(tensor_parts, dim=-1)
            elif audio_obj and isinstance(audio_obj[0], np.ndarray):
                audio_tensor = torch.from_numpy(np.concatenate(audio_obj, axis=-1))
        elif isinstance(audio_obj, torch.Tensor):
            audio_tensor = audio_obj
        elif isinstance(audio_obj, np.ndarray):
            audio_tensor = torch.from_numpy(audio_obj)

    if audio_tensor is None:
        raise RuntimeError("MiniCPM offline test returned no final audio output.")

    return text_output, audio_tensor, sample_rate


def _extract_text_output(outputs: list[Any]) -> str:
    for stage_output in outputs:
        if getattr(stage_output, "final_output_type", None) != "text":
            continue

        request_output = getattr(stage_output, "request_output", None)
        if request_output is None or not getattr(request_output, "outputs", None):
            continue

        text_output = request_output.outputs[0].text
        if text_output is not None:
            return str(text_output)

    raise RuntimeError("MiniCPM offline multimodal test returned no final text output.")


def _assert_valid_audio_output(outputs: list[Any], *, label: str) -> tuple[str | None, np.ndarray, int, float]:
    assert outputs, f"MiniCPM offline test returned no stage outputs for {label}."

    text_output, audio_tensor, sample_rate = _extract_text_and_audio(outputs)
    audio_np = audio_tensor.detach().cpu().float().numpy().reshape(-1)
    rms = float(np.sqrt(np.mean(audio_np**2)))

    assert sample_rate == 24000, f"Expected MiniCPM sample rate 24000 Hz for {label}, got {sample_rate}"
    assert audio_np.size > 0, f"MiniCPM audio output is empty for {label}"
    assert audio_np.size >= sample_rate // 20, f"MiniCPM audio output is unexpectedly short for {label}"
    assert np.isfinite(rms), f"MiniCPM audio RMS is not finite for {label}"
    assert rms > 1e-3, f"MiniCPM audio RMS too low for {label} ({rms:.6f}), likely silence"

    if text_output is not None:
        assert text_output.strip(), f"MiniCPM text output is empty for {label}"
        assert "<|tts_bos|>" not in text_output
        assert "<|tts_eos|>" not in text_output

    return text_output, audio_np, sample_rate, rms


def _assert_audio_duration_at_least(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    minimum_seconds: float,
    label: str,
) -> float:
    duration_s = float(audio_np.size / sample_rate)
    assert duration_s > minimum_seconds, (
        f"MiniCPM audio output is shorter than {minimum_seconds:.1f}s for {label} ({duration_s:.2f}s)"
    )
    return duration_s


def _maybe_seed_synthetic_media(*, label: str) -> int | None:
    seed_text = os.environ.get(SYNTHETIC_MEDIA_SEED_ENV, "").strip()
    if not seed_text:
        return None

    seed = int(seed_text)
    random.seed(seed)
    np.random.seed(seed)
    print(f"MiniCPM offline synthetic media seed ({label}): {seed}")
    return seed


def _array_sha256(array: np.ndarray) -> str:
    array_np = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array_np.dtype).encode("utf-8"))
    digest.update(str(tuple(array_np.shape)).encode("utf-8"))
    digest.update(array_np.tobytes())
    return digest.hexdigest()


def _maybe_save_json_artifact(*, label: str, payload: dict[str, Any]) -> Path | None:
    output_dir = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if not output_dir:
        return None

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"minicpmo45_{label}.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def _maybe_save_image_artifact(*, label: str, image_np: np.ndarray) -> Path | None:
    output_dir = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if not output_dir:
        return None

    from PIL import Image

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"minicpmo45_{label}.png"
    Image.fromarray(np.asarray(image_np, dtype=np.uint8)).save(artifact_path)
    return artifact_path


def _assert_valid_text_output(outputs: list[Any], *, label: str) -> str:
    assert outputs, f"MiniCPM offline test returned no stage outputs for {label}."

    text_output = _extract_text_output(outputs).strip()
    assert text_output, f"MiniCPM text output is empty for {label}"
    assert "(<image>./</image>)" not in text_output
    assert "(<video>./</video>)" not in text_output
    assert "(<audio>./</audio>)" not in text_output
    assert "<|tts_bos|>" not in text_output
    assert "<|tts_eos|>" not in text_output
    return text_output


def _maybe_save_audio_artifact(
    *,
    label: str,
    audio_np: np.ndarray,
    sample_rate: int,
) -> Path | None:
    output_dir = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if not output_dir:
        return None

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"minicpmo45_{label}.wav"
    sf.write(str(artifact_path), np.asarray(audio_np, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return artifact_path


def _maybe_save_text_artifact(*, label: str, text: str) -> Path | None:
    output_dir = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if not output_dir:
        return None

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"minicpmo45_{label}.txt"
    artifact_path.write_text(text, encoding="utf-8")
    return artifact_path


def _best_effort_close_omni(omni: Any, timeout_s: float = 15.0) -> None:
    close_error: list[BaseException] = []

    def _run_close() -> None:
        try:
            omni.close()
        except BaseException as e:
            close_error.append(e)

    close_thread = threading.Thread(
        target=_run_close,
        name="minicpmo45-offline-test-close",
        daemon=True,
    )
    close_thread.start()
    close_thread.join(timeout_s)

    if close_thread.is_alive():
        finalizer = getattr(omni, "_weak_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            finalizer.detach()
        print(
            f"MiniCPM offline test: omni.close() exceeded {timeout_s:.1f}s; "
            "returning without waiting for full teardown."
        )
        return

    if close_error:
        raise close_error[0]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_text_to_audio_with_and_without_ref() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        prompt_no_ref = _build_tts_prompt(
            model_path,
            "Please read this sentence aloud: vLLM Omni is testing MiniCPM text to audio generation.",
        )
        outputs_no_ref = list(
            omni.generate(
                prompt_no_ref,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        _, audio_np_no_ref, sample_rate_no_ref, rms_no_ref = _assert_valid_audio_output(
            outputs_no_ref,
            label="no_ref",
        )

        print(
            f"MiniCPM offline audio (no_ref): duration={audio_np_no_ref.size / sample_rate_no_ref:.2f}s "
            f"sample_rate={sample_rate_no_ref} rms={rms_no_ref:.4f}"
        )
        artifact_path_no_ref = _maybe_save_audio_artifact(
            label="no_ref",
            audio_np=audio_np_no_ref,
            sample_rate=sample_rate_no_ref,
        )
        if artifact_path_no_ref is not None:
            print(f"MiniCPM offline audio artifact (no_ref): {artifact_path_no_ref}")

        ref_audio_payload = _load_ref_audio_payload_from_env()
        if ref_audio_payload is not None:
            prompt_with_ref = _build_tts_prompt(
                model_path,
                "Please read this sentence aloud using the provided voice style: "
                "vLLM Omni is testing MiniCPM text to audio generation with reference audio.",
                ref_audio_payload=ref_audio_payload,
            )
            outputs_with_ref = list(
                omni.generate(
                    prompt_with_ref,
                    sampling_params_list=omni.default_sampling_params_list,
                    use_tqdm=False,
                )
            )

            _, audio_np_with_ref, sample_rate_with_ref, rms_with_ref = _assert_valid_audio_output(
                outputs_with_ref,
                label="with_ref",
            )

            print(
                f"MiniCPM offline audio (with_ref): duration={audio_np_with_ref.size / sample_rate_with_ref:.2f}s "
                f"sample_rate={sample_rate_with_ref} rms={rms_with_ref:.4f}"
            )
            artifact_path_with_ref = _maybe_save_audio_artifact(
                label="with_ref",
                audio_np=audio_np_with_ref,
                sample_rate=sample_rate_with_ref,
            )
            if artifact_path_with_ref is not None:
                print(f"MiniCPM offline audio artifact (with_ref): {artifact_path_with_ref}")
        else:
            print("MiniCPM offline audio (with_ref): skipped because no ref audio path was provided.")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_async_chunk_text_to_audio() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_async_chunk_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        prompt = _build_tts_prompt(
            model_path,
            "Please read this single long sentence aloud exactly once without shortening it: "
            "vLLM Omni is running an async chunk MiniCPM speech test, and this sentence intentionally "
            "includes enough detail about streaming text to audio generation, multimodal reasoning, "
            "stage connectors, careful debugging, and stable speech synthesis behavior to last well "
            "over ten seconds when spoken at a natural pace.",
        )
        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output, audio_np, sample_rate, rms = _assert_valid_audio_output(
            outputs,
            label="async_chunk_text_to_audio",
        )
        duration_s = _assert_audio_duration_at_least(
            audio_np,
            sample_rate,
            minimum_seconds=ASYNC_LONG_AUDIO_MIN_SECONDS,
            label="async_chunk_text_to_audio",
        )
        assert text_output is not None

        print(f"MiniCPM offline async_chunk audio: duration={duration_s:.2f}s sample_rate={sample_rate} rms={rms:.4f}")
        print(f"MiniCPM offline async_chunk thinker text: {text_output}")

        artifact_path = _maybe_save_audio_artifact(
            label="async_chunk_text_to_audio",
            audio_np=audio_np,
            sample_rate=sample_rate,
        )
        if artifact_path is not None:
            print(f"MiniCPM offline async_chunk audio artifact: {artifact_path}")

        text_artifact_path = _maybe_save_text_artifact(
            label="async_chunk_text_to_audio",
            text=text_output,
        )
        if text_artifact_path is not None:
            print(f"MiniCPM offline async_chunk text artifact: {text_artifact_path}")

        artifact_root = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
        if artifact_root:
            debug_root = Path(artifact_root) / "debug" / "minicpmo4_5_async_chunk"
            print(f"MiniCPM offline async_chunk debug artifact root: {debug_root}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_async_chunk_text_image_to_audio() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_async_chunk_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        media_seed = _maybe_seed_synthetic_media(label="async_chunk_text_image_to_audio")
        image = generate_synthetic_image(224, 224)["np_array"]
        image_sha256 = _array_sha256(image)
        print(f"MiniCPM offline async_chunk input image sha256: {image_sha256}")
        metadata_path = _maybe_save_json_artifact(
            label="async_chunk_text_image_to_audio_input",
            payload={
                "media_seed": media_seed,
                "image_sha256": image_sha256,
                "image_shape": list(image.shape),
            },
        )
        if metadata_path is not None:
            print(f"MiniCPM offline async_chunk image input metadata artifact: {metadata_path}")
        image_artifact_path = _maybe_save_image_artifact(
            label="async_chunk_text_image_to_audio_input",
            image_np=image,
        )
        if image_artifact_path is not None:
            print(f"MiniCPM offline async_chunk image input artifact: {image_artifact_path}")
        prompt = _build_multimodal_text_prompt(
            "Describe the image in one single detailed spoken sentence of at least sixty words, "
            "mentioning every visible shape, its color, its approximate size, its position "
            "relative to the other shapes, the plain background, and the overall layout, and keep "
            "the answer natural but long enough to last more than ten seconds.",
            image=image,
            modalities=["audio"],
            system_prompt=AUDIO_OUTPUT_SYSTEM_PROMPT,
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output, audio_np, sample_rate, rms = _assert_valid_audio_output(
            outputs,
            label="async_chunk_text_image_to_audio",
        )
        duration_s = _assert_audio_duration_at_least(
            audio_np,
            sample_rate,
            minimum_seconds=ASYNC_LONG_AUDIO_MIN_SECONDS,
            label="async_chunk_text_image_to_audio",
        )
        print(
            f"MiniCPM offline async_chunk image-audio output: duration={duration_s:.2f}s "
            f"sample_rate={sample_rate} rms={rms:.4f} text={text_output!r}"
        )

        artifact_path = _maybe_save_audio_artifact(
            label="async_chunk_text_image_to_audio",
            audio_np=audio_np,
            sample_rate=sample_rate,
        )
        if artifact_path is not None:
            print(f"MiniCPM offline async_chunk image-audio artifact: {artifact_path}")

        if text_output is not None:
            text_artifact_path = _maybe_save_text_artifact(
                label="async_chunk_text_image_to_audio",
                text=text_output,
            )
            if text_artifact_path is not None:
                print(f"MiniCPM offline async_chunk image-audio text artifact: {text_artifact_path}")

        artifact_root = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
        if artifact_root:
            debug_root = Path(artifact_root) / "debug" / "minicpmo4_5_async_chunk"
            print(f"MiniCPM offline async_chunk image-audio debug artifact root: {debug_root}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_async_chunk_text_video_to_audio() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_async_chunk_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        media_seed = _maybe_seed_synthetic_media(label="async_chunk_text_video_to_audio")
        video = generate_synthetic_video(64, 64, 30)["np_array"]
        video_sha256 = _array_sha256(video)
        first_frame_sha256 = _array_sha256(video[0])
        print(f"MiniCPM offline async_chunk input video sha256: {video_sha256} first_frame_sha256={first_frame_sha256}")
        metadata_path = _maybe_save_json_artifact(
            label="async_chunk_text_video_to_audio_input",
            payload={
                "media_seed": media_seed,
                "video_sha256": video_sha256,
                "video_shape": list(video.shape),
                "first_frame_sha256": first_frame_sha256,
            },
        )
        if metadata_path is not None:
            print(f"MiniCPM offline async_chunk video input metadata artifact: {metadata_path}")
        first_frame_path = _maybe_save_image_artifact(
            label="async_chunk_text_video_to_audio_input_first_frame",
            image_np=video[0],
        )
        if first_frame_path is not None:
            print(f"MiniCPM offline async_chunk video first-frame artifact: {first_frame_path}")
        prompt = _build_multimodal_text_prompt(
            "Describe the video in one single detailed spoken sentence of at least sixty words, "
            "covering the moving objects, their colors, their approximate sizes, the direction and "
            "pattern of their motion over time, the dark background, and the overall scene, and "
            "keep the answer natural but long enough to last more than ten seconds.",
            video=video,
            modalities=["audio"],
            system_prompt=AUDIO_OUTPUT_SYSTEM_PROMPT,
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output, audio_np, sample_rate, rms = _assert_valid_audio_output(
            outputs,
            label="async_chunk_text_video_to_audio",
        )
        duration_s = _assert_audio_duration_at_least(
            audio_np,
            sample_rate,
            minimum_seconds=ASYNC_LONG_AUDIO_MIN_SECONDS,
            label="async_chunk_text_video_to_audio",
        )
        print(
            f"MiniCPM offline async_chunk video-audio output: duration={duration_s:.2f}s "
            f"sample_rate={sample_rate} rms={rms:.4f} text={text_output!r}"
        )

        artifact_path = _maybe_save_audio_artifact(
            label="async_chunk_text_video_to_audio",
            audio_np=audio_np,
            sample_rate=sample_rate,
        )
        if artifact_path is not None:
            print(f"MiniCPM offline async_chunk video-audio artifact: {artifact_path}")

        if text_output is not None:
            text_artifact_path = _maybe_save_text_artifact(
                label="async_chunk_text_video_to_audio",
                text=text_output,
            )
            if text_artifact_path is not None:
                print(f"MiniCPM offline async_chunk video-audio text artifact: {text_artifact_path}")

        artifact_root = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
        if artifact_root:
            debug_root = Path(artifact_root) / "debug" / "minicpmo4_5_async_chunk"
            print(f"MiniCPM offline async_chunk video-audio debug artifact root: {debug_root}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_async_chunk_text_image_to_text() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_async_chunk_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        image = generate_synthetic_image(224, 224)["np_array"]
        prompt = _build_multimodal_text_prompt(
            "Describe the image briefly in one short sentence.",
            image=image,
            modalities=["text"],
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output = _assert_valid_text_output(outputs, label="async_chunk_text_image_to_text")
        print(f"MiniCPM offline async_chunk image-text output: {text_output}")

        text_artifact_path = _maybe_save_text_artifact(
            label="async_chunk_text_image_to_text",
            text=text_output,
        )
        if text_artifact_path is not None:
            print(f"MiniCPM offline async_chunk image-text artifact: {text_artifact_path}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_async_chunk_text_audio_to_text() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_async_chunk_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
        if len(audio.shape) == 2:
            audio = audio.squeeze(-1)
        prompt = _build_multimodal_text_prompt(
            "What is being spoken in the audio? Answer briefly.",
            audio=(audio, 16000),
            modalities=["text"],
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output = _assert_valid_text_output(outputs, label="async_chunk_text_audio_to_text")
        print(f"MiniCPM offline async_chunk audio-text output: {text_output}")

        text_artifact_path = _maybe_save_text_artifact(
            label="async_chunk_text_audio_to_text",
            text=text_output,
        )
        if text_artifact_path is not None:
            print(f"MiniCPM offline async_chunk audio-text artifact: {text_artifact_path}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_text_image_to_text() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        image = generate_synthetic_image(224, 224)["np_array"]
        prompt = _build_multimodal_text_prompt(
            "Describe the image briefly in one short sentence.",
            image=image,
            modalities=["text"],
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output = _assert_valid_text_output(outputs, label="text_image_to_text")
        print(f"MiniCPM offline multimodal text output: {text_output}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_text_audio_to_text() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
        if len(audio.shape) == 2:
            audio = audio.squeeze(-1)
        prompt = _build_multimodal_text_prompt(
            "What is being spoken in the audio? Answer briefly.",
            audio=(audio, 16000),
            modalities=["text"],
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output = _assert_valid_text_output(outputs, label="text_audio_to_text")
        print(f"MiniCPM offline audio-text output: {text_output}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_text_video_to_text() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        video = generate_synthetic_video(64, 64, 30)["np_array"]
        prompt = _build_multimodal_text_prompt(
            "Describe the video briefly in one short sentence.",
            video=video,
            modalities=["text"],
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output = _assert_valid_text_output(outputs, label="text_video_to_text")
        print(f"MiniCPM offline video-text output: {text_output}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_text_image_to_audio() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        image = generate_synthetic_image(224, 224)["np_array"]
        prompt = _build_multimodal_text_prompt(
            "Describe the image briefly in one short spoken sentence.",
            image=image,
            modalities=["audio"],
            system_prompt=(
                "You are MiniCPM, a helpful multimodal assistant. "
                "When audio output is requested, reply with speech only."
            ),
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output, audio_np, sample_rate, rms = _assert_valid_audio_output(
            outputs,
            label="text_image_to_audio",
        )
        print(
            f"MiniCPM offline image-audio output: duration={audio_np.size / sample_rate:.2f}s "
            f"sample_rate={sample_rate} rms={rms:.4f} text={text_output!r}"
        )
        artifact_path = _maybe_save_audio_artifact(
            label="text_image_to_audio",
            audio_np=audio_np,
            sample_rate=sample_rate,
        )
        if artifact_path is not None:
            print(f"MiniCPM offline image-audio artifact: {artifact_path}")
    finally:
        _best_effort_close_omni(omni)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_minicpmo45_text_video_to_audio() -> None:
    from vllm_omni.entrypoints.omni import Omni

    model_path = _resolve_model_path()
    omni = Omni(
        model=model_path,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    try:
        video = generate_synthetic_video(64, 64, 30)["np_array"]
        prompt = _build_multimodal_text_prompt(
            "Describe the video briefly in one short spoken sentence.",
            video=video,
            modalities=["audio"],
            system_prompt=(
                "You are MiniCPM, a helpful multimodal assistant. "
                "When audio output is requested, reply with speech only."
            ),
        )

        outputs = list(
            omni.generate(
                prompt,
                sampling_params_list=omni.default_sampling_params_list,
                use_tqdm=False,
            )
        )

        text_output, audio_np, sample_rate, rms = _assert_valid_audio_output(
            outputs,
            label="text_video_to_audio",
        )
        print(
            f"MiniCPM offline video-audio output: duration={audio_np.size / sample_rate:.2f}s "
            f"sample_rate={sample_rate} rms={rms:.4f} text={text_output!r}"
        )
        artifact_path = _maybe_save_audio_artifact(
            label="text_video_to_audio",
            audio_np=audio_np,
            sample_rate=sample_rate,
        )
        if artifact_path is not None:
            print(f"MiniCPM offline video-audio artifact: {artifact_path}")
    finally:
        _best_effort_close_omni(omni)
