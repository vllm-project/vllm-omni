"""E2E offline inference tests for SoulX-Singer (single-stage, preprocess inline)."""

import functools
import importlib
import json
import os
import shutil
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PROMPT_AUDIO = get_asset_path("soulxsinger/zh_prompt.mp3")
TARGET_AUDIO = get_asset_path("soulxsinger/music.mp3")
PROMPT_METADATA = get_asset_path("soulxsinger/zh_prompt.json")
TARGET_METADATA = get_asset_path("soulxsinger/music.json")
SAMPLE_RATE = 24_000

# phone_set.json (phoneme vocab) is not shipped on HuggingFace; SVS loads it
# from the model dir. Stage it from this pinned upstream commit so the SVS DiT
# can run in CI; the SVS tests skip gracefully if it cannot be fetched.
_PHONE_SET_URL = (
    "https://raw.githubusercontent.com/Soul-AILab/SoulX-Singer/"
    "81aeb3ae772c70093c3de74dc23c92d983801ae4/soulxsinger/utils/phoneme/phone_set.json"
)

if not PROMPT_AUDIO.is_file() or not TARGET_AUDIO.is_file():
    pytest.skip(
        f"Missing SoulX-Singer audio assets: {PROMPT_AUDIO.name}, {TARGET_AUDIO.name}",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.tts]

_CASES = (
    pytest.param(
        "SoulXSingerPipeline",
        "soulxsinger_svs.yaml",
        {
            "language": "Mandarin",
            "vocal_sep": False,
            "control": "score",
            "auto_shift": False,
            "pitch_shift": 0,
        },
        ("g2pM", "g2p_en"),
        id="svs",
    ),
    pytest.param(
        "SoulXSingerSVCPipeline",
        "soulxsinger_svc.yaml",
        {"vocal_sep": False, "auto_shift": False, "pitch_shift": 0},
        (),
        id="svc",
    ),
)


@functools.lru_cache(maxsize=1)
def _resolve_weights() -> tuple[Path, Path, Path]:
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download("Soul-AILab/SoulX-Singer", allow_patterns=["*"]))

    # phone_set.json is not on HF; best-effort stage it from pinned upstream so
    # SVS can load. SVS tests skip (not fail) if both this and a manual copy are absent.
    phone_set = base / "phoneme" / "phone_set.json"
    if not phone_set.is_file() and not (base / "phone_set.json").is_file():
        try:
            from urllib.request import urlretrieve

            phone_set.parent.mkdir(exist_ok=True)
            urlretrieve(_PHONE_SET_URL, phone_set)
        except Exception:
            pass

    # make a temporary svc directory
    svc_dir = base.parent / "SoulX-Singer-SVC"
    svc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "config.yaml", svc_dir / "config.yaml")
    shutil.move(base / "model-svc.pt", svc_dir / "model-svc.pt")

    if raw := os.environ.get("SOULX_PREPROCESS_WEIGHTS_DIR"):
        pre = Path(raw).expanduser().resolve()
        if (pre / "rmvpe" / "rmvpe.pt").is_file():
            return base, svc_dir, pre

    from huggingface_hub import snapshot_download

    pre = Path(snapshot_download("Soul-AILab/SoulX-Singer-Preprocess", allow_patterns=["*"]))
    return base, svc_dir, pre


@pytest.fixture(scope="session")
def soulx_weights() -> tuple[Path, Path, Path]:
    try:
        return _resolve_weights()
    except Exception as exc:
        pytest.skip(f"Set SOULXSINGER_MODEL_DIR / SOULX_PREPROCESS_WEIGHTS_DIR. ({exc})")


def _flatten_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        chunks = [c.detach().cpu().float().numpy().reshape(-1) for c in audio_val if c is not None]
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("architecture,deploy_yaml,extra_args,py_deps", _CASES)
def test_soulxsinger_multistage_from_audio(
    soulx_weights: tuple[Path, Path, Path],
    architecture: str,
    deploy_yaml: str,
    extra_args: dict,
    py_deps: tuple[str, ...],
) -> None:
    for mod in py_deps:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            pytest.fail(f"SoulX SVS requires {mod}: {exc}")

    base_dir, svc_dir, preprocess_dir = soulx_weights

    # SVS mode requires phone_set.json in the model directory
    if architecture == "SoulXSingerPipeline":
        json.dump(
            {"model_type": "soulxsinger", "architectures": ["SoulXSingerPipeline"], "max_num_seqs": 1},
            open(base_dir / "config.json", "w"),
            indent=4,
        )
        model = str(base_dir)
        if not (base_dir / "phoneme" / "phone_set.json").is_file() and not (base_dir / "phone_set.json").is_file():
            pytest.skip(
                "SoulX-Singer SVS test requires phoneme/phone_set.json. "
                "Copy it from github.com/Soul-AILab/SoulX-Singer into the model dir. "
                "See `examples/offline_inference/text_to_speech/README.md` for details."
            )
    if architecture == "SoulXSingerSVCPipeline":
        json.dump(
            {"model_type": "soulxsinger", "architectures": ["SoulXSingerSVCPipeline"], "max_num_seqs": 1},
            open(svc_dir / "config.json", "w"),
            indent=4,
        )
        model = str(svc_dir)

    with OmniRunner(
        model,
        stage_configs_path=get_deploy_config_path(deploy_yaml),
        async_chunk=False,
    ) as runner:
        sampling = OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=3.0,
            seed=42,
            extra_args={
                "prompt_audio": str(PROMPT_AUDIO),
                "target_audio": str(TARGET_AUDIO),
                "preprocess_weights_dir": str(preprocess_dir),
                **extra_args,
            },
        )
        prompt = {"prompt_token_ids": [0]}
        outputs = runner.generate([prompt], sampling)

    assert outputs and outputs[0].error is None, outputs[0].error if outputs else "no output"
    mm = outputs[0].multimodal_output
    assert isinstance(mm, dict) and "audio" in mm
    audio = _flatten_audio(mm["audio"])
    assert 12_000 <= audio.size
    assert np.isfinite(audio).all() and float(np.max(np.abs(audio))) > 1e-4
    duration_s = audio.size / SAMPLE_RATE
    assert 50.0 <= duration_s <= 52.0, f"duration={duration_s:.1f}s"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_soulxsinger_svs_precomputed(soulx_weights: tuple[Path, Path, Path]) -> None:
    """SVS from precomputed note/lyric metadata.

    Exercises the SVS DiT + vocoder serving path while skipping the heavy
    upstream preprocess (funasr ASR + ROSVOT note transcription), so it runs in
    the merge gate without those non-pip dependencies.
    """
    base_dir, _svc_dir, preprocess_dir = soulx_weights

    if not PROMPT_METADATA.is_file() or not TARGET_METADATA.is_file():
        pytest.skip(f"Missing SoulX-Singer metadata fixtures: {PROMPT_METADATA.name}, {TARGET_METADATA.name}")
    if not (base_dir / "phoneme" / "phone_set.json").is_file() and not (base_dir / "phone_set.json").is_file():
        pytest.skip(
            "SoulX-Singer SVS requires phoneme/phone_set.json (upstream fetch unavailable). "
            "See `examples/offline_inference/text_to_speech/README.md`."
        )

    json.dump(
        {"model_type": "soulxsinger", "architectures": ["SoulXSingerPipeline"], "max_num_seqs": 1},
        open(base_dir / "config.json", "w"),
        indent=4,
    )

    with OmniRunner(
        str(base_dir),
        stage_configs_path=get_deploy_config_path("soulxsinger_svs.yaml"),
        async_chunk=False,
    ) as runner:
        sampling = OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=3.0,
            seed=42,
            extra_args={
                "language": "Mandarin",
                "control": "score",
                "prompt_metadata_path": str(PROMPT_METADATA),
                "target_metadata_path": str(TARGET_METADATA),
                "audio_path": str(PROMPT_AUDIO),
                "preprocess_weights_dir": str(preprocess_dir),
            },
        )
        prompt = {"prompt_token_ids": [0]}
        outputs = runner.generate([prompt], sampling)

    assert outputs and outputs[0].error is None, outputs[0].error if outputs else "no output"
    mm = outputs[0].multimodal_output
    assert isinstance(mm, dict) and "audio" in mm
    audio = _flatten_audio(mm["audio"])
    assert 12_000 <= audio.size
    assert np.isfinite(audio).all() and float(np.max(np.abs(audio))) > 1e-4
    duration_s = audio.size / SAMPLE_RATE
    assert 50.0 <= duration_s <= 52.0, f"duration={duration_s:.1f}s"
