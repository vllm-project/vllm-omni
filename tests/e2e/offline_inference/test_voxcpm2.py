"""E2E test for VoxCPM2 native AR offline inference."""

import os

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner

VOXCPM2_MODEL = "openbmb/VoxCPM2"
STAGE_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "model_executor",
    "stage_configs",
    "voxcpm2.yaml",
)
SAMPLE_RATE = 48000


@pytest.fixture(scope="module")
def voxcpm2_engine():
    """Create VoxCPM2 engine for testing."""
    with OmniRunner(VOXCPM2_MODEL, stage_configs_path=STAGE_CONFIG) as runner:
        yield runner.omni


def _extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output."""
    assert isinstance(multimodal_output, dict), f"Expected dict, got {type(multimodal_output)}"

    # Output processor accumulates per-step audio chunks under "audio".
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    assert audio is not None, f"No audio key, got {list(multimodal_output.keys())}"

    if isinstance(audio, list):
        valid = [torch.as_tensor(x).float().cpu().reshape(-1) for x in audio if x is not None]
        assert valid, "No valid audio tensors in output list"
        audio = torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]

    assert isinstance(audio, torch.Tensor), f"Expected Tensor, got {type(audio)}"
    return audio


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxcpm2_zero_shot_001(voxcpm2_engine):
    """Test zero-shot TTS produces valid audio output."""
    outputs = voxcpm2_engine.generate([{"prompt": "Hello, this is a test."}])
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].outputs[0].multimodal_output)
    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxcpm2_voice_clone_002(voxcpm2_engine):
    """Test voice cloning with a reference audio file.

    Uses the example ``reference_speaker.wav`` bundled with the voxcpm
    package. Skipped if the file is not present.
    """
    # Try to locate a reference wav from the voxcpm package / env override
    candidates = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        candidates.append(os.path.join(env_path, "..", "examples", "reference_speaker.wav"))
    try:
        import voxcpm  # noqa: F401 (only used to locate path)

        vox_dir = os.path.dirname(os.path.dirname(os.path.abspath(voxcpm.__file__)))
        candidates.append(os.path.join(vox_dir, "examples", "reference_speaker.wav"))
    except ImportError:
        pass

    ref_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if ref_path is None:
        pytest.skip("No reference audio available for voice clone test")

    outputs = voxcpm2_engine.generate(
        [
            {
                "prompt": "Hello, this is a voice clone demo.",
                "additional_information": {"reference_audio": ref_path},
            }
        ]
    )
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].outputs[0].multimodal_output)
    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxcpm2_prefill_decode_mixed_batch_003(voxcpm2_engine):
    """Regression: prefill+decode mixed batch must not crash (PR #2903)."""
    long_prompt = (
        "This is a deliberately long prompt that will stay in the decode "
        "phase for many steps so that subsequent shorter prompts keep "
        "entering prefill alongside it, reproducing the prefill plus "
        "decode mixed batch scheduling pattern."
    )
    short_prompts = [
        "Hello one.",
        "Hello two.",
        "Hello three.",
        "Hello four.",
    ]
    requests = [{"prompt": long_prompt}] + [{"prompt": p} for p in short_prompts]

    outputs = voxcpm2_engine.generate(requests)
    assert len(outputs) == len(requests)

    for i, out in enumerate(outputs):
        audio = _extract_audio(out.outputs[0].multimodal_output)
        duration_s = audio.shape[0] / SAMPLE_RATE
        assert 0.1 < duration_s < 30.0, f"Request {i} audio duration out of range: {duration_s:.2f}s"
