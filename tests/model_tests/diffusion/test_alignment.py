"""
Tests for alignment between _DIFFUSION_MODELS and DIFFUSION_TEST_SETTINGS; if
tests in this file are failing, you are probably adding a new model, and need
to account for it in DIFFUSION_TEST_SETTINGS
"""

import pytest

from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# These diffusion models currently do not have any added (tiny) model tests.
# In general, pipelines should only be in the list below if they have not been
# migrated yet or there is a compelling reason that they should not be (which for
# now may just be CI pressure etc due to the volume of models and the cost of
# initializing the server).
#
# The tests here validate that all keys in the diffusion registry are accounted for;
# if you are adding a new model and see the tests below fail, please follow the pattern
# for adding a new tiny model builder and corresponding entry in DIFFUSION_TEST_SETTINGS.
EXCLUDED_MODELS = [
    "QwenImagePipeline",
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
    "QwenImageLayeredPipeline",
    "GlmImagePipeline",
    "ZImagePipeline",
    "OvisImagePipeline",
    "WanPipeline",
    "WanVACEPipeline",
    "LTX2Pipeline",
    "LTX2ImageToVideoPipeline",
    "LTX2TwoStagesPipeline",
    "LTX2ImageToVideoTwoStagesPipeline",
    "LTX2T2VDMD2Pipeline",
    "LTX2I2VDMD2Pipeline",
    "LTX23Pipeline",
    "LTX23ImageToVideoPipeline",
    "StableAudioPipeline",
    "WanImageToVideoPipeline",
    "WanS2VPipeline",
    "WanT2VDMD2Pipeline",
    "WanI2VDMD2Pipeline",
    "LongCatImagePipeline",
    "BagelPipeline",
    "LancePipeline",
    "MingImagePipeline",
    "InternVLAA1Pipeline",
    "LongCatImageEditPipeline",
    "StableDiffusion3Pipeline",
    "FluxKontextPipeline",
    "HunyuanImage3ForCausalMM",
    "ErnieImagePipeline",
    "NextStep11Pipeline",
    "FluxPipeline",
    "FluxDMD2Pipeline",
    "Krea2Pipeline",
    "QwenImageDMD2Pipeline",
    "OmniGen2Pipeline",
    "HeliosPipeline",
    "HeliosPyramidPipeline",
    "Flux2Pipeline",
    "DreamIDOmniPipeline",
    "SenseNovaU1Pipeline",
    "AudioXPipeline",
    "HunyuanVideo15Pipeline",
    "HunyuanVideo15ImageToVideoPipeline",
    "MagiHumanPipeline",
    "OmniVoicePipeline",
    "OmniVoice",
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3OmniPipeline",
    "DiffusersAdapterPipeline",
    "HiDreamImagePipeline",
    "DreamZeroPipeline",
    "StableDiffusionXLPipeline",
    "Gr00tN1d7Pipeline",
    "SoulXSingerPipeline",
    "SoulXSingerSVCPipeline",
]


def test_all_non_excluded_pipelines_have_tests():
    """Ensure that every pipeline in the diffusion registry either has a test
    configuration in DIFFUSION_TEST_SETTINGS, or is explicitly excluded for a
    known reason."""
    non_excluded_diff_pipes = _DIFFUSION_MODELS.keys() - set(EXCLUDED_MODELS)
    missing_tests = non_excluded_diff_pipes - DIFFUSION_TEST_SETTINGS.keys()
    assert len(missing_tests) == 0, f"Pipelines missing test settings: {missing_tests}"


def test_no_excluded_models_have_test_settings():
    """Ensure that no models in the exclude list have a test configuration in
    DIFFUSION_TEST_SETTINGS (i.e., if they have test settings, we should not
    exclude them)."""
    non_excluded_diff_pipes = _DIFFUSION_MODELS.keys() - set(EXCLUDED_MODELS)
    missing_pipes = DIFFUSION_TEST_SETTINGS.keys() - set(non_excluded_diff_pipes)
    assert len(missing_pipes) == 0
