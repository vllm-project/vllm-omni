# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    "QwenImageLayeredPipeline",
    # Uses a custom multi-image editing contract; covered by the dedicated
    # JoyImage tiny-pipeline tests and real-checkpoint E2E accuracy test.
    "JoyImageEditPipeline",
    # Dedicated CPU/unit coverage in tests/diffusion/models/qwen_image_21/.
    # Tiny-model offline suite is not wired yet (2.1 transformer/VAE configs).
    "QwenImage21Pipeline",
    # Custom DiT/scheduler with a Qwen3.5 encoder; no tiny builder yet.
    # Covered by tests/diffusion/test_mossimage_*.py and
    # test_mossimage_registry.py.
    "MOSSImagePipeline",
    "GlmImagePipeline",
    "ZImagePipeline",
    "OvisImagePipeline",
    # Requires upstream AR hidden states and token boundaries; covered by
    # dedicated MammothModa2 bridge/pipeline tests and real-checkpoint E2E.
    "MammothModa2DiTPipeline",
    # Action expert that needs the reasoning stage's KV prefix; covered by
    # tests/diffusion/models/alpamayo2_super and the full-model E2E test.
    "Alpamayo2SuperPipeline",
    "WanPipeline",
    # Uses a full checkpoint and video input; covered by its checkpoint-gated E2E test.
    "SeedVR2Pipeline",
    "WanDMDPipeline",
    "WanVACEPipeline",
    "LTX2TwoStagePipeline",
    "LTX2DistilledOneStagePipeline",
    "LTX2DistilledTwoStagePipeline",
    "LTX2DistilledPipeline",
    "LTX2T2VDMD2Pipeline",
    "LTX2I2VDMD2Pipeline",
    # Physis uses full video model checkpoints and has dedicated contract tests.
    "PhysisH3Pipeline",
    "PhysisLTXPipeline",
    "PhysisWanPipeline",
    "MiniMaxH3Pipeline",
    "MiniMaxH3ModularPipeline",
    # Needs the full MiniMax-H3-Image checkpoint (126 GiB); covered by
    # tests/diffusion/models/minimax_h3/test_minimax_h3_image*.py.
    "MiniMaxH3ImagePipeline",
    # Decoder-only stage consumes upstream latents; covered by H3 decoder
    # handoff tests and real-checkpoint disaggregated E2E validation.
    "MiniMaxH3DecoderPipeline",
    "StableAudioPipeline",
    # Loads a single upstream model.safetensors + T5Gemma, no diffusers tiny
    # config; covered by the gated-checkpoint offline/online E2E tests.
    "StableAudio3Pipeline",
    "WanImageToVideoPipeline",
    "WanS2VPipeline",
    "WanT2VDMD2Pipeline",
    "WanI2VDMD2Pipeline",
    "LongCatVideoAvatarPipeline",
    "BagelPipeline",
    "BooguImagePipeline",
    "BooguImageTurboPipeline",
    "LancePipeline",
    "MingImagePipeline",
    "MingImageDiffusionPipeline",
    "MingImageLayeredDiffusionPipeline",
    "InternVLAA1Pipeline",
    "HunyuanImage3ForCausalMM",
    "ErnieImagePipeline",
    "NextStep11Pipeline",
    "FluxDMD2Pipeline",
    "QwenImageDMD2Pipeline",
    "OmniGen2Pipeline",
    "HeliosPipeline",
    "HeliosPyramidPipeline",
    "SenseNovaU1Pipeline",
    "HunyuanVideo15Pipeline",
    "HunyuanVideo15ImageToVideoPipeline",
    "LingBotVideoPipeline",
    # Requires camera/action assets and persistent AR state; covered by its
    # dedicated contract tests and real-checkpoint E2E smoke instead.
    "LingBotWorldCausalDMDPipeline",
    # No common tiny builder for the native Qwen/PixelUMM layout yet; covered by
    # tests/diffusion/models/hy_nomas and real-checkpoint offline/accuracy/serving E2E.
    "HyNomasPipeline",
    "HyNomasModularPipeline",
    # Requires camera/action assets and persistent AR state; covered by its
    # dedicated CPU contract tests and manual real-checkpoint smoke tests instead.
    "ABotWorldCausalPipeline",
    # Requires full MiniMax-H3 FL2VA weights, the TaoMate LoRA, and AR sessions;
    # covered by tests/diffusion/models/taomate_h3 and manual real-checkpoint E2E.
    "TaoMateH3Pipeline",
    # The 270+ GiB multimodal checkpoint needs 4/8-way native sequence/head
    # parallel execution; covered by focused native contract and GPU tests.
    "Magi2Pipeline",
    # Pro DiT is ~56 GiB bf16 and the Hub checkpoint is gated. Covered by
    # tests/diffusion/models/kandinsky6 and the real-checkpoint E2E smoke.
    "Kandinsky6TI2VAPipeline",
    "OmniVoicePipeline",
    "OmniVoice",
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3OmniPipeline",
    # Causal AR bimanual sim model requiring action/camera assets and
    # persistent AR state; covered by tests/diffusion/models/cosmos3_nano_sim_bimanual.
    "Cosmos3NanoSimBimanualPipeline",
    "DiffusersAdapterPipeline",
    "HiDreamO1ImagePipeline",
    "DreamZeroPipeline",
    # Experimental Noisy PP / cell path; covered by tests/diffusion/models/causvid_wan
    # and tests/diffusion/ar_diffusion instead of the tiny-model offline suite.
    "CausVidWanPipeline",
    "Gr00tN1d7Pipeline",
    "Pi0Pipeline",
    "Pi05Pipeline",
    "SanaWmPipeline",
    # Audio pipeline conditioned by a separate Qwen2.5-Omni encoder stage;
    # covered by tests/diffusion/models/auk (CPU pipeline tests, parity) and
    # the real-checkpoint E2E test instead of a tiny model.
    "AuKPipeline",
    # Segmentation, not generation: the tiny-model suite checks generated
    # images/videos. Covered by tests/diffusion/models/sam3 and the
    # /v1/images/segmentations endpoint tests.
    "Sam3Model",
    "Sam3VideoModel",
    # Depth estimation, not generation. Covered by
    # tests/diffusion/models/video_depth_anything and the /v1/videos/depth
    # endpoint tests.
    "VideoDepthAnythingPipeline",
    # Covered by the dedicated e2e harnesses
    # tests/e2e/offline_inference/test_flux3.py and
    # tests/e2e/online_serving/test_flux3.py (FLUX3_MODEL_PATH) instead of the
    # common tiny-model suite.
    "Flux3Pipeline",
    # Modular Diffusers checkpoint; covered by tests/e2e/offline_inference/test_pan2.py
    # with a tiny modular checkpoint. Real-checkpoint parity is checked after merge.
    "PAN2ModularPipeline",
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
