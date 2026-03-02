"""
End-to-end tests for MammothModa2 model support.

Test Coverage:
  1. Configuration parsing and dual vocabulary handling
  2. Stage input processor (AR -> DiT conversion) and edge cases
  3. AR-stage T2I end-to-end pipeline (requires GPU + model weights)
  4. Stage YAML config validation

Model weights: env var ``MAMMOTHMODA2_MODEL_PATH``
  (default: examples/offline_inference/mammothmodal2_preview/MammothModa2-Preview)
Stage configs: ``MAMMOTHMODA2_T2I_STAGE_CONFIG`` / ``MAMMOTHMODA2_SUMMARIZE_STAGE_CONFIG``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import SamplingParams

from tests.utils import hardware_test
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# ---------------------------------------------------------------------------
# Token ID constants (Qwen2.5-VL base tokenizer + MammothModa2 gen vocab)
# ---------------------------------------------------------------------------
# Qwen2.5-VL special vision token IDs (match Mammothmoda2Qwen2_5_VLTextConfig defaults)
_IMAGE_TOKEN_ID = 151655  # "<|image_pad|>"
_VIDEO_TOKEN_ID = 151656  # "<|video_pad|>"
_VISION_START_TOKEN_ID = 151652  # "<|vision_start|>"
_VISION_END_TOKEN_ID = 151653  # "<|vision_end|>"
# MammothModa2 generation vocab (from t2i_generation_config.json)
_BASE_VOCAB_SIZE = 152064  # Qwen2.5 base vocab size; also used as eol_token_id
_VISUAL_TOKEN_START_ID = 152072  # first visual generation token
_VISUAL_TOKEN_END_ID = 168456  # last visual generation token
_GEN_VOCAB_SIZE = 32800  # size of the visual generation vocabulary
# AR stage image grid: each token covers _AR_PATCH_SIZE x _AR_PATCH_SIZE pixels
_AR_PATCH_SIZE = 16
# AR sampling top-k covers the full visual generation vocabulary
_AR_TOP_K = 2048

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "offline_inference" / "mammothmodal2_preview"
_STAGE_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs"
MODEL_PATH = os.environ.get(
    "MAMMOTHMODA2_MODEL_PATH", str(Path(__file__).resolve().parents[3] / "MammothModa2-Preview")
)
T2I_STAGE_CONFIG = os.environ.get("MAMMOTHMODA2_T2I_STAGE_CONFIG", str(_STAGE_CONFIGS_DIR / "mammoth_moda2.yaml"))
SUMMARIZE_STAGE_CONFIG = os.environ.get(
    "MAMMOTHMODA2_SUMMARIZE_STAGE_CONFIG", str(_STAGE_CONFIGS_DIR / "mammoth_moda2_ar.yaml")
)


def _load_t2i_gen_config(model_dir: str) -> dict:
    cfg_path = Path(model_dir) / "t2i_generation_config.json"
    if not cfg_path.exists():
        pytest.skip(f"t2i_generation_config.json not found at {cfg_path}")
    with cfg_path.open() as f:
        return json.load(f)


def _format_t2i_prompt(user_prompt: str, ar_width: int, ar_height: int) -> str:
    return (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{ar_width}*{ar_height}<|image token|>"
    )


# ---------------------------------------------------------------------------
# 1. Configuration parsing & dual vocabulary (CPU-only)
# ---------------------------------------------------------------------------
class TestConfigParsing:
    """Tests for Mammothmoda2Config, dual vocabulary, and tokenizer registration."""

    def test_autoconfig_registration(self):
        """AutoConfig should resolve 'mammothmoda2' model_type."""
        from transformers import AutoConfig

        from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config  # noqa: F401

        cfg = AutoConfig.for_model(
            model_type="mammothmoda2",
            llm_config={"model_type": "mammothmoda2_qwen2_5_vl"},
        )
        assert cfg.model_type == "mammothmoda2"

    def test_dual_vocab_size_computation(self):
        """With extra_gen_vocab=True: vocab_size == gen_vocab_start_index + gen_vocab_size."""
        from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Qwen2_5_VLTextConfig

        tc = Mammothmoda2Qwen2_5_VLTextConfig(
            vocab_size=_BASE_VOCAB_SIZE, extra_gen_vocab=True, gen_vocab_size=_GEN_VOCAB_SIZE
        )
        assert tc.gen_vocab_start_index == _BASE_VOCAB_SIZE
        assert tc.vocab_size == _BASE_VOCAB_SIZE + _GEN_VOCAB_SIZE

    def test_proxy_properties(self):
        """Top-level config should proxy token IDs from llm_config."""
        from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

        cfg = Mammothmoda2Config(
            llm_config={
                "model_type": "mammothmoda2_qwen2_5_vl",
                "image_token_id": _IMAGE_TOKEN_ID,
                "video_token_id": _VIDEO_TOKEN_ID,
                "vision_start_token_id": _VISION_START_TOKEN_ID,
                "vision_end_token_id": _VISION_END_TOKEN_ID,
            }
        )
        assert cfg.image_token_id == _IMAGE_TOKEN_ID
        assert cfg.video_token_id == _VIDEO_TOKEN_ID

    def test_missing_llm_config_raises(self):
        """Proxy property access with llm_config=None should raise AttributeError."""
        from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

        with pytest.raises(AttributeError, match="llm_config is None"):
            _ = Mammothmoda2Config(llm_config=None).image_token_id

    def test_t2i_generation_config_json(self):
        """t2i_generation_config.json must contain required token-ID and sampling fields."""
        cfg = _load_t2i_gen_config(MODEL_PATH)
        for key in ("eol_token_id", "visual_token_start_id", "visual_token_end_id", "top_k"):
            assert key in cfg and isinstance(cfg[key], int)

    def test_model_config_visual_ids(self):
        """config.json llm_config must contain the four Qwen2.5-VL vision token IDs."""
        cfg_path = Path(MODEL_PATH) / "config.json"
        if not cfg_path.exists():
            pytest.skip(f"config.json not found at {cfg_path}")
        with cfg_path.open() as f:
            llm_cfg = json.load(f).get("llm_config", {})
        for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            assert key in llm_cfg and isinstance(llm_cfg[key], int), (
                f"Missing or non-int field '{key}' in config.json llm_config"
            )


# ---------------------------------------------------------------------------
# 2. Stage input processor: ar2dit helpers + tests (CPU-only)
# ---------------------------------------------------------------------------
def _mock_ar(prompt_ids: list[int], gen_ids: list[int], hidden_dim: int = 128) -> MagicMock:
    """Mock AR output. ar2dit strips the last gen token, so hidden_states has len(p)+len(g)-1 rows."""
    hidden = torch.randn(len(prompt_ids) + len(gen_ids) - 1, hidden_dim)
    c = MagicMock()
    c.token_ids = gen_ids
    c.multimodal_output = {"latent": hidden}
    ar = MagicMock()
    ar.prompt_token_ids = prompt_ids
    ar.outputs = [c]
    ar.request_id = "test-req"
    return ar


def _stage(ar_outputs: list) -> list:
    s = MagicMock()
    s.engine_outputs = ar_outputs
    return [s]


def _p(image_height: int = 512, image_width: int = 512, visual_ids: list[int] | None = None, **kw) -> dict:
    if visual_ids is None:
        visual_ids = [_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID, _VISION_END_TOKEN_ID]
    return {
        "additional_information": {
            "omni_task": ["t2i"],
            "ar_width": [image_width // _AR_PATCH_SIZE],
            "ar_height": [image_height // _AR_PATCH_SIZE],
            "eol_token_id": [kw.get("eol_token_id", _BASE_VOCAB_SIZE)],
            "visual_token_start_id": [kw.get("visual_token_start_id", _VISUAL_TOKEN_START_ID)],
            "visual_token_end_id": [kw.get("visual_token_end_id", _VISUAL_TOKEN_END_ID)],
            "image_height": [image_height],
            "image_width": [image_width],
            "num_inference_steps": [kw.get("num_inference_steps", 50)],
            "text_guidance_scale": [kw.get("text_guidance_scale", 9.0)],
            "cfg_range": list(kw.get("cfg_range", [0.0, 1.0])),
            "visual_ids": visual_ids,
        }
    }


class TestAR2DitProcessor:
    """Unit tests for the AR->DiT stage input processor."""

    def test_basic_output_structure(self):
        """ar2dit should produce one dict with expected keys for each input."""
        ar_out = _mock_ar(list(range(10)), list(range(_VISUAL_TOKEN_START_ID, _VISUAL_TOKEN_START_ID + 20)) + [0])
        dit_inputs = ar2dit(_stage([ar_out]), engine_input_source=[0], prompts=[_p()])
        assert len(dit_inputs) == 1
        inp = dit_inputs[0]
        assert isinstance(inp, dict) and inp["prompt_token_ids"] == [0]
        info = inp["additional_information"]
        for key in ("text_prompt_embeds", "image_prompt_embeds", "image_height", "image_width"):
            assert key in info

    def test_embed_shapes_and_dtype(self):
        """text/image condition embeds must be 2D float32 with correct leading dim."""
        n_gen = 30
        ar_out = _mock_ar(
            list(range(15)), list(range(_VISUAL_TOKEN_START_ID, _VISUAL_TOKEN_START_ID + n_gen)) + [0], hidden_dim=128
        )
        info = ar2dit(_stage([ar_out]), engine_input_source=[0], prompts=[_p()])[0]["additional_information"]
        assert info["image_prompt_embeds"].shape == (n_gen, 128)
        assert info["text_prompt_embeds"].dtype == torch.float32

    def test_missing_latent_raises(self):
        """ar2dit must raise ValueError when AR output has no 'latent' key."""
        c = MagicMock()
        c.token_ids = [_VISUAL_TOKEN_START_ID, 0]
        c.multimodal_output = {}
        ar = MagicMock()
        ar.prompt_token_ids = [1, 2, 3]
        ar.outputs = [c]
        ar.request_id = "req-no-latent"
        with pytest.raises(ValueError, match="missing latent"):
            ar2dit(_stage([ar]), engine_input_source=[0], prompts=[_p()])

    def test_hidden_states_mismatch_raises(self):
        """ar2dit must raise AssertionError on hidden-states length mismatch."""
        prompt_ids, gen_ids = list(range(10)), list(range(_VISUAL_TOKEN_START_ID, _VISUAL_TOKEN_START_ID + 5)) + [0]
        c = MagicMock()
        c.token_ids = gen_ids
        c.multimodal_output = {"latent": torch.randn(len(prompt_ids) + len(gen_ids) + 5, 64)}
        ar = MagicMock()
        ar.prompt_token_ids = prompt_ids
        ar.outputs = [c]
        ar.request_id = "req-mismatch"
        with pytest.raises(AssertionError, match="Hidden states length mismatch"):
            ar2dit(_stage([ar]), engine_input_source=[0], prompts=[_p()])

    def test_visual_ids_excluded_from_text_embeds(self):
        """Prompt tokens that are visual_ids should not appear in text_prompt_embeds."""
        visual_ids = [_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID, _VISION_END_TOKEN_ID]
        prompt_ids = [100, _IMAGE_TOKEN_ID, 200, _VISION_START_TOKEN_ID, 300]  # 3 plain + 2 visual_id
        gen_ids = [_VISUAL_TOKEN_START_ID, 0]
        hidden = torch.randn(len(prompt_ids) + len(gen_ids) - 1, 32)
        c = MagicMock()
        c.token_ids = gen_ids
        c.multimodal_output = {"latent": hidden}
        ar = MagicMock()
        ar.prompt_token_ids = prompt_ids
        ar.outputs = [c]
        ar.request_id = "req-visual"
        info = ar2dit(_stage([ar]), engine_input_source=[0], prompts=[_p(visual_ids=visual_ids)])[0][
            "additional_information"
        ]
        assert info["text_prompt_embeds"].shape[0] == 3


# ---------------------------------------------------------------------------
# 3. End-to-end T2I pipeline (requires GPU + model weights)
# ---------------------------------------------------------------------------
@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_mammothmoda2_t2i_e2e():
    """
    End-to-end text-to-image generation with MammothModa2 (AR -> DiT).

    Verifies:
      - Omni pipeline initialises with the two-stage YAML config.
      - AR stage produces latent output consumed by DiT stage.
      - DiT stage outputs an image tensor.
    """
    from vllm_omni import Omni

    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model weights not found at {MODEL_PATH}")
    if not Path(T2I_STAGE_CONFIG).exists():
        pytest.skip(f"Stage config not found at {T2I_STAGE_CONFIG}")

    gen_cfg = _load_t2i_gen_config(MODEL_PATH)
    eol_token_id = int(gen_cfg["eol_token_id"])
    visual_start = int(gen_cfg["visual_token_start_id"])
    visual_end = int(gen_cfg["visual_token_end_id"])

    height, width = 256, 256  # small for CI speed
    ar_height, ar_width = height // _AR_PATCH_SIZE, width // _AR_PATCH_SIZE
    expected_grid_tokens = ar_height * (ar_width + 1)

    prompt_text = "A cat sitting on a laptop keyboard"
    formatted_prompt = _format_t2i_prompt(prompt_text, ar_width, ar_height)

    omni = Omni(
        model=MODEL_PATH,
        stage_configs_path=T2I_STAGE_CONFIG,
        trust_remote_code=True,
    )
    try:
        ar_sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=_AR_TOP_K,
            max_tokens=max(1, expected_grid_tokens + 1),
            detokenize=False,
        )
        dit_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            detokenize=False,
        )

        additional_information = {
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "ar_height": [ar_height],
            "eol_token_id": [eol_token_id],
            "visual_token_start_id": [visual_start],
            "visual_token_end_id": [visual_end],
            "image_height": [height],
            "image_width": [width],
            "num_inference_steps": [2],  # minimal steps for CI
            "text_guidance_scale": [1.0],  # no CFG for speed
            "cfg_range": [0.0, 1.0],
            "visual_ids": [_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID, _VISION_END_TOKEN_ID],
        }

        inputs = [
            {
                "prompt": formatted_prompt,
                "additional_information": additional_information,
            }
        ]

        outputs = list(omni.generate(inputs, [ar_sampling, dit_sampling]))

        assert len(outputs) > 0, "Pipeline produced no outputs"

        # Validate that an image tensor was produced by the final (DiT) stage.
        found_image = False
        for out in outputs:
            ro_list = getattr(out, "request_output", out)
            if not isinstance(ro_list, list):
                ro_list = [ro_list]
            for ro in ro_list:
                completion_outputs = getattr(ro, "outputs", None)
                if not isinstance(completion_outputs, list):
                    continue
                for completion in completion_outputs:
                    mm = getattr(completion, "multimodal_output", None)
                    if isinstance(mm, dict) and "image" in mm:
                        img_payload = mm["image"]
                        img_list = img_payload if isinstance(img_payload, list) else [img_payload]
                        for img_tensor in img_list:
                            assert isinstance(img_tensor, torch.Tensor), (
                                f"Expected image tensor, got {type(img_tensor)}"
                            )
                            # DiT output is (C, H*2, W*2) or (1, C, H*2, W*2)
                            assert img_tensor.ndim in (3, 4), f"Expected 3D or 4D image tensor, got {img_tensor.ndim}D"
                            found_image = True

        assert found_image, "No image tensor found in pipeline output"
    finally:
        omni.close()


# ---------------------------------------------------------------------------
# 4. Stage YAML config validation (CPU-only)
# ---------------------------------------------------------------------------
class TestStageConfigValidation:
    """Validate stage YAML configs used by MammothModa2."""

    def test_t2i_config_two_stages(self):
        """T2I YAML must define exactly 2 stages (AR->latent, DiT->image) with correct wiring."""
        import yaml

        if not Path(T2I_STAGE_CONFIG).exists():
            pytest.skip(f"Not found: {T2I_STAGE_CONFIG}")
        with open(T2I_STAGE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        stages = cfg["stage_args"]
        assert len(stages) == 2
        assert stages[0]["engine_args"]["model_stage"] == "ar"
        assert stages[0]["engine_args"]["engine_output_type"] == "latent"
        assert stages[1]["engine_args"]["model_stage"] == "dit"
        assert stages[1]["engine_args"]["engine_output_type"] == "image"
        assert stages[1].get("engine_input_source") == [0]
        assert "mammoth_moda2" in stages[1].get("custom_process_input_func", "")

    def test_summarize_config_single_ar_stage(self):
        """Image-summarisation YAML must be a single AR stage outputting text."""
        import yaml

        if not Path(SUMMARIZE_STAGE_CONFIG).exists():
            pytest.skip(f"Not found: {SUMMARIZE_STAGE_CONFIG}")
        with open(SUMMARIZE_STAGE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        stages = cfg["stage_args"]
        assert len(stages) == 1
        assert stages[0]["engine_args"]["model_stage"] == "ar"
        assert stages[0]["engine_args"]["engine_output_type"] == "text"
        assert stages[0].get("final_output") is True
