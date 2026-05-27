# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX-2.3 pipeline integration.

These tests verify:
- Pipeline is properly registered in the diffusion registry
- Post-process function is registered
- Cache-DiT enablers are registered
- Pipeline does NOT inherit from LTX2Pipeline
- Vocoder sample rate detection logic
- Re-export module works correctly
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_pipeline(sequence_parallel_size: int = 1):
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

    pipeline = object.__new__(LTX23Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


class TestPipelineIndependence:
    """Verify LTX23Pipeline is fully independent from LTX2Pipeline."""

    def test_ltx23_pipeline_does_not_inherit_from_ltx2(self):
        """LTX23Pipeline must NOT inherit from LTX2Pipeline."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert not issubclass(LTX23Pipeline, LTX2Pipeline), (
            "LTX23Pipeline should be fully independent and not inherit from LTX2Pipeline"
        )

    def test_ltx23_pipeline_is_nn_module(self):
        """LTX23Pipeline must be an nn.Module."""
        import torch.nn as nn

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert issubclass(LTX23Pipeline, nn.Module)

    def test_ltx23_pipeline_has_progress_bar(self):
        """LTX23Pipeline must mix in ProgressBarMixin."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

        assert issubclass(LTX23Pipeline, ProgressBarMixin)


class TestRegistryIntegration:
    """Verify all LTX-2.3 pipeline variants are registered."""

    def test_pipeline_models_registered(self):
        """LTX-2.3 pipeline variants must be in _DIFFUSION_MODELS."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_MODELS, f"{name} not found in _DIFFUSION_MODELS"

    def test_pipeline_module_paths(self):
        """Registry entries must point to the correct modules."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        # T2V -> pipeline_ltx2_3
        assert _DIFFUSION_MODELS["LTX23Pipeline"] == ("ltx2", "pipeline_ltx2_3", "LTX23Pipeline")

        # I2V -> pipeline_ltx2_3_image2video
        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3_image2video",
            "LTX23ImageToVideoPipeline",
        )

    def test_post_process_funcs_registered(self):
        """Pipeline variants must map to get_ltx2_post_process_func."""
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"
            assert _DIFFUSION_POST_PROCESS_FUNCS[name] == "get_ltx2_post_process_func"

    def test_cache_dit_enablers_registered(self):
        """Pipeline variants must be registered in CUSTOM_DIT_ENABLERS."""
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"


class TestVocoderSampleRateDetection:
    """Test _detect_vocoder_output_sample_rate logic."""

    def test_detects_48khz_from_config(self):
        """Should detect output_sampling_rate=48000 from vocoder/config.json."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000, "input_sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result == 48000

    def test_returns_none_for_no_output_sr(self):
        """Should return None if vocoder config has no output_sampling_rate."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result is None

    def test_returns_none_for_missing_directory(self):
        """Should return None if vocoder directory doesn't exist."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        result = _detect_vocoder_output_sample_rate("/nonexistent/path")
        assert result is None


class TestPostProcessFunction:
    """Test the post-process function factory."""

    def test_post_process_includes_audio_sample_rate(self):
        """Post-process func should include audio_sample_rate when detected."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import get_ltx2_post_process_func

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000}, f)

            # Create a minimal od_config mock
            class MockConfig:
                model = tmpdir

            func = get_ltx2_post_process_func(MockConfig())

            video = torch.zeros(1, 3, 4, 64, 64)
            audio = torch.zeros(1, 1, 48000)
            result = func((video, audio))

            assert isinstance(result, dict)
            assert "video" in result
            assert "audio" in result
            assert result["audio_sample_rate"] == 48000

    def test_post_process_without_vocoder_config(self):
        """Post-process func should work without vocoder config (no audio_sample_rate key)."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import get_ltx2_post_process_func

        class MockConfig:
            model = "/nonexistent/path"

        func = get_ltx2_post_process_func(MockConfig())

        video = torch.zeros(1, 3, 4, 64, 64)
        audio = torch.zeros(1, 1, 16000)
        result = func((video, audio))

        assert isinstance(result, dict)
        assert "video" in result
        assert "audio" in result
        assert "audio_sample_rate" not in result


class TestReExportModule:
    """Test that pipeline_ltx2_3_image2video.py correctly re-exports."""

    def test_i2v_classes_importable(self):
        """I2V classes must be importable from the re-export module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        assert LTX23ImageToVideoPipeline is not None

    def test_post_process_func_importable(self):
        """get_ltx2_post_process_func must be importable from re-export module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import get_ltx2_post_process_func

        assert callable(get_ltx2_post_process_func)

    def test_i2v_classes_are_same_as_direct_import(self):
        """Re-exported classes must be the same objects as direct imports."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline as Direct
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import (
            LTX23ImageToVideoPipeline as ReExported,
        )

        assert Direct is ReExported


class TestInitExports:
    """Test that __init__.py exports all LTX-2.3 classes."""

    def test_all_ltx23_classes_exported(self):
        """All LTX-2.3 pipeline classes must be in the ltx2 package __all__."""
        from vllm_omni.diffusion.models import ltx2

        expected_classes = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected_classes:
            assert hasattr(ltx2, name), f"{name} not exported from ltx2 package"
            assert name in ltx2.__all__, f"{name} not in ltx2.__all__"


class TestAudioLatentSPPadding:
    def test_prepare_audio_latents_pads_generated_dummy_length_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=2)

        latents, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=8,
            num_mel_bins=64,
            audio_latent_length=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert original_num_frames == 1
        assert padded_num_frames == 2
        assert latents.shape == (1, 2, 128)

    def test_prepare_audio_latents_pads_provided_packed_sequence_dim_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(40, dtype=torch.float32).view(1, 10, 4)

        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=8,
            audio_latent_length=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 4)
        torch.testing.assert_close(padded[:, :10], latents)
        torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 2, 4))

    def test_prepare_audio_latents_accepts_already_padded_4d_latents_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(96, dtype=torch.float32).view(1, 2, 12, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)
        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=16,
            audio_latent_length=audio_latent_length,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert audio_latent_length == 10
        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 8)
        torch.testing.assert_close(padded, pipeline._pack_audio_latents(latents))

    def test_resolve_audio_latent_length_preserves_legacy_4d_shape_inference(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 2, 13, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)

        assert audio_latent_length == 13

    def test_prepare_audio_latents_rejects_incompatible_provided_length(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 11, 4)

        with pytest.raises(ValueError, match="incompatible audio frame count"):
            pipeline.prepare_audio_latents(
                batch_size=1,
                num_channels_latents=2,
                num_mel_bins=8,
                audio_latent_length=10,
                dtype=torch.float32,
                device=torch.device("cpu"),
                latents=latents,
            )
