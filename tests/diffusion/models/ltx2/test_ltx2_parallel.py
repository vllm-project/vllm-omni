# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX guidance and forward-parallel behavior."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestCFGParallelHelpers:
    """Test LTX-2.3 CFG helper math without loading model weights."""

    def test_combine_cfg_noise_matches_x0_space_formula(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX23Pipeline

        pipe = object.__new__(LTX23Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        video_combined, audio_combined = pipe.combine_cfg_noise(
            (video_pos, audio_pos),
            (video_neg, audio_neg),
            scale,
            video_latents=video_sample,
            audio_latents=audio_sample,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        x0_video_cond = video_sample - video_pos * video_sigma
        x0_video_uncond = video_sample - video_neg * video_sigma
        x0_video_guided = x0_video_cond + (scale - 1) * (x0_video_cond - x0_video_uncond)
        expected_video = (video_sample - x0_video_guided) / video_sigma

        x0_audio_cond = audio_sample - audio_pos * audio_sigma
        x0_audio_uncond = audio_sample - audio_neg * audio_sigma
        x0_audio_guided = x0_audio_cond + (scale - 1) * (x0_audio_cond - x0_audio_uncond)
        expected_audio = (audio_sample - x0_audio_guided) / audio_sigma
        assert torch.allclose(video_combined, expected_video)
        assert torch.allclose(audio_combined, expected_audio)

    def test_two_rank_cfg_parallel_smoke_uses_rank_local_branch_and_x0_formula(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2 import ltx2_guidance
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2 as ltx23

        pipe = object.__new__(ltx23.LTX23Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0, -1.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2, 0.3]]])
        audio_neg = torch.tensor([[[0.1, 0.4, -0.5]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if tensor.shape == video_pos.shape:
                    return [video_pos, video_neg]
                return [audio_pos, audio_neg]

        monkeypatch.setattr(ltx2_guidance, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx2_guidance, "get_cfg_group", lambda: FakeCfgGroup())

        expected_video = ltx2_guidance.combine_velocity_via_x0(
            video_sample,
            video_pos,
            video_neg,
            video_sigma,
            scale,
        )
        expected_audio = ltx2_guidance.combine_velocity_via_x0(
            audio_sample,
            audio_pos,
            audio_neg,
            audio_sigma,
            scale,
        )

        for rank, expected_branch in ((0, "positive"), (1, "negative")):
            calls = []
            monkeypatch.setattr(ltx2_guidance, "get_classifier_free_guidance_rank", lambda rank=rank: rank)

            def fake_predict_noise(**kwargs):
                calls.append(kwargs["branch"])
                if kwargs["branch"] == "positive":
                    return video_pos, audio_pos
                return video_neg, audio_neg

            object.__setattr__(pipe, "predict_noise", fake_predict_noise)
            video_combined, audio_combined = pipe.predict_noise_with_parallel_cfg(
                true_cfg_scale=scale,
                positive_kwargs={"branch": "positive"},
                negative_kwargs={"branch": "negative"},
                cfg_normalize=False,
                video_latents=video_sample,
                audio_latents=audio_sample,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )

            assert calls == [expected_branch]
            torch.testing.assert_close(video_combined, expected_video)
            torch.testing.assert_close(audio_combined, expected_audio)

        assert "_cfg_video_latents" not in pipe.__dict__
        assert "_cfg_audio_latents" not in pipe.__dict__


class TestCFGParallelForwardPath:
    """Test the LTX-2.3 CFG-parallel denoising path without loading model weights."""

    @pytest.mark.parametrize("pipeline_cls_name", ["LTX2Pipeline", "LTX23Pipeline"])
    def test_forward_collates_request_prompt_embeds_and_mask_aliases(self, monkeypatch, pipeline_cls_name):
        from vllm_omni.diffusion.models.ltx2 import ltx2_pipeline_runtime
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2 as ltx23
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipeline_cls = getattr(ltx23, pipeline_cls_name)
        pipe = object.__new__(pipeline_cls)
        torch.nn.Module.__init__(pipe)
        pipe.device = torch.device("cpu")
        pipe.tokenizer_max_length = 4
        monkeypatch.setattr(ltx2_pipeline_runtime, "get_classifier_free_guidance_world_size", lambda: 1)

        class StopAtEncodePromptError(Exception):
            pass

        captured = {}

        def fake_encode_prompt(**kwargs):
            captured.update(kwargs)
            raise StopAtEncodePromptError

        object.__setattr__(pipe, "encode_prompt", fake_encode_prompt)

        prompt_embeds_a = torch.zeros(2, 3)
        prompt_embeds_b = torch.ones(2, 3)
        negative_prompt_embeds_a = torch.full((2, 3), 2.0)
        negative_prompt_embeds_b = torch.full((2, 3), 3.0)
        prompt_attention_mask_a = torch.tensor([True, True])
        prompt_attention_mask_b = torch.tensor([True, False])
        negative_attention_mask_a = torch.tensor([False, True])
        negative_attention_mask_b = torch.tensor([False, False])

        requests = [
            OmniDiffusionRequest(
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "prompt_attention_mask": prompt_attention_mask_a,
                    "negative_prompt_attention_mask": negative_attention_mask_a,
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=32,
                    width=32,
                    num_frames=1,
                    frame_rate=1.0,
                    num_inference_steps=2,
                ),
                request_id="ltx23-prompt-local-a",
            ),
            OmniDiffusionRequest(
                prompt={
                    "prompt": "prompt-b",
                    "negative_prompt": "negative-b",
                    "prompt_embeds": prompt_embeds_b,
                    "negative_prompt_embeds": negative_prompt_embeds_b,
                    "attention_mask": prompt_attention_mask_b,
                    "negative_attention_mask": negative_attention_mask_b,
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=32,
                    width=32,
                    num_frames=1,
                    frame_rate=1.0,
                    num_inference_steps=2,
                ),
                request_id="ltx23-prompt-local-b",
            ),
        ]

        with pytest.raises(StopAtEncodePromptError):
            pipe.forward(DiffusionRequestBatch(requests=requests))

        assert captured["prompt"] is None
        assert captured["negative_prompt"] is None
        torch.testing.assert_close(
            captured["prompt_embeds"],
            torch.stack([prompt_embeds_a, prompt_embeds_b], dim=0),
        )
        torch.testing.assert_close(
            captured["negative_prompt_embeds"],
            torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
        )
        torch.testing.assert_close(
            captured["prompt_attention_mask"],
            torch.stack([prompt_attention_mask_a, prompt_attention_mask_b], dim=0),
        )
        torch.testing.assert_close(
            captured["negative_prompt_attention_mask"],
            torch.stack([negative_attention_mask_a, negative_attention_mask_b], dim=0),
        )

    @pytest.mark.parametrize(("cfg_rank", "expected_prompt_value"), [(0, 1.0), (1, 0.0)])
    @pytest.mark.parametrize(
        ("frame_rate_input", "audio_sampling_rate", "expected_frame_rate"),
        [(1.0, 1, 1.0), (None, 24, 24.0)],
    )
    def test_forward_cfg_parallel_steps_video_and_audio_scheduler(
        self,
        monkeypatch,
        cfg_rank,
        expected_prompt_value,
        frame_rate_input,
        audio_sampling_rate,
        expected_frame_rate,
    ):
        from vllm_omni.diffusion.models.ltx2 import ltx2_denoise, ltx2_guidance, ltx2_pipeline_runtime
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2 as ltx23
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = object.__new__(ltx23.LTX23Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.device = torch.device("cpu")
        pipe.tokenizer_max_length = 1
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.audio_sampling_rate = audio_sampling_rate
        pipe.audio_hop_length = 1
        pipe.audio_vae_temporal_compression_ratio = 1
        pipe.audio_vae_mel_compression_ratio = 1
        pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
        pipe.tokenizer = SimpleNamespace(padding_side="left")
        pipe.vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.audio_vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(mel_bins=2, latent_channels=1),
        )

        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if torch.equal(tensor, video_pos) or torch.equal(tensor, video_neg):
                    return [video_pos, video_neg]
                if torch.equal(tensor, audio_pos) or torch.equal(tensor, audio_neg):
                    return [audio_pos, audio_neg]
                raise AssertionError(f"Unexpected gathered tensor: {tensor}")

        monkeypatch.setattr(ltx2_pipeline_runtime, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx2_guidance, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx2_guidance, "get_classifier_free_guidance_rank", lambda: cfg_rank)
        monkeypatch.setattr(ltx2_guidance, "get_cfg_group", lambda: FakeCfgGroup())

        def fake_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas=None, mu=None):
            scheduler.sigmas = torch.tensor([0.25, 0.25], device=device)
            return torch.tensor([1.0, 0.5], device=device), 2

        monkeypatch.setattr(ltx2_denoise, "retrieve_timesteps", fake_retrieve_timesteps)

        class FakeScheduler:
            def __init__(self, name="video", calls=None):
                self.name = name
                self.calls = [] if calls is None else calls
                self.config = {
                    "max_image_seq_len": 4096,
                    "base_image_seq_len": 1024,
                    "base_shift": 0.95,
                    "max_shift": 2.05,
                }
                self.sigmas = torch.tensor([0.25, 0.25])

            def __deepcopy__(self, memo):
                return FakeScheduler("audio", self.calls)

            def step(self, noise_pred, t, latents, return_dict=False, generator=None):
                self.calls.append((self.name, noise_pred.clone(), t.clone(), latents.clone()))
                return (latents - noise_pred,)

        class FakeConnectors:
            def to(self, device):
                return self

            def __call__(self, prompt_embeds, prompt_attention_mask, padding_side):
                assert padding_side == "left"
                assert prompt_embeds.shape[0] == 2
                return prompt_embeds, prompt_embeds, prompt_attention_mask

        rope_video_fps: list[float] = []

        class FakeRope:
            def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps):
                rope_video_fps.append(fps)
                return torch.zeros(batch_size, num_frames * height * width, 3, device=device)

            def prepare_audio_coords(self, batch_size, num_frames, device):
                return torch.zeros(batch_size, num_frames, 1, device=device)

        class FakeTransformer:
            def __init__(self):
                self.config = SimpleNamespace(in_channels=2)
                self.rope = FakeRope()
                self.audio_rope = FakeRope()
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                expected_prompt = torch.full((1, 1, 1), expected_prompt_value)
                torch.testing.assert_close(kwargs["encoder_hidden_states"], expected_prompt)
                torch.testing.assert_close(kwargs["audio_encoder_hidden_states"], expected_prompt)
                assert kwargs["hidden_states"].shape == (1, 1, 2)
                assert kwargs["audio_hidden_states"].shape == (1, 1, 2)
                if cfg_rank == 0:
                    return video_pos, audio_pos
                return video_neg, audio_neg

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self):
                pass

        pipe.scheduler = FakeScheduler()
        pipe.connectors = FakeConnectors()
        pipe.transformer = FakeTransformer()
        object.__setattr__(pipe, "progress_bar", lambda total: DummyProgress())

        def fake_encode_prompt(**kwargs):
            return (
                torch.ones(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
            )

        object.__setattr__(pipe, "encode_prompt", fake_encode_prompt)

        video_latents = torch.tensor([[[1.0, -2.0]]])
        audio_latents = torch.tensor([[[0.5, 3.0]]])
        req = OmniDiffusionRequest(
            prompt={"prompt": "prompt", "negative_prompt": "negative"},
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=32,
                num_frames=1,
                frame_rate=frame_rate_input,
                num_inference_steps=2,
                guidance_scale=4.0,
                latents=video_latents,
                audio_latents=audio_latents,
                output_type="latent",
            ),
            request_id="ltx23-cfg-parallel-forward-test",
        )

        output = pipe.forward(DiffusionRequestBatch(requests=[req]))[0]

        expected_video_noise = ltx2_guidance.combine_velocity_via_x0(
            video_latents,
            video_pos,
            video_neg,
            pipe.scheduler.sigmas[0],
            4.0,
        )
        expected_audio_noise = ltx2_guidance.combine_velocity_via_x0(
            audio_latents,
            audio_pos,
            audio_neg,
            pipe.scheduler.sigmas[0],
            4.0,
        )
        scheduler_call_names = [call[0] for call in pipe.scheduler.calls]
        assert scheduler_call_names == ["video", "audio", "video", "audio"]
        assert len(pipe.transformer.calls) == 2
        torch.testing.assert_close(pipe.scheduler.calls[0][1], expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[1][1], expected_audio_noise)
        torch.testing.assert_close(pipe.scheduler.calls[2][1], expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[3][1], expected_audio_noise)
        torch.testing.assert_close(pipe.scheduler.calls[2][3], video_latents - expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[3][3], audio_latents - expected_audio_noise)

        video_out, audio_out = output.output
        torch.testing.assert_close(video_out, (video_latents - 2 * expected_video_noise).reshape(1, 2, 1, 1, 1))
        torch.testing.assert_close(audio_out, (audio_latents - 2 * expected_audio_noise).reshape(1, 1, 1, 2))

        # fps regression guard: an omitted request fps (frame_rate_input=None) must resolve
        # to the model's own 24.0 default, not crash on None; a provided rate is passed through.
        assert rope_video_fps
        assert all(fps == expected_frame_rate for fps in rope_video_fps)
