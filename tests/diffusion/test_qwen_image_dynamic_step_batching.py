# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _state(
    request_id: str,
    *,
    image_len: int,
    txt_len: int,
    negative_txt_len: int | None = None,
    true_cfg_scale: float | None = None,
    img_shape: list[tuple[int, int, int]] | None = None,
) -> DiffusionRequestState:
    sampling = OmniDiffusionSamplingParams(num_inference_steps=1)
    sampling.true_cfg_scale = true_cfg_scale
    state = DiffusionRequestState(request_id=request_id, sampling=sampling, prompts=["prompt"])
    state.latents = torch.zeros((1, image_len, 2), dtype=torch.float32)
    state.timesteps = torch.tensor([1.0])
    state.prompt_embeds = torch.ones((1, txt_len, 3), dtype=torch.float32)
    state.prompt_embeds_mask = torch.ones((1, txt_len), dtype=torch.bool)
    state.img_shapes = [img_shape or [(1, 1, image_len)]]
    state.txt_seq_lens = [txt_len]
    if negative_txt_len is not None:
        state.do_true_cfg = True
        state.negative_prompt_embeds = torch.full((1, negative_txt_len, 3), 2.0, dtype=torch.float32)
        state.negative_prompt_embeds_mask = torch.ones((1, negative_txt_len), dtype=torch.bool)
        state.negative_txt_seq_lens = [negative_txt_len]
    return state


def test_qwen_image_metadata_separates_request_and_segment_boundaries() -> None:
    states = [
        _state("req-1", image_len=4, txt_len=5),
        _state("req-2", image_len=9, txt_len=3),
    ]

    input_batch = InputBatch.make_batch(states, allow_dynamic=True)
    metadata = DiffusionModelRunner._make_qwen_image_metadata(
        object.__new__(DiffusionModelRunner),
        input_batch,
        branch="positive",
    )

    assert input_batch.is_dynamic is True
    assert input_batch.request_spans[0].token_start == 0
    assert input_batch.request_spans[0].token_count == 4
    assert input_batch.request_spans[1].token_start == 4
    assert input_batch.request_spans[1].token_count == 9

    assert metadata.padded_tokens == 0
    assert metadata.q_cu_seqlens.tolist() == [0, 9, 21]
    assert metadata.kv_cu_seqlens.tolist() == [0, 9, 21]

    segments = metadata.q_segments
    assert segments is not None
    assert [(s.request_id, s.role, s.packed_start, s.length, s.request_start) for s in segments] == [
        ("req-1", "text", 0, 5, 0),
        ("req-1", "image", 5, 4, 0),
        ("req-2", "text", 9, 3, 0),
        ("req-2", "image", 12, 9, 0),
    ]
    assert input_batch.request_spans[1].token_start != segments[2].packed_start


def test_qwen_image_single_request_uses_dynamic_view_when_enabled() -> None:
    input_batch = InputBatch.make_batch(
        [_state("req-1", image_len=4, txt_len=5)],
        allow_dynamic=True,
    )

    assert input_batch.is_dynamic is True
    assert input_batch.latents is None
    assert input_batch.dynamic_latents is not None
    assert len(input_batch.dynamic_latents) == 1


def test_qwen_image_dynamic_metadata_keeps_shape_and_cfg_request_local() -> None:
    states = [
        _state(
            "req-1",
            image_len=4,
            txt_len=5,
            negative_txt_len=6,
            true_cfg_scale=2.0,
            img_shape=[(1, 2, 2)],
        ),
        _state(
            "req-2",
            image_len=9,
            txt_len=3,
            negative_txt_len=4,
            true_cfg_scale=7.0,
            img_shape=[(1, 3, 3)],
        ),
    ]

    input_batch = InputBatch.make_batch(states, allow_dynamic=True)
    positive = DiffusionModelRunner._make_qwen_image_metadata(
        object.__new__(DiffusionModelRunner),
        input_batch,
        branch="positive",
    )
    negative = DiffusionModelRunner._make_qwen_image_metadata(
        object.__new__(DiffusionModelRunner),
        input_batch,
        branch="negative",
    )

    assert input_batch.is_dynamic is True
    assert input_batch.do_true_cfg is True
    assert input_batch.true_cfg_scales == [2.0, 7.0]
    assert input_batch.img_shapes == [[(1, 2, 2)], [(1, 3, 3)]]

    assert positive.position["img_shapes"] == [[(1, 2, 2)], [(1, 3, 3)]]
    assert positive.position["txt_seq_lens"] == [5, 3]
    assert positive.q_cu_seqlens.tolist() == [0, 9, 21]
    assert negative.position["img_shapes"] == [[(1, 2, 2)], [(1, 3, 3)]]
    assert negative.position["txt_seq_lens"] == [6, 4]
    assert negative.q_cu_seqlens.tolist() == [0, 10, 23]

    negative_segments = negative.q_segments
    assert negative_segments is not None
    assert [
        (segment.request_id, segment.role, segment.packed_start, segment.length) for segment in negative_segments
    ] == [
        ("req-1", "negative_text", 0, 6),
        ("req-1", "image", 6, 4),
        ("req-2", "negative_text", 10, 4),
        ("req-2", "image", 14, 9),
    ]


def test_qwen_image_dynamic_batch_fails_fast_with_cache_backend() -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = type(
        "QwenImagePipeline",
        (),
        {
            "__module__": "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image",
            "modules": lambda self: [],
        },
    )()
    runner.od_config = SimpleNamespace(
        step_execution=True,
        max_num_seqs=2,
        enforce_eager=True,
        cache_backend="tea_cache",
        diffusion_kv_cache_dtype=None,
        parallel_config=SimpleNamespace(
            sequence_parallel_size=1,
            ring_degree=1,
            cfg_parallel_size=1,
            use_hsdp=False,
        ),
    )
    input_batch = InputBatch.make_batch(
        [_state("req-1", image_len=4, txt_len=5)],
        allow_dynamic=True,
    )

    with pytest.raises(ValueError, match="cache backends"):
        runner._validate_qwen_image_dynamic_step_batch(input_batch)


@pytest.mark.cuda
def test_flash_attention_flat_varlen_matches_per_request_attention() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention varlen comparison.")
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.attention.backends.utils.fa import HAS_FLASH_ATTN

    if not HAS_FLASH_ATTN:
        pytest.skip("FlashAttention is not installed.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    lengths = [3, 5]
    num_heads = 2
    head_dim = 64
    total_tokens = sum(lengths)
    query = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor([0, lengths[0], total_tokens], device=device, dtype=torch.int32)
    metadata = AttentionMetadata(
        q_cu_seqlens=cu_seqlens,
        kv_cu_seqlens=cu_seqlens,
        max_q_len=max(lengths),
        max_kv_len=max(lengths),
        padded_tokens=0,
    )

    impl = FlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        causal=False,
        num_kv_heads=num_heads,
    )
    actual = impl.forward_cuda(query, key, value, metadata)

    expected_parts: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        q = query[offset : offset + length].float()
        k = key[offset : offset + length].float()
        v = value[offset : offset + length].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * (head_dim**-0.5)
        probs = torch.softmax(scores, dim=-1)
        expected_parts.append(torch.einsum("hqk,khd->qhd", probs, v).to(actual.dtype))
        offset += length

    assert metadata.padded_tokens == 0
    torch.testing.assert_close(actual, torch.cat(expected_parts, dim=0), rtol=2e-2, atol=2e-2)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _DeterministicAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_cu_seqlens_by_call: list[list[int] | None] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata=None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            self.q_cu_seqlens_by_call.append(None)
        else:
            assert attn_metadata.padded_tokens == 0
            assert attn_metadata.q_cu_seqlens is not None
            self.q_cu_seqlens_by_call.append(attn_metadata.q_cu_seqlens.detach().cpu().tolist())
        return query + key + value


def _qwen_image_dynamic_metadata(
    *,
    image_lens: list[int],
    txt_lens: list[int],
    device: torch.device,
):
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        AttentionSegment,
    )

    cu_values = [0]
    segments: list[AttentionSegment] = []
    for request_index, (image_len, txt_len) in enumerate(zip(image_lens, txt_lens, strict=True)):
        packed_start = cu_values[-1]
        segments.append(
            AttentionSegment(
                request_id=f"req-{request_index}",
                request_index=request_index,
                role="text",
                packed_start=packed_start,
                length=txt_len,
                request_start=0,
            )
        )
        segments.append(
            AttentionSegment(
                request_id=f"req-{request_index}",
                request_index=request_index,
                role="image",
                packed_start=packed_start + txt_len,
                length=image_len,
                request_start=0,
            )
        )
        cu_values.append(packed_start + txt_len + image_len)

    cu_seqlens = torch.tensor(cu_values, device=device, dtype=torch.int32)
    max_seq_len = max(txt_len + image_len for txt_len, image_len in zip(txt_lens, image_lens, strict=True))
    return AttentionMetadata(
        q_cu_seqlens=cu_seqlens,
        kv_cu_seqlens=cu_seqlens,
        max_q_len=max_seq_len,
        max_kv_len=max_seq_len,
        q_segments=segments,
        kv_segments=segments,
        padded_tokens=0,
    )


@pytest.mark.cuda
def test_qwen_image_dit_dynamic_mock_attention_is_exact_across_batch_shapes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen-Image DiT RoPE execution.")

    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(_free_port())

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device="cuda"),
    )
    parallel_config = SimpleNamespace(
        use_hsdp=False,
        sequence_parallel_size=1,
        ring_degree=1,
        cfg_parallel_size=1,
    )
    od_config = SimpleNamespace(
        parallel_config=parallel_config,
        quantization_config=None,
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
        )
        initialize_model_parallel()
        try:
            torch.manual_seed(7)
            model = QwenImageTransformer2DModel(
                od_config=od_config,
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                attention_head_dim=8,
                num_attention_heads=2,
                joint_attention_dim=8,
                axes_dims_rope=(2, 2, 4),
            ).to(device="cuda", dtype=torch.float32)
            for param in model.parameters():
                torch.nn.init.uniform_(param, -0.02, 0.02)
            model.eval()

            attention_modules: list[_DeterministicAttention] = []
            for block in model.transformer_blocks:
                attention = _DeterministicAttention()
                block.attn.attn = attention
                attention_modules.append(attention)

            image_lens = [4, 9]
            txt_lens = [3, 5]
            img_shapes = [[(1, 2, 2)], [(1, 3, 3)]]
            generator = torch.Generator(device="cpu").manual_seed(11)
            hidden_states = [
                torch.randn((1, image_len, 4), generator=generator, dtype=torch.float32).to("cuda")
                for image_len in image_lens
            ]
            encoder_states = [
                torch.randn((1, txt_len, 8), generator=generator, dtype=torch.float32).to("cuda")
                for txt_len in txt_lens
            ]
            timestep = torch.tensor([0.9, 0.7], device="cuda", dtype=torch.float32)

            with torch.inference_mode():
                dense_outputs = []
                for index, txt_len in enumerate(txt_lens):
                    dense_outputs.append(
                        model(
                            hidden_states=hidden_states[index],
                            encoder_hidden_states=encoder_states[index],
                            encoder_hidden_states_mask=torch.ones(
                                (1, txt_len),
                                device="cuda",
                                dtype=torch.bool,
                            ),
                            timestep=timestep[index : index + 1],
                            img_shapes=[img_shapes[index]],
                            txt_seq_lens=[txt_len],
                            guidance=None,
                        ).sample
                    )

                max_txt_len = max(txt_lens)
                batched_encoder = torch.zeros((2, max_txt_len, 8), device="cuda")
                batched_mask = torch.zeros((2, max_txt_len), device="cuda", dtype=torch.bool)
                for index, txt_len in enumerate(txt_lens):
                    batched_encoder[index, :txt_len].copy_(encoder_states[index][0])
                    batched_mask[index, :txt_len] = True

                attention_id = "qwen_image.joint.positive"
                multi_metadata = _qwen_image_dynamic_metadata(
                    image_lens=image_lens,
                    txt_lens=txt_lens,
                    device=torch.device("cuda"),
                )
                with set_forward_context(
                    vllm_config=vllm_config,
                    omni_diffusion_config=od_config,
                    attn_metadata={attention_id: multi_metadata},
                ):
                    dynamic_multi_outputs = model(
                        hidden_states=hidden_states,
                        encoder_hidden_states=batched_encoder,
                        encoder_hidden_states_mask=batched_mask,
                        timestep=timestep,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_lens,
                        guidance=None,
                        attention_kwargs={"attention_id": attention_id},
                    ).sample

                dynamic_single_outputs = []
                for index, txt_len in enumerate(txt_lens):
                    single_metadata = _qwen_image_dynamic_metadata(
                        image_lens=[image_lens[index]],
                        txt_lens=[txt_len],
                        device=torch.device("cuda"),
                    )
                    with set_forward_context(
                        vllm_config=vllm_config,
                        omni_diffusion_config=od_config,
                        attn_metadata={attention_id: single_metadata},
                    ):
                        dynamic_single_outputs.append(
                            model(
                                hidden_states=[hidden_states[index]],
                                encoder_hidden_states=encoder_states[index],
                                encoder_hidden_states_mask=torch.ones(
                                    (1, txt_len),
                                    device="cuda",
                                    dtype=torch.bool,
                                ),
                                timestep=timestep[index : index + 1],
                                img_shapes=[img_shapes[index]],
                                txt_seq_lens=[txt_len],
                                guidance=None,
                                attention_kwargs={"attention_id": attention_id},
                            ).sample[0]
                        )

            torch.accelerator.synchronize()
            for dense, dynamic_single, dynamic_multi in zip(
                dense_outputs,
                dynamic_single_outputs,
                dynamic_multi_outputs,
                strict=True,
            ):
                torch.testing.assert_close(dynamic_single, dense, rtol=0, atol=0)
                torch.testing.assert_close(dynamic_multi, dense, rtol=0, atol=0)
                torch.testing.assert_close(dynamic_multi, dynamic_single, rtol=0, atol=0)

            assert attention_modules[0].q_cu_seqlens_by_call == [
                None,
                None,
                [0, 7, 21],
                [0, 7],
                [0, 14],
            ]
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.cuda
def test_qwen_image_transformer_dynamic_batch_matches_dense_serial() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen-Image dynamic transformer comparison.")

    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.utils.fa import HAS_FLASH_ATTN
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel

    if not HAS_FLASH_ATTN:
        pytest.skip("FlashAttention is not installed.")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(_free_port())

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device="cuda"),
    )
    parallel_config = SimpleNamespace(
        use_hsdp=False,
        sequence_parallel_size=1,
        ring_degree=1,
        cfg_parallel_size=1,
    )
    od_config = SimpleNamespace(
        parallel_config=parallel_config,
        quantization_config=None,
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
        )
        initialize_model_parallel()
        try:
            torch.manual_seed(1234)
            model = QwenImageTransformer2DModel(
                od_config=od_config,
                patch_size=2,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                attention_head_dim=64,
                num_attention_heads=2,
                joint_attention_dim=8,
                axes_dims_rope=(4, 4, 4),
            ).to(device="cuda", dtype=torch.bfloat16)
            for param in model.parameters():
                torch.nn.init.uniform_(param, -0.02, 0.02)
            model.eval()

            image_lens = [16, 64]
            txt_lens = [5, 7]
            img_shapes = [[(1, 4, 4)], [(1, 8, 8)]]
            generator = torch.Generator(device="cpu").manual_seed(5)
            hidden_states = [
                torch.randn((1, image_len, 4), generator=generator, dtype=torch.float32).to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for image_len in image_lens
            ]
            encoder_states = [
                torch.randn((1, txt_len, 8), generator=generator, dtype=torch.float32).to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for txt_len in txt_lens
            ]
            timestep = torch.tensor([0.9, 0.7], device="cuda", dtype=torch.bfloat16)

            with torch.inference_mode():
                dense_outputs = []
                for index, txt_len in enumerate(txt_lens):
                    dense_outputs.append(
                        model(
                            hidden_states=hidden_states[index],
                            encoder_hidden_states=encoder_states[index],
                            encoder_hidden_states_mask=torch.ones(
                                (1, txt_len),
                                device="cuda",
                                dtype=torch.bool,
                            ),
                            timestep=timestep[index : index + 1],
                            img_shapes=[img_shapes[index]],
                            txt_seq_lens=[txt_len],
                            guidance=None,
                        ).sample
                    )

                cu_values = [0]
                for txt_len, image_len in zip(txt_lens, image_lens, strict=True):
                    cu_values.append(cu_values[-1] + txt_len + image_len)
                cu_seqlens = torch.tensor(cu_values, device="cuda", dtype=torch.int32)
                attn_metadata = AttentionMetadata(
                    q_cu_seqlens=cu_seqlens,
                    kv_cu_seqlens=cu_seqlens,
                    max_q_len=max(txt_len + image_len for txt_len, image_len in zip(txt_lens, image_lens, strict=True)),
                    max_kv_len=max(
                        txt_len + image_len for txt_len, image_len in zip(txt_lens, image_lens, strict=True)
                    ),
                    padded_tokens=0,
                )

                max_txt_len = max(txt_lens)
                batched_encoder = torch.zeros((2, max_txt_len, 8), device="cuda", dtype=torch.bfloat16)
                batched_mask = torch.zeros((2, max_txt_len), device="cuda", dtype=torch.bool)
                for index, txt_len in enumerate(txt_lens):
                    batched_encoder[index, :txt_len].copy_(encoder_states[index][0])
                    batched_mask[index, :txt_len] = True

                with set_forward_context(
                    vllm_config=vllm_config,
                    omni_diffusion_config=od_config,
                    attn_metadata={"qwen_image.joint.positive": attn_metadata},
                ):
                    dynamic_outputs = model(
                        hidden_states=hidden_states,
                        encoder_hidden_states=batched_encoder,
                        encoder_hidden_states_mask=batched_mask,
                        timestep=timestep,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_lens,
                        guidance=None,
                        attention_kwargs={"attention_id": "qwen_image.joint.positive"},
                    ).sample

                single_cu_seqlens = torch.tensor(
                    [0, txt_lens[0] + image_lens[0]],
                    device="cuda",
                    dtype=torch.int32,
                )
                single_attn_metadata = AttentionMetadata(
                    q_cu_seqlens=single_cu_seqlens,
                    kv_cu_seqlens=single_cu_seqlens,
                    max_q_len=txt_lens[0] + image_lens[0],
                    max_kv_len=txt_lens[0] + image_lens[0],
                    padded_tokens=0,
                )
                with set_forward_context(
                    vllm_config=vllm_config,
                    omni_diffusion_config=od_config,
                    attn_metadata={"qwen_image.joint.positive": single_attn_metadata},
                ):
                    dynamic_single_output = model(
                        hidden_states=[hidden_states[0]],
                        encoder_hidden_states=encoder_states[0],
                        encoder_hidden_states_mask=torch.ones(
                            (1, txt_lens[0]),
                            device="cuda",
                            dtype=torch.bool,
                        ),
                        timestep=timestep[:1],
                        img_shapes=[img_shapes[0]],
                        txt_seq_lens=[txt_lens[0]],
                        guidance=None,
                        attention_kwargs={"attention_id": "qwen_image.joint.positive"},
                    ).sample[0]

            for dense, dynamic in zip(dense_outputs, dynamic_outputs, strict=True):
                torch.testing.assert_close(dynamic, dense, rtol=0, atol=1e-3)
            torch.testing.assert_close(dynamic_outputs[0], dynamic_single_output, rtol=0, atol=0)
        finally:
            cleanup_dist_env_and_memory()


class _FakeDynamicQwenImagePipeline(torch.nn.Module):
    supports_step_execution = True

    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0
        self.seen_input_batch: InputBatch | None = None
        self.seen_attn_metadata = None

    @property
    def interrupt(self) -> bool:
        return False

    def prepare_encode(self, state: DiffusionRequestState, **kwargs) -> DiffusionRequestState:
        del kwargs
        self.prepare_calls += 1
        image_len = int(state.sampling.extra_args["image_len"])
        txt_len = int(state.sampling.extra_args["txt_len"])
        negative_txt_len = state.sampling.extra_args.get("negative_txt_len")
        img_shape = state.sampling.extra_args.get("img_shape", [(1, 1, image_len)])
        state.timesteps = torch.tensor([1.0])
        state.latents = torch.zeros((1, image_len, 2), dtype=torch.float32)
        state.prompt_embeds = torch.ones((1, txt_len, 3), dtype=torch.float32)
        state.prompt_embeds_mask = torch.ones((1, txt_len), dtype=torch.bool)
        state.img_shapes = [img_shape]
        state.txt_seq_lens = [txt_len]
        if negative_txt_len is not None:
            negative_txt_len = int(negative_txt_len)
            state.do_true_cfg = True
            state.negative_prompt_embeds = torch.full((1, negative_txt_len, 3), 2.0, dtype=torch.float32)
            state.negative_prompt_embeds_mask = torch.ones((1, negative_txt_len), dtype=torch.bool)
            state.negative_txt_seq_lens = [negative_txt_len]
        return state

    def denoise_step(self, input_batch: InputBatch, **kwargs) -> list[torch.Tensor]:
        del kwargs
        from vllm_omni.diffusion.forward_context import get_forward_context

        self.denoise_calls += 1
        self.seen_input_batch = input_batch
        self.seen_attn_metadata = get_forward_context().attn_metadata
        assert input_batch.is_dynamic is True
        assert input_batch.dynamic_latents is not None
        fill_values = (
            input_batch.true_cfg_scales
            if input_batch.do_true_cfg
            else [float(index + 1) for index in range(len(input_batch.dynamic_latents))]
        )
        return [
            torch.full_like(latents, fill_value=float(fill_value))
            for latents, fill_value in zip(input_batch.dynamic_latents, fill_values, strict=True)
        ]

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs) -> None:
        del kwargs
        self.scheduler_calls += 1
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state: DiffusionRequestState, **kwargs) -> DiffusionOutput:
        del kwargs
        self.decode_calls += 1
        return DiffusionOutput(output=state.latents.clone())


QwenImagePipeline = type(
    "QwenImagePipeline",
    (_FakeDynamicQwenImagePipeline,),
    {"__module__": "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image"},
)


def _runner() -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = None
    runner.od_config = SimpleNamespace(
        cache_backend="none",
        enforce_eager=True,
        step_execution=True,
        max_num_seqs=2,
        parallel_config=SimpleNamespace(
            use_hsdp=False,
            sequence_parallel_size=1,
            ring_degree=1,
            cfg_parallel_size=1,
        ),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = QwenImagePipeline()
    runner.state_cache = {}
    return runner


def _request(
    request_id: str,
    *,
    image_len: int,
    txt_len: int,
    negative_txt_len: int | None = None,
    true_cfg_scale: float | None = None,
    img_shape: list[tuple[int, int, int]] | None = None,
) -> OmniDiffusionRequest:
    extra_args = {"image_len": image_len, "txt_len": txt_len}
    if negative_txt_len is not None:
        extra_args["negative_txt_len"] = negative_txt_len
    if img_shape is not None:
        extra_args["img_shape"] = img_shape
    return OmniDiffusionRequest(
        prompts=[f"prompt-{request_id}"],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            true_cfg_scale=true_cfg_scale,
            extra_args=extra_args,
        ),
        request_id=request_id,
    )


def test_qwen_image_dynamic_stepwise_updates_request_local_latents() -> None:
    runner = _runner()
    requests = [
        _request("req-1", image_len=4, txt_len=5),
        _request("req-2", image_len=9, txt_len=3),
    ]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=req.request_id, req=req) for req in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )

    result = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    first = result.get_request_output("req-1")
    second = result.get_request_output("req-2")
    assert first is not None and first.finished is True and first.result is not None
    assert second is not None and second.finished is True and second.result is not None
    torch.testing.assert_close(first.result.output, torch.ones((1, 4, 2)))
    torch.testing.assert_close(second.result.output, torch.full((1, 9, 2), 2.0))

    pipeline = runner.pipeline
    assert pipeline.prepare_calls == 2
    assert pipeline.denoise_calls == 1
    assert pipeline.scheduler_calls == 2
    assert pipeline.decode_calls == 2
    assert pipeline.seen_input_batch is not None
    assert pipeline.seen_input_batch.latents is None
    assert pipeline.seen_input_batch.dynamic_latents is not None
    assert "req-1" not in runner.state_cache
    assert "req-2" not in runner.state_cache


def test_qwen_image_dynamic_stepwise_uses_request_local_cfg_and_shape_metadata() -> None:
    runner = _runner()
    requests = [
        _request(
            "req-1",
            image_len=4,
            txt_len=5,
            negative_txt_len=6,
            true_cfg_scale=2.0,
            img_shape=[(1, 2, 2)],
        ),
        _request(
            "req-2",
            image_len=9,
            txt_len=3,
            negative_txt_len=4,
            true_cfg_scale=7.0,
            img_shape=[(1, 3, 3)],
        ),
    ]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=req.request_id, req=req) for req in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )

    result = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    first = result.get_request_output("req-1")
    second = result.get_request_output("req-2")
    assert first is not None and first.finished is True and first.result is not None
    assert second is not None and second.finished is True and second.result is not None
    torch.testing.assert_close(first.result.output, torch.full((1, 4, 2), 2.0))
    torch.testing.assert_close(second.result.output, torch.full((1, 9, 2), 7.0))

    pipeline = runner.pipeline
    assert pipeline.seen_input_batch is not None
    assert pipeline.seen_input_batch.true_cfg_scales == [2.0, 7.0]
    assert pipeline.seen_input_batch.img_shapes == [[(1, 2, 2)], [(1, 3, 3)]]
    assert pipeline.seen_attn_metadata is not None

    positive = pipeline.seen_attn_metadata["qwen_image.joint.positive"]
    negative = pipeline.seen_attn_metadata["qwen_image.joint.negative"]
    assert positive.position["img_shapes"] == [[(1, 2, 2)], [(1, 3, 3)]]
    assert positive.position["txt_seq_lens"] == [5, 3]
    assert positive.q_cu_seqlens.tolist() == [0, 9, 21]
    assert negative.position["img_shapes"] == [[(1, 2, 2)], [(1, 3, 3)]]
    assert negative.position["txt_seq_lens"] == [6, 4]
    assert negative.q_cu_seqlens.tolist() == [0, 10, 23]
