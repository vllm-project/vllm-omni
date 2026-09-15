# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import socket
import tempfile

import numpy as np
import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    set_forward_context,
)
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.parallel,
]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_dist_env(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
        }
    )


def _transformer_config() -> TransformerConfig:
    return TransformerConfig.from_dict(
        {
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 64,
            "num_layers": 2,
            "num_double_stream_layers": 1,
            "num_refiner_layers": 1,
            "num_attention_heads": 4,
            "num_kv_heads": 1,
            "multiple_of": 32,
            "norm_eps": 1e-5,
            "axes_dim_rope": [8, 4, 4],
            "axes_lens": [64, 32, 32],
            "instruction_feature_configs": {
                "instruction_feat_dim": 32,
                "reduce_type": "mean",
                "num_instruction_feature_layers": 1,
            },
            "prompt_tuning_configs": {"use_prompt_tuning": False},
            "timestep_scale": 1.0,
        }
    )


def _od_config(world_size: int) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="boogu-tiny",
        dtype=torch.float32,
        tf_model_config=_transformer_config(),
        parallel_config=DiffusionParallelConfig(
            sequence_parallel_size=world_size,
            ulysses_degree=world_size,
            ulysses_mode="advanced_uaa",
        ),
    )


def _checkpoint_name(name: str) -> str:
    if ".to_out." in name:
        name = name.replace(".to_out.", ".to_out.0.")
    for projection in (
        "img_to_q",
        "img_to_k",
        "img_to_v",
        "instruct_to_q",
        "instruct_to_k",
        "instruct_to_v",
        "instruct_out",
        "img_out",
    ):
        token = f".img_instruct_attn.{projection}."
        if token in name:
            return name.replace(
                token,
                f".img_instruct_attn.processor.{projection}.",
            )
    return name


def _random_checkpoint(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(2026)
    return [
        (
            _checkpoint_name(name),
            torch.empty_like(param).uniform_(
                -0.02,
                0.02,
                generator=generator,
            ),
        )
        for name, param in model.named_parameters()
    ]


def _worker(
    rank: int,
    world_size: int,
    port: int,
    output_file: str,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank, world_size, port)
    init_distributed_environment(world_size=world_size, rank=rank)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    try:
        from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
            BooguImageDoubleStreamRotaryPosEmbed,
            BooguImageTransformer2DModel,
        )

        od_config = _od_config(world_size)
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=str(device)))),
            set_forward_context(omni_diffusion_config=od_config),
            set_current_diffusion_config(od_config),
        ):
            source = BooguImageTransformer2DModel(od_config).to(device)
            checkpoint = _random_checkpoint(source)
            del source

            model = BooguImageTransformer2DModel(od_config).to(device)
            assert model.load_weights(checkpoint) == set(dict(model.named_parameters()))
            model.eval()

            if world_size > 1:
                apply_sequence_parallel(
                    model,
                    SequenceParallelConfig(ulysses_degree=world_size),
                    model._sp_plan,
                )
                get_forward_context().sp_plan_hooks_applied = True

            generator = torch.Generator(device=device).manual_seed(17)
            latents = torch.randn(
                1,
                4,
                8,
                8,
                device=device,
                generator=generator,
            )
            timestep = torch.tensor([0.5], device=device)
            instruction = torch.randn(
                1,
                7,
                32,
                device=device,
                generator=generator,
            )
            instruction_mask = torch.ones(
                1,
                7,
                dtype=torch.bool,
                device=device,
            )
            ref_image = [
                [
                    torch.randn(
                        4,
                        6,
                        10,
                        device=device,
                        generator=generator,
                    )
                ]
            ]
            # Two differently sized references: their packed segments straddle
            # the SP rank boundary, which is what the per-rank reference-refiner
            # mask has to reproduce.
            ref_images_multi = [
                [
                    torch.randn(4, 6, 10, device=device, generator=generator),
                    torch.randn(4, 4, 8, device=device, generator=generator),
                ]
            ]
            freqs_real = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_real(
                model.axes_dim_rope,
                model.axes_lens,
                theta=10000,
            )

            with torch.inference_mode():
                t2i = model(
                    latents,
                    timestep,
                    instruction,
                    freqs_real,
                    instruction_mask,
                )
                edit = model(
                    latents,
                    timestep,
                    instruction,
                    freqs_real,
                    instruction_mask,
                    ref_image_hidden_states=ref_image,
                )
                edit_multi = model(
                    latents,
                    timestep,
                    instruction,
                    freqs_real,
                    instruction_mask,
                    ref_image_hidden_states=ref_images_multi,
                )

            if rank == 0:
                np.savez(
                    output_file,
                    t2i=t2i.cpu().numpy(),
                    edit=edit.cpu().numpy(),
                    edit_multi=edit_multi.cpu().numpy(),
                )
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_boogu_sp_matches_single_gpu() -> None:
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 2:
        pytest.skip("BOOGU SP test requires two CUDA GPUs.")

    output_files = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".npz",
            ) as handle:
                output_files.append(handle.name)

        torch.multiprocessing.spawn(
            _worker,
            args=(1, _find_free_port(), output_files[0]),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _worker,
            args=(2, _find_free_port(), output_files[1]),
            nprocs=2,
        )

        with (
            np.load(output_files[0], allow_pickle=False) as baseline,
            np.load(output_files[1], allow_pickle=False) as sp_output,
        ):
            for key in ("t2i", "edit", "edit_multi"):
                # fp32 SDPA: the observed SP1-vs-SP2 gap is ~3e-8, so 2e-4
                # would also pass with the SP masking logic entirely removed
                # (that mutation measures ~5e-7). Mask correctness is pinned
                # structurally in tests/diffusion/models/boogu_image/
                # test_sp_layout.py; this bound just stays close to the real
                # reduction-order noise instead of hiding logic errors.
                np.testing.assert_allclose(
                    sp_output[key],
                    baseline[key],
                    rtol=1e-5,
                    atol=1e-5,
                )
    finally:
        for path in output_files:
            try:
                os.remove(path)
            except OSError:
                pass
