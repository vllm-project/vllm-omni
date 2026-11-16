# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Two-process USP smoke test for the Helios transformer.

Pins the multi-rank mechanism that ``test_helios_usp_shard.py`` (unit tests
of ``_sp_split_seq`` alone) cannot see:

  * the manual ``_sp_shard_depth`` bump in ``forward()`` must make attn1 run
    the Ulysses all-to-all (not ``NoParallelAttention``);
  * attn2 cross-attention must stay rank-local (``skip_sequence_parallel=
    True``): if replicated text K/V were fed through Ulysses they would be
    ws-fold duplicated and outputs would diverge from the single-rank run;
  * end-to-end correctness: ws=2 output equals the ws=1 baseline for the same
    weights/inputs (divisible shapes, so the padding path is not involved).

Runs ``HeliosTransformer3DModel.forward`` with tiny random weights via
``torch.multiprocessing.spawn``: first a baseline (world_size=1), then an
SP run (world_size=2, ulysses_degree=2); rank-0 outputs are compared.

Requires >= 2 devices; skipped otherwise.
"""

import os
import tempfile

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelConfig,
    get_sp_plan_from_model,
)
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.platforms import current_omni_platform

_L4_TWO_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=2)
pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, *_L4_TWO_GPU]

RANDOM_SEED = 42
DTYPE = torch.float32  # keep both runs on the SDPA fallback for tight tolerance

# Tiny but architecturally faithful config:
#   inner_dim = 4 * 32 = 128, rope_dim sums to head_dim (32),
#   current chunk 2x4x6 latent -> (1,2,2) patch -> 2*2*3 = 12 tokens,
#   short history 1x4x6 latent -> (1,2,2) patch -> 6 tokens.
# All component lengths are even, so ws=2 needs no padding replicas.
TINY_MODEL_KWARGS = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=32,
    in_channels=4,
    out_channels=4,
    text_dim=64,
    freq_dim=32,
    ffn_dim=128,
    num_layers=2,
    rope_dim=(12, 10, 10),
    rope_theta=10000.0,
    guidance_cross_attn=True,
    zero_history_timestep=True,
    has_multi_term_memory_patch=True,
    is_amplify_history=False,
)

BATCH = 1
CUR_FRAMES, LAT_H, LAT_W = 2, 4, 6  # -> 12 current tokens
HIST_FRAMES = 1  # short history -> 6 tokens
TEXT_LEN = 5


def _update_env(envs_dict: dict[str, str]) -> None:
    for k, v in envs_dict.items():
        os.environ[k] = v


def _make_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(RANDOM_SEED + 1)
    return {
        "hidden_states": torch.randn(BATCH, 4, CUR_FRAMES, LAT_H, LAT_W, device=device, dtype=DTYPE),
        "timestep": torch.full((BATCH,), 500.0, device=device),
        "encoder_hidden_states": torch.randn(
            BATCH, TEXT_LEN, TINY_MODEL_KWARGS["text_dim"], device=device, dtype=DTYPE
        ),
        "indices_latents_history_short": torch.tensor([[0]], device=device),
        "latents_history_short": torch.randn(BATCH, 4, HIST_FRAMES, LAT_H, LAT_W, device=device, dtype=DTYPE),
    }


def _spy_strategies(model: HeliosTransformer3DModel) -> dict[str, list[str]]:
    """Record the parallel strategy each Attention resolves to during forward."""
    seen: dict[str, list[str]] = {"attn1": [], "attn2": []}
    for name, attn in (("attn1", model.blocks[0].attn1.attn), ("attn2", model.blocks[0].attn2.attn)):
        original = attn._get_active_parallel_strategy

        def spy(original=original, name=name):
            strategy = original()
            seen[name].append(getattr(strategy, "name", type(strategy).__name__))
            return strategy

        attn._get_active_parallel_strategy = spy
    return seen


def helios_forward_on_model(
    local_rank: int,
    world_size: int,
    ulysses_degree: int,
    output_file: str,
    model_state_file: str,
    is_baseline: bool,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(DTYPE)

    _update_env(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12361",
        }
    )
    init_distributed_environment()

    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=ulysses_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        allgather_degree=1,
        cfg_parallel_size=1,
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        model="test_model",
        dtype=DTYPE,
        parallel_config=parallel_config,
        diffusion_attention_backend="TORCH_SDPA",
    )
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=ulysses_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        allgather_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
        torch.manual_seed(RANDOM_SEED)
        model = HeliosTransformer3DModel(**TINY_MODEL_KWARGS).to(device).to(DTYPE)

        if is_baseline and local_rank == 0:
            with open(model_state_file, "wb") as f:
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f)

        if world_size > 1:
            torch.distributed.barrier()

        with open(model_state_file, "rb") as f:
            model_state = torch.load(f)
        model.load_state_dict({k: v.to(device).to(DTYPE) for k, v in model_state.items()})

        if not is_baseline:
            # Mirror registry._apply_sequence_parallel_if_enabled: install the
            # proj_out gather hook from the model's own _sp_plan and flip
            # sp_plan_hooks_applied so sp_active follows _sp_shard_depth
            # (i.e. the manual bump in Helios forward is what enables Ulysses).
            plan = get_sp_plan_from_model(model)
            assert plan is not None, "HeliosTransformer3DModel must define _sp_plan"
            apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=ulysses_degree), plan)
            get_forward_context().sp_plan_hooks_applied = True

        seen = _spy_strategies(model)
        inputs = _make_inputs(device)
        output = model(**inputs).sample

        assert output.shape == (BATCH, 4, CUR_FRAMES, LAT_H, LAT_W), f"unexpected output shape {output.shape}"

        if is_baseline:
            assert seen["attn1"] and all(n == "none" for n in seen["attn1"]), (
                f"baseline attn1 must use NoParallelAttention ('none'), saw {seen['attn1']}"
            )
        else:
            # attn1: SP communication must be active inside the block loop
            # (regression here = the manual _sp_shard_depth bump was lost).
            assert seen["attn1"] and all(n == "ulysses" for n in seen["attn1"]), (
                f"attn1 must run Ulysses under USP (manual _sp_shard_depth bump), saw {seen['attn1']}"
            )
            # attn2: replicated text K/V must stay rank-local
            # (regression here = the ws-fold duplicated-keys bug).
            assert seen["attn2"] and all(n == "none" for n in seen["attn2"]), (
                f"attn2 must skip sequence parallel (replicated text K/V), saw {seen['attn2']}"
            )

        if local_rank == 0:
            with open(output_file, "wb") as f:
                torch.save(output.detach().cpu().float(), f)

        destroy_distributed_env()


def test_helios_usp2_matches_single_rank():
    """ws=2 (Ulysses) must reproduce the ws=1 output for identical weights/inputs."""
    available_devices = current_omni_platform.get_device_count()
    if available_devices < 2:
        pytest.skip(f"Test requires 2 devices but only {available_devices} available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        baseline_output_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        sp_output_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        model_state_file = f.name

    try:
        torch.multiprocessing.spawn(
            helios_forward_on_model,
            args=(1, 1, baseline_output_file, model_state_file, True),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            helios_forward_on_model,
            args=(2, 2, sp_output_file, model_state_file, False),
            nprocs=2,
        )

        with open(baseline_output_file, "rb") as f:
            baseline = torch.load(f)
        with open(sp_output_file, "rb") as f:
            sp = torch.load(f)

        assert baseline.shape == sp.shape
        max_abs_diff = (baseline - sp).abs().max().item()
        assert torch.allclose(baseline, sp, atol=1e-3, rtol=1e-3), (
            f"USP=2 output diverges from single-rank baseline: max_abs_diff={max_abs_diff:.6e}"
        )
    finally:
        for path in [baseline_output_file, sp_output_file, model_state_file]:
            if os.path.exists(path):
                os.remove(path)
