# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's engine objects for the unit tests, built the way the engine builds them.

A stage's ``VllmConfig`` comes from a deploy file through the orchestrator's
own resolution, and a stage from that config through vLLM's model
initializer. The model directory holds what the engine writes for the
config-less Hub repos: a two-line ``config.json``.
"""

import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import GPT2Model, LlamaModel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.engine.stage_init_utils import (
    build_vllm_config,
    get_stage_connector_spec,
    load_omni_transfer_config_for_model,
)
from vllm_omni.model_executor.models.chatterbox.chatterbox import ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import S3GenDecoder
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import HEAD_KEYS, ChatterboxT3ForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.conditioning import TOKENIZER_BUFFERS, VoiceConditioning
from vllm_omni.model_executor.models.chatterbox.voice_encoder import VoiceEncConfig, VoiceEncoder
from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

# What ``OmniEngineArgs`` writes for a Chatterbox repo that ships no config.json.
CONFIG_JSON = {"model_type": "chatterbox", "architectures": ["ChatterboxForConditionalGeneration"]}
# Each checkpoint's own deploy file.
DEPLOY = {"turbo": "chatterbox_turbo.yaml", "original": "chatterbox.yaml"}
# Played tokens stage 1 decodes a chunk with, as the deploy files set ``codec_left_context_frames``.
LEFT_CONTEXT = 50
# Stage 1's seed in the deploy files, which seeds its flow noise.
SEED = 0


def random_voice(cond_tokens: int, prompt_tokens: int) -> VoiceConditioning:
    """A reference's conditioning with random values, drawn from the global generator.

    Args:
        cond_tokens: Stage 0's prompt tokens, at most the variant's ``cond_prompt_len``.
        prompt_tokens: Stage 1's prompt tokens; its mel has two frames per token.
    """
    return VoiceConditioning(
        cond_tokens=torch.randint(0, 6561, (1, cond_tokens)),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.randint(0, 6561, (1, prompt_tokens)),
        prompt_feat=torch.randn(1, 2 * prompt_tokens, 80),
        embedding=torch.randn(1, 192),
    )


def write_model_dir(path: Path) -> str:
    """Make ``path`` a Chatterbox model directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(CONFIG_JSON))
    return str(path)


def stage_vllm_config(model: str, deploy: str, stage_id: int, **overrides: object) -> VllmConfig:
    """One stage's engine config, as the orchestrator builds it from the deploy file.

    Args:
        model: The model directory.
        deploy: The deploy file's name in ``vllm_omni/deploy``, or a path to one.
        stage_id: The stage.
        **overrides: Per-stage engine arguments over the deploy file's, as
            ``--stage-overrides`` gives them.

    Returns:
        The stage's ``VllmConfig``, with the connector the deploy file names.
    """
    resolution = resolve_omni_config(
        model,
        trust_remote_code=None,
        deploy_config_path=deploy if Path(deploy).is_absolute() else get_deploy_config_path(deploy),
        cli_overrides={},
        stage_overrides={str(stage_id): overrides} if overrides else None,
        strategy_config_path=None,
    )
    stage = resolution.stage_configs[stage_id]
    spec = get_stage_connector_spec(
        load_omni_transfer_config_for_model(model, resolution.config_path),
        stage_id,
        async_chunk=stage.connector_config.async_chunk,
    )
    # build_vllm_config sets this for every multi-stage engine; keep it out of the test process.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        vllm_config, _ = build_vllm_config(stage, model, stage_connector_spec=spec)
    return vllm_config


def build_stage(vllm_config: VllmConfig, device: str) -> ChatterboxForConditionalGeneration:
    """The registered architecture as vLLM's loader initializes it, before any weights load."""
    with set_default_torch_dtype(vllm_config.model_config.dtype), torch.device(device):
        return initialize_model(vllm_config)


def load_stage_weights(stage: ChatterboxForConditionalGeneration, vllm_config: VllmConfig) -> None:
    """Load the model directory's checkpoint through vLLM's loader, as a worker does.

    The loader fails on any parameter the stage reports it did not fill.
    """
    with set_current_vllm_config(vllm_config):
        get_model_loader(vllm_config.load_config).load_weights(stage, vllm_config.model_config)


@contextmanager
def model_parallel() -> Iterator[None]:
    """vLLM's process groups for one rank, which its parallel layers read when built.

    Gloo serves both a CPU and a GPU stage: one rank runs no collective.
    """
    if parallel_state.model_parallel_is_initialized():
        yield
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    parallel_state.init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    with set_current_vllm_config(VllmConfig()):
        parallel_state.ensure_model_parallel_initialized(1, 1)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def t3_checkpoint(stage: ChatterboxT3ForConditionalGeneration, seed: int) -> dict[str, torch.Tensor]:
    """A stage 0 checkpoint in the released file's layout, with seeded random values.

    Upstream's T3 wraps transformers' GPT-2 (Turbo) or LLaMA (Original) and
    saves it under ``tfmr.``; the backbone here is that module, built from
    the backbone config the stage itself made. The heads are the stage's own
    modules under the checkpoint's names, and ``text_head`` is the
    training-only head the file also carries.

    Args:
        stage: The stage whose checkpoint to write.
        seed: Seed of the values.

    Returns:
        The checkpoint's tensors by name, in the stage's dtype.
    """
    backbone_cls = GPT2Model if stage.tfmr.config.model_type == "gpt2" else LlamaModel
    with torch.device("meta"):
        backbone = backbone_cls(stage.tfmr.config)
    head_names = {parameter: name for name, parameter in HEAD_KEYS.items()}
    names = {f"tfmr.{name}": tensor for name, tensor in backbone.state_dict().items()}
    names |= {head_names.get(name, name): tensor for name, tensor in stage.heads.state_dict().items()}
    generator = torch.Generator().manual_seed(seed)
    checkpoint = {
        name: torch.randn(tensor.shape, generator=generator).mul_(0.02).to(stage.heads.text_emb.weight.dtype)
        for name, tensor in names.items()
    }
    checkpoint["text_head.weight"] = torch.zeros_like(checkpoint["text_emb.weight"])
    return checkpoint


def write_t3_checkpoint(model_dir: str, stage: ChatterboxT3ForConditionalGeneration, seed: int) -> dict:
    """Write ``t3_checkpoint`` as the file the stage's ``allow_patterns_overrides`` names."""
    checkpoint = t3_checkpoint(stage, seed)
    save_file(checkpoint, str(Path(model_dir) / stage.config.t3_weights))
    return checkpoint


def write_s3gen_checkpoint(model_dir: str, config: ChatterboxConfig, seed: int) -> dict[str, torch.Tensor]:
    """Write a stage 1 checkpoint in the released file's layout, with seeded initial weights.

    The file holds the decoder's ``flow.*`` and ``mel2wav.*`` tensors, from
    a decoder that has never run, and the reference encoders the API process
    loads from it: the S3 tokenizer under ``tokenizer.`` (with the two
    buffers upstream's subclass saves beside its parameters) and CAMPPlus
    under ``speaker_encoder.``, each from its module definition.

    Args:
        model_dir: The model directory.
        config: The checkpoint's config.
        seed: Seed of the weights.

    Returns:
        The checkpoint's tensors by name.
    """
    from s3tokenizer.model_v2 import S3TokenizerV2

    torch.manual_seed(seed)
    decoder = S3GenDecoder(config, config.max_new_tokens, LEFT_CONTEXT, SEED).state_dict()
    tokenizer = S3TokenizerV2(config.s3_tokenizer_name).state_dict()
    speaker_encoder = CAMPPlus(feat_dim=80, embedding_size=192, memory_efficient=False).state_dict()
    buffers = dict(zip(TOKENIZER_BUFFERS, (torch.zeros(128, 201), torch.hann_window(400)), strict=True))
    checkpoint = {
        **decoder,
        **{f"tokenizer.{name}": tensor for name, tensor in tokenizer.items()},
        **buffers,
        **{f"speaker_encoder.{name}": tensor for name, tensor in speaker_encoder.items()},
    }
    save_file(checkpoint, str(Path(model_dir) / config.s3gen_weights))
    return checkpoint


def write_voice_encoder(model_dir: str, seed: int) -> dict[str, torch.Tensor]:
    """Write ``ve.safetensors`` from the voice encoder's definition with seeded initial weights."""
    torch.manual_seed(seed)
    weights = VoiceEncoder(VoiceEncConfig()).state_dict()
    save_file(weights, str(Path(model_dir) / "ve.safetensors"))
    return weights
