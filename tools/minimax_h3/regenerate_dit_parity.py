# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-forward DiT parity for Open-H3-Regenerate-2K on real weights.

Runs one MiniMax-H3 transformer forward on a fixed random input at the two
refine sigmas, either as the reference (Diffusers ``MiniMaxH3Transformer3DModel``
from the release's ``base_model/``, LoRA merged as ``cast(W_fp32 + B @ A)`` like
the release) or as vLLM-Omni (native FL2VA checkpoint, LoRA fused at load), and
compares the saved video velocities. ``--no-adapter`` runs both on base H3.

The release's Sharp curve branches are not applied: their up projections are
all zero, which this script asserts, so they add nothing to the reference.

    python tools/minimax_h3/regenerate_dit_parity.py reference ref.pt
    python tools/minimax_h3/regenerate_dit_parity.py vllm-omni omni.pt
    python tools/minimax_h3/regenerate_dit_parity.py compare ref.pt omni.pt

Each forward needs one GPU with about 80 GB free.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import torch

RELEASE_REPO = "Efficient-Large-Model/Open-H3-Regenerate-2K"
RELEASE_REVISION = "5809c33730c18b6b1c827b40974855a4288641b3"
BASE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
LORA_FILE = "lora/open_h3_regenerate_2k_lora.safetensors"
CURVE_FILE = "lora/sharp_curve_branches.safetensors"

SEED = 1234
LATENT_T, LATENT_H, LATENT_W = 7, 16, 32  # 7 latent frames, 256x512 px
AUDIO_T = 40
TEXT_LEN, TEXT_DIM = 48, 5120
SIGMAS = (0.909375, 2 / 3)  # first and last steps of the default 3-step refine


def fixed_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    video = torch.randn(1, 24, LATENT_T, LATENT_H, LATENT_W, generator=generator)
    audio = torch.randn(2, 32, AUDIO_T, generator=generator)  # [channel, latent_dim, T]
    text = torch.randn(TEXT_LEN, TEXT_DIM, generator=generator).to(torch.bfloat16)
    return video, audio, text


def snapshot(repo: str, revision: str, patterns: list[str], local_dir: str | None) -> Path:
    if local_dir:
        return Path(local_dir)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def check_curves_inert(release: Path) -> None:
    from safetensors.torch import load_file

    ups = {name: tensor for name, tensor in load_file(str(release / CURVE_FILE)).items() if ".curve_B." in name}
    assert ups and all(not tensor.any() for tensor in ups.values()), "Sharp curve branches contribute"


def run_reference(args: argparse.Namespace) -> dict[float, torch.Tensor]:
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from safetensors.torch import load_file

    release = snapshot(RELEASE_REPO, RELEASE_REVISION, ["base_model/transformer/*", "lora/*"], args.release_dir)
    device = torch.device("cuda")
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        release / "base_model", subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = model.requires_grad_(False).to(device).eval()
    model.set_attention_backend("_native_cudnn")
    if not args.no_adapter:
        check_curves_inert(release)
        raw = load_file(str(release / LORA_FILE))
        with torch.no_grad():
            for key, down in raw.items():
                if not key.endswith(".lora_A.weight"):
                    continue
                up = raw[key.removesuffix(".lora_A.weight") + ".lora_B.weight"]
                weight = model.get_submodule(key[len("transformer.") : -len(".lora_A.weight")]).weight
                weight.copy_((weight.float() + up.to(device).float() @ down.to(device).float()).to(weight.dtype))

    video, audio, text = fixed_inputs()
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(TEXT_LEN, dtype=torch.long),
        num_latent_frames=LATENT_T,
        latent_height=LATENT_H,
        latent_width=LATENT_W,
        num_audio_latents=AUDIO_T,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
    )
    position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio = layout
    video_rows = patchify_video_latents(video, (1, 2, 2)).reshape(1, -1, 96)
    audio_rows = audio.permute(0, 2, 1).reshape(1, -1, 32)  # channel-major rows
    outputs = {}
    with torch.inference_mode():
        for sigma in SIGMAS:
            # Audio rows are locked at timestep 1, as the release's refiner does.
            unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video_indices=video_indices,
                audio_indices=audio_indices,
                num_condition_video_rows=cond_video,
                num_condition_audio_rows=cond_audio,
                num_text_tokens=int(text_indices.numel()),
                video_timestep=1.0 - sigma,
                audio_timestep=1.0,
                condition_video_timestep=1.0,
                condition_audio_timestep=1.0,
            )
            velocity, _ = model(
                hidden_states=video_rows.to(device, torch.bfloat16),
                audio_hidden_states=audio_rows.to(device, torch.bfloat16),
                encoder_hidden_states=text[None].to(device),
                timestep=unique.to(device),
                timestep_indices=inverse.to(device),
                token_tags=token_tags.to(device),
                position_ids=position_ids.to(device),
                video_indices=video_indices.to(device),
                audio_indices=audio_indices.to(device),
                text_indices=text_indices.to(device),
                attention_kwargs={},
                return_dict=False,
            )
            outputs[sigma] = velocity.float().reshape(-1, 96).cpu()
    return outputs


def run_vllm_omni(args: argparse.Namespace) -> dict[float, torch.Tensor]:
    from safetensors import safe_open
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_pack_audio_latent,
        minimax_h3_patchify_video_latent,
    )
    from vllm_omni.diffusion.models.minimax_h3.regenerate import MiniMaxH3RegenerateAdapter
    from vllm_omni.platforms import current_omni_platform

    base = snapshot(BASE_REPO, BASE_REVISION, ["FL2VA/transformer/*"], args.base_dir)
    transformer_dir = base / "FL2VA" / "transformer" if (base / "FL2VA").is_dir() else base / "transformer"
    device = torch.device("cuda", 0)
    current_omni_platform.set_device(device)
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    od_config = OmniDiffusionConfig(model=str(transformer_dir.parent))
    od_config.set_tf_model_config(
        TransformerConfig.from_dict(json.loads((transformer_dir / "config.json").read_text()))
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tempfile.mktemp()}",
            backend="nccl",
            timeout=timedelta(seconds=120),
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            model = MiniMaxH3DiTModel(od_config, diffusers_weights=False)
        torch.set_default_dtype(torch.float32)

        def stream() -> Iterator[tuple[str, torch.Tensor]]:
            for shard in sorted(transformer_dir.glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        yield name, handle.get_tensor(name)

        adapter = None
        if not args.no_adapter:
            release = snapshot(RELEASE_REPO, RELEASE_REVISION, ["lora/*", "latent_upsampler/*"], args.release_dir)
            adapter = MiniMaxH3RegenerateAdapter.from_path(
                release,
                head_dim=model.arch.attention_head_dim,
                num_blocks=model.arch.num_layers,
                num_refiner_blocks=model.arch.token_refiner_num_layers,
            )
        loaded = model.load_weights(adapter.apply(stream()) if adapter else stream())
        model.post_load_weights()
        if adapter:
            adapter.validate_fully_applied(loaded)
        model.eval()

        video, audio, text = fixed_inputs()
        packed = minimax_h3_packed_sequence(
            text_len=TEXT_LEN,
            latent_t=LATENT_T,
            latent_h=LATENT_H,
            latent_w=LATENT_W,
            audio_t=AUDIO_T,
            include_keyframe_cond=False,
        )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = 1
        branch = MiniMaxH3DenoiseBranch(packed=packed, text_embeddings=text.to(device), token_tags=tags, device=device)
        audio_rows = minimax_h3_pack_audio_latent(audio).to(device)
        branch.locked_audio_rows = audio_rows
        video_rows = minimax_h3_patchify_video_latent(video, patch_size=(1, 2, 2)).to(device)
        outputs = {}
        with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), torch.inference_mode():
            for sigma in SIGMAS:
                kwargs = branch.forward_kwargs(
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    t_video=1.0 - sigma,
                    t_audio=1.0 - sigma,
                    imgvid_cond_timestep=1.0,
                    audio_ref_cond_timestep=1.0,
                    video_target_timesteps=None,
                    audio_target_timesteps=None,
                )
                velocity, _ = model(**kwargs)
                outputs[sigma] = velocity.float()[branch.update_mask_dev].reshape(-1, 96).cpu()
    return outputs


def compare(args: argparse.Namespace) -> None:
    reference, candidate = torch.load(args.reference), torch.load(args.candidate)
    for sigma in SIGMAS:
        expected, actual = reference[sigma], candidate[sigma]
        rel_l2 = float((actual - expected).norm() / expected.norm())
        cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
        print(f"sigma {sigma:.6f}: rel. L2 {rel_l2:.2e}, cosine {cosine:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reference", "vllm-omni"):
        command = commands.add_parser(name)
        command.add_argument("output", help="where to save the video velocities")
        command.add_argument("--no-adapter", action="store_true", help="run base H3 without the LoRA")
        command.add_argument("--release-dir", help=f"local {RELEASE_REPO} snapshot at {RELEASE_REVISION[:9]}")
        if name == "vllm-omni":
            command.add_argument("--base-dir", help=f"local {BASE_REPO} snapshot at {BASE_REVISION[:9]}")
    command = commands.add_parser("compare")
    command.add_argument("reference")
    command.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args)
        return
    outputs = run_reference(args) if args.command == "reference" else run_vllm_omni(args)
    torch.save(outputs, args.output)
    print(f"saved {args.output}: " + ", ".join(f"sigma {s:.6f} std {float(v.std()):.4f}" for s, v in outputs.items()))


if __name__ == "__main__":
    main()
