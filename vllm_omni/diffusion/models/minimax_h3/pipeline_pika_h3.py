# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serve a Pika release directly, with its pinned base and fixed adapter."""

from pathlib import Path
from typing import ClassVar

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.pika import PIKA_BASE_REVISION, PikaSpec, read_pika_spec
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    get_minimax_h3_post_process_func,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_extras.pika_h3 import PIKA_MODEL_ID, PIKA_REVISION

__all__ = ["PikaH3Pipeline", "get_minimax_h3_post_process_func"]


def resolve_pika_h3_release(
    model: str, *, task_type: str | None, revision: str | None, base_model_path: str | None = None
) -> tuple[Path, Path, PikaSpec]:
    """Resolve only the selected adapter and native base partition.

    A local base override reuses an existing original H3 download. It is a
    trusted checkpoint, as with the normal H3 model-path argument.
    """
    task = str(task_type or "auto").lower()
    if task not in {"auto", "t2va", "fl2va", "ref2va"}:
        raise ValueError("Pika requires one task partition: fl2va or ref2va")
    root = Path(model)
    local_spec = read_pika_spec(root) if (root / "manifest.json").is_file() else None
    if task == "auto" and local_spec is not None:
        task = local_spec[1].task
    task = "ref2va" if task == "ref2va" else "fl2va"
    if not root.is_dir():
        if model.rstrip("/") != PIKA_MODEL_ID:
            raise ValueError(f"Expected {PIKA_MODEL_ID} or a local Pika release directory")
        root = Path(
            download_weights_from_hf_specific(
                model_name_or_path=PIKA_MODEL_ID,
                cache_dir=None,
                allow_patterns=[f"{task}/*", "LICENSE", "NOTICE", "licenses/*"],
                revision=revision or PIKA_REVISION,
                require_all=True,
            )
        )
    adapter = root if local_spec is not None else root / task
    resolved = read_pika_spec(adapter)
    if resolved is None or resolved[1].task != task:
        raise ValueError(f"Pika {task} requires its matching adapter manifest")
    subdir = "Ref2VA" if task == "ref2va" else "FL2VA"
    if base_model_path is None:
        base = (
            Path(
                download_weights_from_hf_specific(
                    model_name_or_path="MiniMaxAI/MiniMax-H3",
                    cache_dir=None,
                    allow_patterns=[f"{subdir}/**"],
                    revision=PIKA_BASE_REVISION,
                    require_all=True,
                )
            )
            / subdir
        )
    else:
        base = Path(base_model_path)
        if base.name != subdir:
            base = base / subdir
    if not (base / "model_index.json").is_file():
        raise ValueError(f"Pika requires the original native H3 {subdir} partition: {base}")
    return base, adapter, resolved[1]


class PikaH3Pipeline(MiniMaxH3Pipeline):
    """A fixed Pika model; ordinary H3 retains request-switchable LoRAs."""

    diffusion_lora_fixed_adapter: ClassVar[bool] = True
    default_num_inference_steps: ClassVar[int] = 6

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        if od_config.lora_path is not None or od_config.lora_scale != 1.0:
            raise ValueError("Pika bundles its matching adapter at scale 1; remove the startup LoRA override")
        additional = od_config.additional_config or {}
        base, adapter, spec = resolve_pika_h3_release(
            str(od_config.model),
            task_type=od_config.task_type,
            revision=od_config.revision,
            base_model_path=additional.get("pika_base_model_path"),
        )
        # The worker and loader share this config. Resolve before constructing
        # modules so both base-weight loading and LoRA initialization use it.
        od_config.model = str(base)
        od_config.revision = None
        od_config.task_type = spec.task
        od_config.lora_path = str(adapter)
        super().__init__(od_config=od_config, prefix=prefix)
        self._fixed_pika_spec = spec

    def _active_pika_spec(self, sampling: OmniDiffusionSamplingParams) -> PikaSpec:
        if sampling.lora_request is not None or sampling.lora_scale != 1.0:
            raise OmniClientError("Pika bundles its adapter at scale 1; remove the request LoRA override")
        return self._fixed_pika_spec
