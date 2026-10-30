# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP32 Soprano decoder in the existing non-AR generation worker."""

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.soprano.contracts import SAMPLE_RATE
from vllm_omni.model_executor.models.soprano.vocos import SopranoDecoder

logger = init_logger(__name__)


class SopranoDecoderForGeneration(nn.Module):
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    supports_native_payload_input = True
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        if vllm_config.model_config.dtype != torch.float32:
            raise ValueError("Soprano decoder requires dtype=float32")
        self.decoder = SopranoDecoder()

    def compute_logits(self, hidden_states, sampling_metadata=None) -> None:
        # GenerationModelRunner emits waveform payloads without AR sampling;
        # vLLM's model inspection still requires this generation interface.
        return None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # IDs are generation-worker control slots, never audio codec indices.
        return torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.float32)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The default loader offers AR safetensors for both stages. None are
        # decoder parameters; record every ignored AR key with its owner.
        self.ar_keys_owned_by_stage0 = [name for name, _ in weights]
        cfg = self.vllm_config.model_config
        root = Path(cfg.model)
        if not root.is_dir():
            root = Path(
                download_weights_from_hf(
                    cfg.model, self.vllm_config.load_config.download_dir, ["decoder.pth"], revision=cfg.revision
                )
            )
        state = torch.load(root / "decoder.pth", map_location="cpu", weights_only=True)
        self.decoder.load_state_dict(state, strict=True)
        self.checkpoint_key_audit = [
            {"key": key, "shape": list(value.shape), "destination": f"decoder.{key}"} for key, value in state.items()
        ]
        self.decoder.to(device=self.vllm_config.device_config.device, dtype=torch.float32).eval()
        for row in self.checkpoint_key_audit:
            logger.info("Soprano decoder key %s -> %s shape=%s", row["key"], row["destination"], row["shape"])
        logger.info(
            "Soprano decoder: %d AR keys owned by stage 0, %d decoder tensors strictly loaded",
            len(self.ar_keys_owned_by_stage0),
            len(state),
        )
        return {f"decoder.{name}" for name, _ in self.decoder.named_parameters()}

    def get_dummy_runtime_additional_information(self, num_reqs):
        return [{"hidden_states": {"output": torch.zeros(2, 512, dtype=torch.float32)}} for _ in range(num_reqs)]

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        runtime_additional_information=None,
        request_ids=None,
        **runner_metadata,
    ):
        """Delta contract: each complete request yields exactly one waveform."""
        infos = runtime_additional_information or []
        if not infos:
            raise ValueError("Soprano decoder has no request payload")
        audio = []
        truncated = []
        synthesis_errors = []
        device = self.decoder.head.out.weight.device
        for info in infos:
            error = info.get("additional_information", {}).get("soprano_error")
            hidden = info.get("hidden_states", {}).get("output")
            invalid = error or hidden is None or hidden.ndim != 2 or hidden.shape[1] != 512 or hidden.shape[0] < 2
            if invalid:
                # A terminal request error must not escape the worker loop.
                # Empty audio is accompanied by an explicit failure marker;
                # callers must not encode it as a successful WAV.
                audio.append(torch.empty(0, device=device, dtype=torch.float32))
                synthesis_errors.append(torch.tensor([True]))
            else:
                hidden = hidden.to(device=device, dtype=torch.float32)
                audio.append(self.decoder(hidden.T.unsqueeze(0))[0])
                synthesis_errors.append(torch.tensor([False]))
            sampled_ids = info.get("ids", {}).get("output", [])
            truncated.append(torch.tensor([bool(sampled_ids and sampled_ids[-1] != 3)]))
        return OmniOutput(
            text_hidden_states=torch.empty(0, device=audio[0].device),
            multimodal_outputs={
                "model_outputs": audio,
                "sr": SAMPLE_RATE,
                "audio_truncated": truncated,
                "audio_synthesis_error": synthesis_errors,
            },
        )
