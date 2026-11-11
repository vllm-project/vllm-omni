# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP32 Soprano decoder in the existing non-AR generation worker."""

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.soprano.contracts import SAMPLE_RATE
from vllm_omni.model_executor.models.soprano.streaming import DecoderWindow
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
        self._streams: dict[str, DecoderWindow] = {}
        self._failed_streams: set[str] = set()

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Runner IDs (not batch rows) own state; finish and abort both reclaim it."""
        for request_id in finished_req_ids:
            self._failed_streams.discard(request_id)
            removed = self._streams.pop(request_id, None)
            logger.debug(
                "Soprano decoder cleanup req=%s active=%s remaining=%d",
                request_id,
                removed is not None,
                len(self._streams),
            )

    def compute_logits(self, hidden_states, sampling_metadata=None) -> None:
        # GenerationModelRunner emits waveform payloads without AR sampling;
        # vLLM's model inspection still requires this generation interface.
        return None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # IDs are generation-worker control slots, never audio codec indices.
        return torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.float32)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Stage 0 owns and audits AR safetensors. Do not consume its iterator
        # merely to record names; this stage loads only the decoder checkpoint.
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
        logger.info("Soprano decoder: %d decoder tensors strictly loaded; AR weights belong to stage 0", len(state))
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
        """Delta contract: each call returns only newly committed samples.

        Full-payload deployments still yield one complete waveform. Async
        deployments use bounded windows keyed by the generation runner's
        request IDs, with terminal flush and on_requests_finished abort cleanup.
        """
        infos = runtime_additional_information or []
        if not infos:
            raise ValueError("Soprano decoder has no request payload")
        audio = []
        truncated = []
        synthesis_errors = []
        device = self.decoder.head.out.weight.device
        for row, info in enumerate(infos):
            extra = info.get("additional_information", {})
            error = extra.get("soprano_error")
            hidden = info.get("hidden_states", {}).get("output")
            if extra.get("soprano_stream"):
                if request_ids is None or len(request_ids) != len(infos):
                    raise ValueError("Soprano stream needs one generation request ID per batch row")
                request_id = request_ids[row]
                finished = bool(info.get("meta", {}).get("stream_finished", False))
                failed = request_id in self._failed_streams
                committed = False
                try:
                    if failed:
                        audio.append(torch.empty(0, device=device, dtype=torch.float32))
                    else:
                        if error or hidden is None:
                            raise ValueError(error or "Missing Soprano streaming hidden payload")
                        chunk_size = int(extra["soprano_chunk_size"])
                        state = self._streams.setdefault(request_id, DecoderWindow(chunk_size))
                        if chunk_size != state.chunk_size:
                            raise ValueError("Soprano chunk_size changed within a synthesis unit")
                        audio.append(
                            state.push(self.decoder, hidden.to(device=device, dtype=torch.float32), finished=finished)
                        )
                        committed = True
                        logger.debug(
                            "Soprano audio delta req=%s samples=%d total_hidden=%d "
                            "context_rows=%d finished=%s monotonic_ns=%d",
                            request_id,
                            audio[-1].numel(),
                            state.total,
                            0 if state.hidden is None else state.hidden.shape[0],
                            finished,
                            perf_counter_ns(),
                        )
                except (ValueError, KeyError) as exc:
                    logger.error("Soprano stream req=%s: %s", request_id, exc)
                    audio.append(torch.empty(0, device=device, dtype=torch.float32))
                    failed = True
                finally:
                    if not committed:
                        # Late chunks must not restart a failed synthesis unit.
                        # The runner's finish/abort callback owns this tombstone.
                        self._failed_streams.add(request_id)
                    if finished or not committed:
                        self._streams.pop(request_id, None)
                        logger.debug(
                            "Soprano decoder release req=%s finished=%s remaining=%d",
                            request_id,
                            finished,
                            len(self._streams),
                        )
                synthesis_errors.append(torch.tensor([failed]))
                truncated.append(torch.tensor([bool(extra.get("soprano_truncated", False))]))
                continue
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
