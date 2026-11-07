# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU FP32 heads over the merged hybrid backbone's final normalized token.

Execution support does not establish paired quality acceptance of the merged
backbone and its unchanged stage6 head. No language-generated score is used.
"""

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.drvexo_speech_8b.general_decision_head import GeneralDecisionHead

from .checkpoint import UnifiedBF16Checkpoint

TASKS = ("choice", "noul", "score")
MAX_PROMPT_TOKENS = 4480
logger = init_logger(__name__)


class UnifiedClassifier:
    def __init__(self, checkpoint: UnifiedBF16Checkpoint):
        row = checkpoint.components.get("general_decision_head")
        metadata = row.get("metadata") if isinstance(row, dict) else None
        if (
            not isinstance(metadata, dict)
            or metadata.get("hidden_size") != 2560
            or metadata.get("projection_size") != 0
        ):
            raise ValueError("Expected the trained native 2560-dimensional unprojected heads")
        with torch.device("cpu"):
            self.head = GeneralDecisionHead(hidden_size=2560, projection_size=0).float()
        state = dict(checkpoint.iter_component("general_decision_head"))
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Decision heads require finite source FP32 tensors")
        self.head.load_state_dict(state, strict=True)
        self.head.eval().requires_grad_(False)

    @staticmethod
    def validate_scheduler(config):
        # Decision scoring is always bound in this complete package. Keep the
        # validated deployment contract for plain chat too, but identify every
        # incompatible setting instead of reporting an undifferentiated error.
        checks = (
            (config.model_config.enforce_eager, "enforce_eager=True"),
            (not config.cache_config.enable_prefix_caching, "enable_prefix_caching=False"),
            (config.scheduler_config.max_num_seqs == 1, "max_num_seqs=1"),
            (config.scheduler_config.enable_chunked_prefill, "enable_chunked_prefill=True"),
            (config.scheduler_config.max_num_batched_tokens == 2048, "max_num_batched_tokens=2048"),
            (config.model_config.max_model_len >= MAX_PROMPT_TOKENS, f"max_model_len>={MAX_PROMPT_TOKENS}"),
            (config.model_config.dtype == torch.bfloat16, "dtype=bfloat16"),
        )
        failures = [setting for valid, setting in checks if not valid]
        if failures:
            raise ValueError("4B complete-package decision support requires " + ", ".join(failures))

    @torch.inference_mode()
    def score(self, hidden: torch.Tensor, task: str) -> torch.Tensor:
        if task not in TASKS or hidden.shape != (2560,) or not torch.isfinite(hidden).all():
            raise ValueError("Invalid native final-prefill representation/task")
        value = hidden.detach().to(device="cpu", dtype=torch.float32)[None]
        result = self.head.heads[task](self.head.input_norm(value)).reshape(1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite native candidate score")
        return result


class UnifiedPrefill:
    def __init__(self, classifier):
        self.classifier = classifier

    def process(self, hidden, buffers, spans, offsets):
        if hidden.ndim != 2 or hidden.shape[1] != 2560:
            raise ValueError("Expected native 2560-dimensional final-normalized states")
        if len(buffers) != len(spans) or len(buffers) != len(offsets):
            raise ValueError("Decision scheduler metadata mismatch")
        scores, statuses = [], []
        for info, span, offset in zip(buffers, spans, offsets, strict=True):
            score, status = torch.empty(0, dtype=torch.float32), 0
            metadata = info.get("drvexo_general_decision")
            if metadata is None:
                info.pop("_drvexo_unified_prefill", None)
            else:
                try:
                    start, end = span
                    length, task = metadata["prompt_length"], metadata["task"]
                    if (
                        type(length) is not int
                        or not 0 < length <= MAX_PROMPT_TOKENS
                        or task not in TASKS
                        or type(offset) is not int
                        or offset < 0
                        or type(start) is not int
                        or type(end) is not int
                        or not 0 <= start <= end <= len(hidden)
                    ):
                        raise ValueError("Invalid decision prefill contract")
                    contract = (length, task)
                    prior = info.setdefault("_drvexo_unified_prefill", {"contract": contract, "done": False})
                    if prior["contract"] != contract:
                        raise ValueError("Decision contract changed within a live request")
                    if not prior["done"] and offset <= length - 1 < offset + end - start:
                        try:
                            score = self.classifier.score(hidden[start + length - 1 - offset], task)
                        except ValueError as exc:
                            logger.warning("Rejecting invalid DRVEXO decision score: %s", exc)
                            status = -1
                        else:
                            prior["done"] = True
                            status = 1
                except (ValueError, KeyError, TypeError) as exc:
                    logger.debug("Rejecting invalid DRVEXO decision prefill: %s", exc)
                    status = -1
            scores.append(score)
            statuses.append(torch.tensor([status], dtype=torch.int64))
        return {"drvexo_general_scores": scores, "drvexo_general_status": statuses}
