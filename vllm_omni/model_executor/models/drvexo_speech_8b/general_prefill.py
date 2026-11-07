# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-owned dynamic candidate score on the actual shared Thinker."""

import hashlib
import json
from pathlib import Path

import torch


class GeneralClassifier:
    def __init__(self, directory, specification):
        # In the export these imports resolve inside the complete model bundle.
        from .checkpoint import UnifiedCheckpoint
        from .general_decision_head import GeneralDecisionHead

        self.specification = specification
        root = Path(directory).resolve()
        binding_path = root / specification["binding"]
        if hashlib.sha256(binding_path.read_bytes()).hexdigest() != specification["binding_sha256"]:
            raise ValueError("General-head binding hash mismatch")
        binding = json.loads(binding_path.read_text())
        if binding["status"] not in ("trained_native_features_pending_http", "validated_native_http"):
            raise ValueError("General head is not trained on the bound native features")
        for relative, expected in binding["bound_artifacts"].items():
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("General-head binding artifact escapes bundle")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"General-head artifact changed: {relative}")
        config = json.loads((root / "config.json").read_text())
        config.pop("drvexo_general_decision", None)
        if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != binding["config_sha256"]:
            raise ValueError("General-head backbone config changed")
        checkpoint = UnifiedCheckpoint(root)
        metadata = checkpoint.components["components"]["general_decision_head"]["metadata"]
        # Use CPU FP32 explicitly; named precision never silently quantizes this head.
        with torch.device("cpu"):
            self.head = GeneralDecisionHead(4096, int(metadata["projection_size"])).float()
        self.head.load_state_dict(checkpoint.state_dict("general_decision_head"), strict=True)
        self.head.eval().requires_grad_(False)
        self.binding = binding

    def validate_scheduler(self, config):
        if (
            not config.model_config.enforce_eager
            or config.cache_config.enable_prefix_caching
            or config.scheduler_config.max_num_seqs != 1
            or config.scheduler_config.max_num_batched_tokens != 2048
            or config.model_config.dtype != torch.bfloat16
        ):
            raise ValueError(
                "General-head native runtime requires eager BF16 compute, "
                "single sequence, no prefix cache and exactly 2048 prefill budget"
            )

    @torch.inference_mode()
    def score(self, hidden, task):
        if task not in ("choice", "noul", "score"):
            raise ValueError("Unknown decision task")
        if hidden.shape != (4096,) or not torch.isfinite(hidden).all():
            raise ValueError("Invalid final-prefill representation")
        value = hidden.detach().to(device="cpu", dtype=torch.float32)[None]
        value = self.head.projection(self.head.input_norm(value))
        return self.head.heads[task](value).reshape(1).clone()


class GeneralPrefill:
    def __init__(self, classifier):
        self.classifier = classifier

    def process(self, hidden, buffers, spans, offsets):
        if len(buffers) != len(spans) or len(buffers) != len(offsets):
            raise ValueError("General decision scheduler metadata mismatch")
        scores, statuses = [], []
        for info, (start, end), offset in zip(buffers, spans, offsets, strict=True):
            score = torch.empty(0, dtype=torch.float32)
            status = 0
            metadata = info.get("drvexo_general_decision")
            if metadata is not None:
                try:
                    length, task = metadata["prompt_length"], metadata["task"]
                    if (
                        type(length) is not int
                        or not 0 < length < 2048
                        or task not in ("choice", "noul", "score")
                        or type(offset) is not int
                        or offset < 0
                        or not 0 <= start <= end <= len(hidden)
                    ):
                        raise ValueError("Invalid general-prefill boundaries")
                    contract = (length, task)
                    prior = info.get("_drvexo_general_contract")
                    if prior is not None and prior != contract:
                        raise ValueError("General-prefill contract changed")
                    info["_drvexo_general_contract"] = contract
                    if not info.get("_drvexo_general_done") and offset <= length - 1 < offset + end - start:
                        score = self.classifier.score(hidden[start + length - 1 - offset], task)
                        info["_drvexo_general_done"] = True
                        status = 1
                except (ValueError, KeyError, TypeError):
                    status = -1
            scores.append(score)
            statuses.append(torch.tensor([status], dtype=torch.int64))
        return {"drvexo_general_scores": scores, "drvexo_general_status": statuses}
