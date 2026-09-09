# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test-only native input faults, installed only by the reliability server."""

import importlib.abc
import importlib.machinery
import os
import sys
import time

_MODE = os.environ.get("VLLM_OMNI_TEST_NATIVE_INPUT_FAULT", "")
_TARGETS = {
    "vllm_omni.core.sched.omni_ar_scheduler": ("preempt",),
    "vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0": ("empty_encoder",),
    "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni": ("trace", "embedding_oracle"),
}


def _patch(module):
    if _MODE == "preempt":
        cls = module.OmniARScheduler
        original = cls.schedule

        def schedule(self, *args, **kwargs):
            if not getattr(self, "_native_test_preempted", False):
                for request in list(self.running):
                    duplex = getattr(request, "model_intermediate_buffer", {}).get("duplex", {})
                    if (
                        getattr(request, "streaming_prompt_continuous", False)
                        and duplex.get("seq", 0) >= 2
                        and request.num_computed_tokens > 0
                    ):
                        before = request.num_computed_tokens
                        in_flight = request.num_in_flight_tokens
                        self.running.remove(request)
                        # Async lookahead need not reach an in-flight-zero
                        # scheduling boundary. Exercise vLLM's real stale-output
                        # and deferred-block-free fences instead of waiting for
                        # a sync-only opportunity and silently injecting nothing.
                        self._preempt_request(request, time.monotonic(), drop_stale_output=True)
                        self._native_test_preempted = True
                        print(
                            f"[reliability][native-kv-preempt] applied request={request.request_id} "
                            f"computed={before}->0 preemptions={request.num_preemptions} in_flight={in_flight}",
                            flush=True,
                        )
                        break
            return original(self, *args, **kwargs)

        cls.schedule = schedule
    elif _MODE == "empty_encoder":
        cls = module.MiniCPMO45Stage0DuplexRuntime
        original = cls._stage_audio_embeddings

        def embeddings(self, *args, state=None, **kwargs):
            if str(getattr(state, "session_id", "")).startswith("duplex-input-failure-") and not getattr(
                self, "_native_test_failed", False
            ):
                self._native_test_failed = True
                print("[reliability][native-empty-encoder] applied", flush=True)
                return None
            return original(self, *args, state=state, **kwargs)

        cls._stage_audio_embeddings = embeddings
    elif _MODE == "embedding_oracle":
        import torch

        cls = module.MiniCPMO45OmniForConditionalGeneration
        original = cls.preprocess

        def preprocess(self, input_ids, *args, **kwargs):
            result = original(self, input_ids, *args, **kwargs)
            duplex = kwargs.get("duplex")
            offset = kwargs.get("duplex_token_offset", 0)
            prompt_len = kwargs.get("duplex_prompt_len", 0)
            if (
                self.model_stage == "llm"
                and isinstance(duplex, dict)
                and duplex.get("data_plane")
                and offset < prompt_len
            ):
                history = self._minicpmo45_duplex_input_histories[kwargs["request_id"]]
                expected_ids = input_ids.clone()
                expected_embeds = self.get_input_embeddings(expected_ids).to(dtype=result[1].dtype).clone()
                history.overlay(offset=offset, input_ids=expected_ids, embeddings=expected_embeds)
                # Diagnostic only: deliberately synchronizes each real GPU
                # prefill. Never use this run as a performance baseline.
                torch.testing.assert_close(result[0], expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(result[1], expected_embeds, rtol=0, atol=0)
                print(
                    f"[reliability][native-embedding-oracle] exact request={kwargs['request_id']} "
                    f"seq={duplex.get('seq')} offset={offset} tokens={len(input_ids)}",
                    flush=True,
                )
            return result

        cls.preprocess = preprocess
    elif _MODE == "trace":
        cls = module.MiniCPMO45OmniForConditionalGeneration
        original = cls.preprocess

        def preprocess(self, input_ids, *args, **kwargs):
            result = original(self, input_ids, *args, **kwargs)
            duplex = kwargs.get("duplex")
            if isinstance(duplex, dict) and duplex.get("data_plane"):
                identity = (kwargs.get("request_id"), duplex.get("epoch"), duplex.get("seq"))
                if getattr(self, "_native_trace_identity", None) != identity:
                    self._native_trace_identity = identity
                    state = self._duplex_data_plane_helper().sessions.get(
                        (duplex.get("session_id"), duplex.get("incarnation", 0))
                    )
                    prepared = getattr(state, "prepared_inputs_embeds", None)
                    print(
                        f"[reliability][native-trace] seq={duplex.get('seq')} "
                        f"final={duplex.get('final')} prompt={kwargs.get('duplex_prompt_len')} "
                        f"offset={kwargs.get('duplex_token_offset')} scheduled={len(input_ids)} "
                        f"prepared={len(prepared) if prepared is not None else None}",
                        flush=True,
                    )
            return result

        cls.preprocess = preprocess


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if _MODE not in _TARGETS.get(fullname, ()) or not _MODE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        sys.meta_path.remove(self)
        spec.loader = _Loader(spec.loader)
        return spec


if any(_MODE in modes for modes in _TARGETS.values()):
    sys.meta_path.insert(0, _Finder())
