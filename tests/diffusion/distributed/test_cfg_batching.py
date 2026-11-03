# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CFG batching in CFGParallelMixin: both branches as one predict_noise call on one CFG rank."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Pipeline(CFGParallelMixin):
    supports_cfg_batching = True

    def __init__(self, *, batched: bool = True, cache_backend: str = "none", tuple_out: bool = False):
        self.od_config = SimpleNamespace(enable_cfg_batching=batched, cache_backend=cache_backend)
        self.transformer = torch.nn.Module()
        self.calls: list[tuple[int, str | None]] = []
        self.tuple_out = tuple_out

    def predict_noise(self, hidden_states, timestep, encoder_hidden_states, added_cond_kwargs, return_dict=False):
        assert return_dict is False
        self.calls.append((hidden_states.shape[0], getattr(self.transformer, "cfg_branch", None)))
        # Row-wise function of every argument, so a wrong merge or split changes the result.
        ctx = encoder_hidden_states.mean(dim=(1, 2)) + added_cond_kwargs["text_embeds"].sum(-1)
        out = hidden_states * timestep.view(-1, 1, 1, 1) + ctx.view(-1, 1, 1, 1)
        return (out, out.mean(dim=(1, 2, 3))) if self.tuple_out else out


def _kwargs(batch: int, seq: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(batch, 4, 8, 8, generator=g),
        "timestep": torch.full((batch,), 0.5),
        "encoder_hidden_states": torch.randn(batch, seq, 16, generator=g),
        "added_cond_kwargs": {"text_embeds": torch.randn(batch, 8, generator=g), "time_ids": torch.ones(batch, 6)},
        "return_dict": False,
    }


def _run(pipe: _Pipeline, pos: dict, neg: dict):
    return pipe.predict_noise_maybe_with_cfg(True, 7.0, pos, neg, cfg_normalize=False)


@pytest.mark.parametrize("batch", [1, 2])
def test_batched_matches_sequential_in_one_call(batch):
    pos, neg = _kwargs(batch, 77, 0), _kwargs(batch, 77, 1)
    neg["hidden_states"] = pos["hidden_states"]  # latents are shared between branches
    batched, sequential = _Pipeline(), _Pipeline(batched=False)
    torch.testing.assert_close(_run(batched, pos, neg), _run(sequential, pos, neg))
    # The merged call covers both branches, so it carries no cfg_branch stamp.
    assert batched.calls == [(2 * batch, None)]
    assert sequential.calls == [(batch, "positive"), (batch, "negative")]


def test_tuple_outputs_are_split_per_element():
    pos, neg = _kwargs(1, 77, 0), _kwargs(1, 77, 1)
    batched, sequential = _Pipeline(tuple_out=True), _Pipeline(batched=False, tuple_out=True)
    for a, b in zip(_run(batched, pos, neg), _run(sequential, pos, neg)):
        torch.testing.assert_close(a, b)
    assert len(batched.calls) == 1


def test_output_slice_applies_after_split():
    pos, neg = _kwargs(1, 77, 0), _kwargs(1, 77, 1)
    out = {}
    for name, pipe in (("batched", _Pipeline()), ("sequential", _Pipeline(batched=False))):
        out[name] = pipe.predict_noise_maybe_with_cfg(True, 7.0, pos, neg, cfg_normalize=False, output_slice=2)
    torch.testing.assert_close(out["batched"], out["sequential"])
    assert out["batched"].shape[1] == 2


def test_mismatched_branches_fall_back_to_sequential():
    pos, neg = _kwargs(1, 77, 0), _kwargs(1, 64, 1)  # different prompt lengths
    pipe = _Pipeline()
    torch.testing.assert_close(_run(pipe, pos, neg), _run(_Pipeline(batched=False), pos, neg))
    assert [n for n, _ in pipe.calls] == [1, 1]


@pytest.mark.parametrize(
    "neg_update",
    [
        {"added_cond_kwargs": {"flag": True}},  # differing scalar
        {"added_cond_kwargs": {"mask": None}},  # None vs tensor
        {"added_cond_kwargs": {"obj": object()}},  # non-comparable object
    ],
    ids=["scalar", "none-vs-tensor", "object"],
)
def test_unmergeable_arguments_fall_back(neg_update):
    pos, neg = _kwargs(1, 77, 0), _kwargs(1, 77, 1)
    pos["added_cond_kwargs"].update({"flag": False, "mask": torch.ones(1, 77), "obj": object()})
    neg["added_cond_kwargs"].update({"flag": False, "mask": torch.ones(1, 77), "obj": pos["added_cond_kwargs"]["obj"]})
    neg["added_cond_kwargs"].update(neg_update["added_cond_kwargs"])
    pipe = _Pipeline()

    def predict_noise(**kwargs):
        pipe.calls.append((kwargs["hidden_states"].shape[0], None))
        return kwargs["hidden_states"]

    setattr(pipe, "predict_noise", predict_noise)
    _run(pipe, pos, neg)
    assert [n for n, _ in pipe.calls] == [1, 1]


def _ids_pipeline(shared: frozenset[str] = frozenset()) -> tuple[_Pipeline, list[dict]]:
    pipe = _Pipeline()
    pipe.cfg_batching_shared_kwargs = shared
    seen: list[dict] = []

    def predict_noise(**kwargs):
        seen.append(kwargs)
        return kwargs["hidden_states"] * 2

    setattr(pipe, "predict_noise", predict_noise)
    return pipe, seen


def _ids_kwargs(ids: torch.Tensor, batch: int = 1) -> dict:
    return {"hidden_states": torch.randn(batch, 4), "timestep": torch.tensor(0.5), "txt_ids": ids, "scale": 1.0}


def test_unbatched_tensors_are_passed_through_not_concatenated():
    """Flux-style [L, 3] ids have no batch dim: never concatenated, shared only when safe."""
    ids, timestep = torch.zeros(77, 3), torch.tensor(0.5)
    pos, neg = _ids_kwargs(ids), _ids_kwargs(ids)
    pos["timestep"] = neg["timestep"] = timestep

    pipe, seen = _ids_pipeline()
    _run(pipe, pos, neg)
    assert len(seen) == 1
    assert seen[0]["hidden_states"].shape[0] == 2
    # The same object in both branches is passed through as is.
    assert seen[0]["txt_ids"] is ids and seen[0]["timestep"] is timestep and seen[0]["scale"] == 1.0

    # Equal but distinct per-branch ids are not compared (that would sync the device) ...
    pipe, seen = _ids_pipeline()
    _run(pipe, _ids_kwargs(torch.zeros(77, 3)), _ids_kwargs(torch.zeros(77, 3)))
    assert len(seen) == 2

    # ... unless the pipeline declares them shared.
    pipe, seen = _ids_pipeline(frozenset({"txt_ids", "timestep"}))
    pos = _ids_kwargs(torch.zeros(77, 3))
    _run(pipe, pos, _ids_kwargs(torch.zeros(77, 3)))
    assert len(seen) == 1 and seen[0]["txt_ids"] is pos["txt_ids"]


@pytest.mark.parametrize("batch", [1, 2])
def test_batch_kwarg_decides_what_is_batched(batch):
    """A [B, ...] tensor is concatenated, one with another dim 0 must be shared."""
    pos, neg = _ids_kwargs(torch.zeros(77, 3), batch), _ids_kwargs(torch.zeros(77, 3), batch)
    pos["pooled"], neg["pooled"] = torch.randn(batch, 8), torch.randn(batch, 8)
    pipe, seen = _ids_pipeline(frozenset({"txt_ids", "timestep"}))
    _run(pipe, pos, neg)
    torch.testing.assert_close(seen[0]["pooled"], torch.cat([pos["pooled"], neg["pooled"]]))
    assert seen[0]["txt_ids"].shape == (77, 3)

    # Without a batch kwarg there is no batch size to go by.
    pipe, seen = _ids_pipeline()
    pipe.cfg_batching_batch_kwarg = "latents"
    _run(pipe, pos, neg)
    assert len(seen) == 2


def test_unbatched_output_raises():
    pos, neg = _kwargs(1, 77, 0), _kwargs(1, 77, 1)
    pipe = _Pipeline()
    setattr(pipe, "predict_noise", lambda **kwargs: kwargs["hidden_states"].sum(0, keepdim=True))
    with pytest.raises(RuntimeError, match="supports_cfg_batching"):
        _run(pipe, pos, neg)


def _pp_pipeline() -> _Pipeline:
    pipe = _Pipeline()
    pipe.od_config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    return pipe


@pytest.mark.parametrize(
    "pipe",
    [
        _Pipeline(batched=False),
        _Pipeline(cache_backend="tea_cache"),
        type("NotOptedIn", (_Pipeline,), {"supports_cfg_batching": False})(),
        _pp_pipeline(),
    ],
    ids=["option-off", "cache-backend", "not-opted-in", "pipeline-parallel"],
)
def test_cfg_batching_off(pipe):
    _run(pipe, _kwargs(1, 77, 0), _kwargs(1, 77, 1))
    assert [n for n, _ in pipe.calls] == [1, 1]


def test_no_cfg_runs_positive_only():
    pipe = _Pipeline()
    pipe.predict_noise_maybe_with_cfg(False, 1.0, _kwargs(1, 77, 0), None)
    assert [n for n, _ in pipe.calls] == [1]
