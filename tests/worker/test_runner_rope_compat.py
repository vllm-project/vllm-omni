# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config
from vllm.config import CacheConfig, CompilationConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class PositionModel(torch.nn.Module):
    supports_mrope = True
    supports_xdrope = True

    def __init__(self, dimensions):
        super().__init__()
        self.positions = torch.arange(dimensions * 4).reshape(dimensions, 4)
        self.calls = []

    # `_filter_mrope_kwargs_for_model` passes only the keyword arguments this
    # signature names, so `image_grid_thw` has to be declared to observe what
    # `_init_mrope_positions` extracted from the request.
    def get_mrope_input_positions(self, input_tokens, mm_features, image_grid_thw=None):
        self.calls.append(("mrope", input_tokens, mm_features, image_grid_thw))
        return self.positions, 7

    def get_xdrope_input_positions(self, input_tokens, mm_features):
        self.calls.append(("xdrope", input_tokens, mm_features))
        return self.positions


@pytest.fixture
def cpu_runner(monkeypatch, tmp_path):
    """Keep the actual parent constructor, configs, batch and position buffers.

    Only device-property queries and pinned host allocations are replaced for
    CPU execution. Model weights are unnecessary for request/position handling.
    """
    # The runner pins every host buffer through this one flag, so turning it
    # off covers the constructor without routing the test's own allocations
    # through a wrapper.
    import vllm.utils.torch_utils as torch_utils
    import vllm.v1.utils as v1_utils
    import vllm.v1.worker.gpu_model_runner as upstream

    import vllm_omni.worker.gpu_model_runner as omni

    monkeypatch.setattr(upstream, "PIN_MEMORY", False, raising=False)
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False, raising=False)
    monkeypatch.setattr(v1_utils, "PIN_MEMORY", False, raising=False)
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True, ranks=[0])
    monkeypatch.setattr(upstream, "get_pp_group", lambda: group)
    monkeypatch.setattr(omni, "get_pp_group", lambda: group)
    monkeypatch.setattr(OmniGPUModelRunner, "_init_device_properties", lambda self: None)
    hf = Qwen2Config(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        architectures=["Qwen2ForCausalLM"],
    )
    hf.save_pretrained(tmp_path)

    def create(runner_class, layout):
        model_config = ModelConfig(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=32,
            enforce_eager=True,
            dtype="float32",
        )
        # Exercise the installed vLLM's real configuration properties and
        # constructor. In newer versions the legacy section selects 4D M-RoPE.
        if layout == "mrope":
            model_config.hf_config.rope_parameters = {"mrope_section": [1, 1, 1]}
        elif layout == "xdrope":
            model_config.hf_config.xdrope_section = [1, 1, 1, 1]
        config = VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(block_size=16),
            scheduler_config=SchedulerConfig(
                max_model_len=32,
                max_num_seqs=2,
                max_num_batched_tokens=32,
                async_scheduling=False,
                is_encoder_decoder=False,
            ),
            compilation_config=CompilationConfig(mode=0, cudagraph_mode="NONE"),
        )
        runner = runner_class(config, torch.device("cpu"))
        runner.model = PositionModel(4 if layout == "xdrope" else 3)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        runner._init_model_kwargs = dict
        return runner

    return create


def scheduled(*, new=(), count=4, finished=()):
    return SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"request": count} if count else {},
        total_num_scheduled_tokens=count,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


@pytest.mark.parametrize("runner_class", [GPUARModelRunner, GPUGenerationModelRunner])
@pytest.mark.parametrize(
    ("layout", "input_kind"),
    # Only the M-RoPE path varies with the input: `_init_xdrope_positions`
    # passes `mm_features` through without extracting a grid from it.
    [("plain", "text"), ("mrope", "text"), ("mrope", "image"), ("mrope", "cached"), ("xdrope", "text")],
)
def test_construct_schedule_and_consume_position_layout(cpu_runner, runner_class, layout, input_kind):
    runner = cpu_runner(runner_class, layout)
    mm_features = []
    if input_kind != "text":
        item = MultiModalKwargsItem(
            image_grid_thw=MultiModalFieldElem(data=torch.tensor([1, 2, 2]), field=MultiModalSharedField(batch_size=1))
        )
        mm_features.append(
            MultiModalFeatureSpec(
                data=item if input_kind == "image" else None,
                modality="image",
                identifier="image",
                mm_position=PlaceholderRange(offset=0, length=4),
            )
        )
    new = NewRequestData(
        req_id="request",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=mm_features,
        sampling_params=SamplingParams(temperature=0, max_tokens=4),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )
    output = scheduled(new=[new])
    # Request admission reads the flag in `_update_states`. Startup profiling
    # reads it earlier still, in `_dummy_run`, which this CPU fixture does not
    # reach, and the two NPU read sites are not exercised here either.
    runner._update_states(output)
    assert runner.input_batch.req_ids == ["request"]
    if layout == "plain":
        assert runner.uses_xdrope_dim == 0
        assert not runner.uses_mrope
        assert runner.model.calls == []
        runner.positions[:4] = torch.arange(4)
        expected = torch.arange(4)
        buffer = None
    else:
        if runner.uses_mrope:
            assert layout == "mrope"
            assert runner.uses_xdrope_dim == 0
            assert not hasattr(runner, "xdrope_positions")
            runner._calc_mrope_positions(output)
            buffer = runner.mrope_positions
            route = "mrope"
        else:
            assert layout == "xdrope"
            assert runner.uses_xdrope_dim == 4
            assert not hasattr(runner, "mrope_positions")
            runner._calc_xdrope_positions(output)
            buffer = runner.xdrope_positions
            route = "xdrope"
        if route == "mrope":
            # The grid the runner extracted from the request reaches the model:
            # present for a request carrying image data, empty otherwise.
            grid = [[1, 2, 2]] if input_kind == "image" else []
            assert runner.model.calls == [(route, [1, 2, 3, 4], mm_features, grid)]
        else:
            assert runner.model.calls == [(route, [1, 2, 3, 4], mm_features)]
        # The runner allocates the buffer from the version's own notion of the
        # position width, which is what vllm-project/vllm#56078 changes.
        # 0.29.0 allocates three M-RoPE rows; the version that folds XD-RoPE in
        # allocates `mrope_num_dims`. Either way the width comes from the runner,
        # not from this fixture's model.
        expected_rows = getattr(runner, "mrope_num_dims", 3) if route == "mrope" else runner.uses_xdrope_dim
        assert buffer.cpu.shape == (expected_rows, runner.max_num_tokens + 1)
        expected = runner.model.positions
        torch.testing.assert_close(buffer.cpu[:, :4], expected)
        buffer.gpu.copy_(buffer.cpu)
    positions = runner._preprocess(output, 4)[2]
    torch.testing.assert_close(positions, expected)

    if buffer is not None:
        # Decode uses the matching upstream position calculator and keeps all
        # position channels, including the fourth legacy XD-RoPE channel.
        runner.input_batch.num_computed_tokens_cpu[0] = 4
        decode = scheduled(count=2)
        if runner.uses_mrope:
            runner._calc_mrope_positions(decode)
            # context 4, 5 plus the delta 7 this model returned from prefill.
            next_positions = torch.tensor([4, 5]) + 7
        else:
            runner._calc_xdrope_positions(decode)
            # XD-RoPE decode carries no delta.
            next_positions = torch.tensor([4, 5])
        expected_decode = next_positions.expand(buffer.cpu.shape[0], -1)
        torch.testing.assert_close(buffer.cpu[:, :2], expected_decode)
        buffer.gpu.copy_(buffer.cpu)
        torch.testing.assert_close(runner._preprocess(decode, 2)[2], expected_decode)

    runner._update_states(scheduled(count=0, finished=["request"]))
    assert runner.requests == {}
    assert runner.input_batch.req_ids == []


def test_the_constructor_supplies_the_flag_when_the_parent_stops_setting_it(cpu_runner, monkeypatch):
    """vllm-project/vllm#56078 removes `uses_xdrope_dim` from the parent runner.

    Simulate that removal on the pinned version, so the pinned CPU lane covers
    the case the fix exists for: the constructor is the single place the six
    read sites depend on, and request admission still reaches one of them.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    parent_init = GPUModelRunner.__init__

    def without_the_attribute(self, *args, **kwargs):
        parent_init(self, *args, **kwargs)
        # A vLLM that already removed it leaves nothing to drop.
        self.__dict__.pop("uses_xdrope_dim", None)

    monkeypatch.setattr(GPUModelRunner, "__init__", without_the_attribute)

    runner = cpu_runner(GPUARModelRunner, "plain")
    new = NewRequestData(
        req_id="request",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=4),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )

    # Admission first: without the default this raises inside `_update_states`,
    # which is where a served request would hit it.
    runner._update_states(scheduled(new=[new]))

    assert runner.input_batch.req_ids == ["request"]
    assert runner.uses_xdrope_dim == 0
