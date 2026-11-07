# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real scheduler/runner batching and posterior RNG, with weight-free networks."""

import multiprocessing
import uuid
from contextlib import closing
from dataclasses import dataclass, replace
from functools import partial

import pytest
import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import BatchFeature, PretrainedConfig
from vllm import SamplingParams
from vllm.config import SchedulerConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.cache.shm import ShmObjectStoreReceiverCache, ShmObjectStoreSenderCache
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerWrapperBase

from tests.helpers.mark import hardware_marks
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.inputs.mm_rng import MMRNGContext, contextualize_mm_features
from vllm_omni.model_executor.mm_request_rng import MM_RNG_ITEM_KEY, RandomDrawSpec, RequestRNGProvider
from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3 as hunyuan
from vllm_omni.model_executor.models.hunyuan_image3.autoencoder_kl_3d import DiagonalGaussianDistribution
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model]


@dataclass
class TinyVAEConfig:
    shift_factor: float = 0.25
    scaling_factor: float = 0.5


class TinyVAE:
    config = TinyVAEConfig()
    ffactor_spatial = 1

    def __init__(self, device, dtype):
        self.device, self.dtype = device, dtype

    def encode(self, images):
        mean = images.unsqueeze(2)
        return AutoencoderKLOutput(
            latent_dist=DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1))
        )


def runner_case(device, dtype, *values):
    return pytest.param(
        (device, dtype),
        *values,
        id="-".join(map(str, (device, dtype, *values))),
        marks=pytest.mark.cpu if device == "cpu" else hardware_marks(res={"cuda": "L4"}),
    )


@pytest.fixture
def runner(request, monkeypatch, mocker):
    device, dtype = getattr(request, "param", ("cpu", torch.float32))
    if device == "cuda" and not torch.get_device_module(device).is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device if device == "cpu" else "cuda:0")
    model = hunyuan.HunyuanImage3ForConditionalGeneration.__new__(hunyuan.HunyuanImage3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = PretrainedConfig(vae_downsample_factor=(1, 1), patch_size=1, vae={"latent_channels": 3})
    model.vae = TinyVAE(device, dtype)
    model.use_data_parallel = False
    model.vision_model = lambda pixels, **kwargs: pixels.to(dtype=dtype)
    model.vision_aligner = torch.nn.Identity()
    model.time_embed = torch.nn.Identity()
    model.patch_embed = lambda latent, time: (latent.flatten(2).transpose(1, 2), None, None)
    model._mrope_joint_img_sep_token_id = 0
    model.model = mocker.Mock()
    model.model.embed_input_ids = lambda ids: torch.zeros((len(ids), 3), device=ids.device, dtype=dtype)

    runner = object.__new__(OmniGPUModelRunner)
    runner.model, runner.device = model, device
    runner.get_model = lambda: model
    runner._init_mm_request_rng()
    runner.requests, runner.encoder_cache = {}, {}
    runner.model_intermediate_buffer, runner.num_prompt_logprobs = {}, {}
    runner.lora_config = runner.observability_config = runner.speculative_config = None
    runner.encoder_cudagraph_manager = None
    runner.is_multimodal_pruning_enabled = runner.requires_sequential_video_encoding = False
    runner.is_pooling_model = runner.uses_mrope = runner.use_async_spec_decode = False
    runner.use_async_scheduling = False
    runner.late_interaction_runner = mocker.Mock()
    runner.input_batch = mocker.Mock(spec=InputBatch)
    runner.input_batch.req_id_to_index = {}
    runner.input_batch.remove_request.side_effect = lambda req: runner.input_batch.req_id_to_index.pop(req, None)
    runner.input_batch.add_request.side_effect = lambda req: runner.input_batch.req_id_to_index.setdefault(
        req.req_id, 0
    )
    runner._may_reorder_batch = lambda output: None
    runner.maybe_save_ec_to_connector = lambda *args: None
    monkeypatch.setattr("vllm_omni.worker.gpu_model_runner.get_pp_group", lambda: mocker.Mock(is_last_rank=True))
    return runner


def images():
    return [torch.full((3, 2, 2), 0.25), torch.full((3, 2, 3), 0.75), torch.full((3, 3, 3), -0.5)]


def _make_request(mocker, req_id, images, seed=42):
    features = BatchFeature(
        {
            "vit_pixel_values": torch.stack([im[:, :1, :1].reshape(1, 3) for im in images]),
            "vit_pixel_attention_mask": torch.ones((len(images), 1), dtype=torch.bool),
            "vit_spatial_shapes": torch.ones((len(images), 2), dtype=torch.long),
            "vae_pixel_values": torch.cat([im.flatten() for im in images]),
            "vae_pixel_size": torch.tensor([im.numel() for im in images]),
            "vae_token_grid_hw": torch.tensor([im.shape[1:] for im in images]),
        }
    )
    processor = hunyuan.HunyuanImage3MultiModalProcessor.__new__(hunyuan.HunyuanImage3MultiModalProcessor)
    processor.info = mocker.Mock(spec=hunyuan.HunyuanImage3ProcessingInfo)
    processor.info.get_image_processor.return_value = lambda *args, **kwargs: features
    kwargs = {} if seed is None else {"vae_generator_seed": seed}
    output = processor._call_hf_processor("", {"images": images}, kwargs, {})
    fields = processor._get_mm_fields_config(output, kwargs)
    elements = {key: field.build_elems(key, output[key]) for key, field in fields.items()}
    from vllm.multimodal.inputs import MultiModalKwargsItem

    mm_features, offset = [], 1
    for i, im in enumerate(images):
        # Synthetic content identities; hashing behavior is tested separately.
        identity = str((im.shape, im.flatten()[0].item(), seed))
        length = im.shape[1] * im.shape[2] + 2
        mm_features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({key: elems[i] for key, elems in elements.items()}),
                modality="image",
                identifier=identity,
                mm_hash=identity,
                mm_position=PlaceholderRange(offset=offset, length=length),
            )
        )
        offset += length
    context = (
        None if seed is None else MMRNGContext("hunyuan-ar-vae-v1", seed, ("vae_generator_seed", "vae_token_grid_hw"))
    )
    if context:
        mm_features = contextualize_mm_features(mm_features, context)
    req = Request(req_id, [1] * (offset + 1), SamplingParams(seed=999), None, mm_features=mm_features)
    req.mm_rng_context = context
    return req


@pytest.fixture
def make_request(mocker):
    return partial(_make_request, mocker)


def schedule(inputs=None, *, new=(), finished=()):
    return SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs=inputs or {},
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


def admit(runner, *requests, finished=()):
    new = [OmniNewRequestData.from_base(NewRequestData.from_request(req, ([],)), req) for req in requests]
    runner._update_states(schedule(new=new, finished=finished))


def reference(runner, images, seed):
    vae = runner.model.vae
    generator = torch.Generator(device=vae.device).manual_seed(seed)
    outputs = []
    for im in images:
        mean = im.to(device=vae.device, dtype=vae.dtype)[None, :, None]
        latent = mean + randn_tensor(mean.shape, generator=generator, device=vae.device, dtype=vae.dtype)
        latent.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
        outputs.append(latent.squeeze(2).flatten(2).transpose(1, 2)[0])
    return outputs, generator.get_state()


def assert_latents(outputs, expected):
    assert len(outputs) == len(expected)
    for output, latent in zip(outputs, expected):
        assert torch.equal(output[: len(latent)], latent)


@pytest.mark.parametrize(
    ("runner", "seed_b"),
    [runner_case("cpu", torch.float32, seed) for seed in (42, 123, 0)]
    + [runner_case("cuda", dtype, 123) for dtype in (torch.float32, torch.float16, torch.bfloat16)],
    indirect=["runner"],
)
def test_mixed_requests_reordering_and_recomputation(runner, seed_b, make_request):
    if runner.device.type == "cuda":
        runner.model.vae.device = torch.device("cuda")
    refs = images()
    a, b = make_request("A", refs), make_request("B", refs[::-1], seed_b)
    admit(runner, a, b)
    token_states = {key: state.generator.get_state() for key, state in runner.requests.items()}
    expected_a, state_a = reference(runner, refs, 42)
    expected_b, state_b = reference(runner, refs[::-1], seed_b)
    for order in ({"A": [0, 1, 2], "B": [0, 1, 2]}, {"B": [2, 0, 1], "A": [1, 2, 0]}):
        outputs = runner._execute_mm_encoder(schedule(order))
        expected = [({"A": expected_a, "B": expected_b}[req])[i] for req, indices in order.items() for i in indices]
        assert_latents(outputs, expected)
    provider = runner._mm_rng_provider
    for req_id, state in (("A", state_a), ("B", state_b)):
        assert torch.equal(provider.streams[provider.handles[req_id]].checkpoints[3], state)
        assert torch.equal(runner.requests[req_id].generator.get_state(), token_states[req_id])
        assert all(MM_RNG_ITEM_KEY not in f.data for f in runner.requests[req_id].mm_features)


@pytest.mark.cpu
def test_rebuilt_and_reordered_batch_refs_preserve_request_indices(runner, make_request, mocker):
    refs = images()
    a = make_request("A", [refs[0], refs[0], refs[1]])
    b = make_request("B", refs[:1])
    missing = make_request("missing", refs[:1], None)
    missing.mm_features[0] = replace(missing.mm_features[0], data=None)
    admit(runner, a, missing, b)
    assert a.mm_features[0].identifier == b.mm_features[0].identifier
    upstream_batch = GPUModelRunner._batch_mm_inputs_from_scheduler

    def rebuilt_batch(instance, scheduled):
        hashes, items, positions = upstream_batch(instance, scheduled)
        assert len(items) == 4  # The upstream path omits the missing payload.
        return hashes[::-1], items[::-1], [(req_id, replace(position)) for req_id, position in positions[::-1]]

    mocker.patch.object(GPUModelRunner, "_batch_mm_inputs_from_scheduler", new=rebuilt_batch)
    scheduled = schedule({"A": [0, 1, 2], "missing": [0], "B": [0]})
    hashes, items, positions = runner._batch_mm_inputs_from_scheduler(scheduled)
    expected_order = [("B", 0), ("A", 2), ("A", 1), ("A", 0)]
    provider = runner._mm_rng_provider
    for (req_id, index), mm_hash, (_, item), (owner, position) in zip(
        expected_order, hashes, items, positions, strict=True
    ):
        feature = runner.requests[req_id].mm_features[index]
        assert owner == req_id and mm_hash == feature.identifier
        assert position is not feature.mm_position
        assert item[MM_RNG_ITEM_KEY].data.tolist() == [provider.handles[req_id], index]
    expected_a, _ = reference(runner, [refs[0], refs[0], refs[1]], 42)
    expected_b, _ = reference(runner, refs[:1], 42)
    assert_latents(runner._execute_mm_encoder(scheduled), [expected_b[0], *expected_a[::-1]])


@pytest.mark.cpu
def test_random_draw_spec_resolves_device_aliases_once(mocker):
    module = mocker.Mock()
    module.current_device.return_value = 1
    get_module = mocker.patch("torch.get_device_module", return_value=module)
    cpu = RandomDrawSpec((1,), torch.float32, torch.device("cpu"))
    assert replace(cpu, device=torch.device("cpu:0")) == cpu
    get_module.assert_not_called()
    implicit = replace(cpu, device=torch.device("cuda"))
    assert implicit.device == torch.device("cuda:1")
    assert implicit == replace(cpu, device=torch.device("cuda:1"))
    module.current_device.assert_called_once()
    module.current_device.return_value = 0
    assert implicit.device == torch.device("cuda:1")
    assert replace(cpu, device=torch.device("cuda")) == replace(cpu, device=torch.device("cuda:0"))
    assert implicit != replace(cpu, device=torch.device("cuda:0"))


@pytest.mark.parametrize(
    "runner",
    [runner_case("cpu", torch.float32)]
    + [runner_case("cuda", dtype) for dtype in (torch.float32, torch.float16, torch.bfloat16)],
    indirect=True,
)
def test_missing_predecessor_replays_rng_without_pixels(runner, make_request):
    refs = images()
    req = make_request("A", refs)
    req.num_computed_tokens = req.mm_features[1].mm_position.offset
    admit(runner, req)
    assert "vae_pixel_values" not in runner.requests["A"].mm_features[0].data
    expected, state = reference(runner, refs, 42)
    output = runner._execute_mm_encoder(schedule({"A": [2]}))
    assert_latents(output, [expected[2]])
    provider = runner._mm_rng_provider
    assert torch.equal(provider.streams[provider.handles["A"]].checkpoints[3], state)
    assert_latents(runner._execute_mm_encoder(schedule({"A": [1]})), [expected[1]])


def scheduler(cache):
    instance = object.__new__(Scheduler)
    instance.encoder_cache_manager = cache
    instance.is_encoder_decoder = False
    instance.scheduler_config = SchedulerConfig(max_model_len=128, is_multimodal_model=True, is_encoder_decoder=False)
    instance.ec_connector = None
    return instance


@pytest.mark.cpu
def test_real_scheduler_cache_subset_and_duplicate_images(runner, make_request):
    refs = images()
    cache = EncoderCacheManager(128)
    sched = scheduler(cache)
    warm = make_request("warm", refs[:1])
    admit(runner, warm)
    runner._execute_mm_encoder(schedule({"warm": [0]}))
    cache.allocate(warm, 0)
    req = make_request("A", [refs[0], refs[0], refs[1]])
    indices, _, _, _, duplicates = sched._try_schedule_encoder_inputs(req, 0, req.num_prompt_tokens, 128)
    assert indices == [1, 2]
    assert duplicates == []  # Same image at two stream positions has distinct identity.
    admit(runner, req)
    expected, _ = reference(runner, [refs[0], refs[0], refs[1]], 42)
    assert_latents(runner._execute_mm_encoder(schedule({"A": indices})), expected[1:])
    assert not torch.equal(expected[0], expected[1])


@pytest.mark.cpu
def test_pause_finish_and_request_id_reuse(runner, make_request):
    req = make_request("A", images())
    admit(runner, req)
    runner._execute_mm_encoder(schedule({"A": [0]}))
    provider = runner._mm_rng_provider
    old_handle = provider.handles["A"]
    runner._update_states(schedule())  # Unscheduled/preempted, no forward or tokens.
    assert provider.handles["A"] == old_handle
    expected, _ = reference(runner, images(), 42)
    assert_latents(runner._execute_mm_encoder(schedule({"A": [1]})), [expected[1]])
    # Abort followed by ID reuse in the same output must clear before registration.
    admit(runner, make_request("A", images(), 0), finished=["A"])
    assert old_handle not in provider.streams
    expected, _ = reference(runner, images(), 0)
    assert_latents(runner._execute_mm_encoder(schedule({"A": [2]})), [expected[2]])
    assert provider.handles["A"] != old_handle
    runner._update_states(schedule(finished=["A"]))
    assert not provider.streams and not provider.handles and not runner._mm_rng_contexts


@pytest.mark.cpu
def test_seedless_and_seeded_groups_keep_global_rng_behavior(runner, make_request):
    seeded, unseeded = make_request("A", images()[:1]), make_request("B", images()[:1], None)
    admit(runner, seeded, unseeded)
    torch.manual_seed(314)
    before = torch.get_rng_state()
    single = runner._execute_mm_encoder(schedule({"B": [0]}))[0]
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    mixed = runner._execute_mm_encoder(schedule({"A": [0], "B": [0]}))[1]
    assert torch.equal(single, mixed)
    assert torch.equal(torch.get_rng_state(), after)
    assert "B" not in runner._mm_rng_provider.handles


@pytest.mark.cpu
def test_missing_request_context_is_rejected(runner, make_request):
    req = make_request("A", images()[:1])
    admit(runner, req)
    runner._mm_rng_contexts.clear()
    with pytest.raises(ValueError, match="request RNG context"):
        runner._execute_mm_encoder(schedule({"A": [0]}))


@pytest.mark.cpu
def test_shm_worker_resolution_of_cached_predecessor(runner, monkeypatch, mocker, make_request):
    monkeypatch.setattr("vllm.envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", f"hunyuan-rng-{uuid.uuid4().hex}")
    mm_config = MultiModalConfig(
        mm_processor_cache_type="shm", mm_processor_cache_gb=0.01, mm_shm_cache_max_object_size_mb=1
    )
    config = mocker.Mock(spec=VllmConfig)
    config.parallel_config = mocker.Mock(world_size=1)
    config.model_config = mocker.Mock()
    config.model_config.get_multimodal_config.return_value = mm_config
    with closing(ShmObjectStoreSenderCache(config)) as sender:
        receiver = ShmObjectStoreReceiverCache(config, multiprocessing.Lock())
        with closing(receiver._shm_cache):
            req = make_request("A", images())
            handles = []
            for feature in req.mm_features:
                address, _ = sender.get_and_update_item((feature.data, []), feature.mm_hash)
                handles.append(replace(feature, data=address))
            req.mm_features = handles
            req.num_computed_tokens = handles[1].mm_position.offset
            data = OmniNewRequestData.from_base(NewRequestData.from_request(req, ([],)), req)
            assert data.mm_rng_context == req.mm_rng_context
            assert data.mm_features[0].data is handles[0].data
            assert "address" in data.mm_features[0].data
            assert OmniNewRequestData.from_request(req, ([],)).mm_rng_context == req.mm_rng_context
            output = schedule({"A": [2]}, new=[data])
            wrapper = object.__new__(WorkerWrapperBase)
            wrapper.mm_receiver_cache = receiver
            wrapper._apply_mm_cache(output)
            runner._update_states(output)
            expected, _ = reference(runner, images(), 42)
            assert_latents(runner._execute_mm_encoder(output), [expected[2]])
            # UUID-only sender reuse: no original image data is provided.
            cached, _ = sender.get_and_update_item(None, handles[0].mm_hash)
            assert "address" in cached
            restored = receiver.get_and_update_features([replace(handles[0], data=cached)])[0]
            assert torch.equal(restored.data["vae_token_grid_hw"].data, torch.tensor([2, 2]))


@pytest.mark.cpu
def test_models_without_rng_capability_keep_upstream_batching(runner, make_request):
    runner.model = torch.nn.Identity()
    runner.get_model = lambda: runner.model
    runner._mm_rng_provider = None
    runner._init_mm_request_rng()
    req = make_request("A", images()[:1], None)
    admit(runner, req)
    scheduled = schedule({"A": [0]})
    expected = GPUModelRunner._batch_mm_inputs_from_scheduler(runner, scheduled)
    actual = runner._batch_mm_inputs_from_scheduler(scheduled)
    assert runner._mm_rng_provider is None
    assert actual[0] == expected[0] == [req.mm_features[0].identifier]
    assert actual[2] == expected[2]
    assert actual[1][0][0] == expected[1][0][0] == "image"
    assert actual[1][0][1] is expected[1][0][1] is req.mm_features[0].data


@pytest.mark.cpu
def test_request_plan_and_feature_scan_are_not_repeated_per_image(runner, make_request, mocker):
    class CountingFeatures(list):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    num_images = 8
    refs = [images()[0]] * num_images
    admit(runner, make_request("A", refs))
    features = CountingFeatures(runner.requests["A"].mm_features)
    runner.requests["A"].mm_features = features
    build_plan = mocker.spy(runner.model, "get_mm_rng_specs")
    expected, _ = reference(runner, refs, 42)
    for indices in (list(range(num_images)), list(reversed(range(num_images))), [num_images - 1]):
        assert_latents(runner._execute_mm_encoder(schedule({"A": indices})), [expected[i] for i in indices])
    assert build_plan.call_count == 1
    assert features.scans == 1
    # The plan cache must be owned by the request lifetime, including ID reuse.
    admit(runner, make_request("A", refs), finished=["A"])
    runner._execute_mm_encoder(schedule({"A": [0]}))
    assert build_plan.call_count == 2


@pytest.mark.cpu
@pytest.mark.parametrize("change", ["seed_removed", "seed_added", "reference", "preceding_shape"])
def test_live_request_update_rejects_changes_to_cached_plan(runner, make_request, change):
    req = make_request("A", images(), None if change == "seed_added" else 42)
    admit(runner, req)
    runner._execute_mm_encoder(schedule({"A": [0]}))
    updated = make_request("A", images(), None if change == "seed_removed" else 42)
    if change == "reference":
        updated.mm_features[0] = replace(updated.mm_features[0], identifier="different-reference")
    elif change == "preceding_shape":
        # Even unchanged IDs cannot authorize changed predecessor draw geometry.
        elem = updated.mm_features[0].data["vae_token_grid_hw"]
        updated.mm_features[0].data["vae_token_grid_hw"] = replace(elem, data=torch.tensor([4, 4]))
    with pytest.raises(ValueError, match="Multimodal RNG"):
        admit(runner, updated)


@pytest.mark.cpu
def test_sparse_image_indices_restore_after_reverse_order_and_failure():
    spec = RandomDrawSpec((1, 3, 1, 2, 2), torch.float32, torch.device("cpu"))
    provider = RequestRNGProvider(hunyuan.HunyuanImage3ForConditionalGeneration.advance_mm_rng)
    handle = provider.register("A", 0, {8: spec, 2: spec, 5: spec})
    reference_generator = torch.Generator(device=spec.device).manual_seed(0)

    def draw(generator):
        return randn_tensor(spec.shape, generator=generator, device=spec.device, dtype=spec.dtype)

    expected = {index: draw(reference_generator) for index in (2, 5, 8)}
    advance = provider.advance

    def fail_replay(generator, draw_spec):
        advance(generator, draw_spec)
        raise RuntimeError("replay")

    provider.advance = fail_replay
    with pytest.raises(RuntimeError, match="replay"):
        with provider.sample(handle, 5, spec):
            pytest.fail("failed predecessor replay reached target sampling")
    assert len(provider.streams[handle].checkpoints) == 1
    provider.advance = advance
    # A failed first sample may retain replayed predecessors, but not its own boundary.
    with pytest.raises(RuntimeError, match="first sample"):
        with provider.sample(handle, 5, spec) as generator:
            draw(generator)
            raise RuntimeError("first sample")
    assert len(provider.streams[handle].checkpoints) == 2
    with pytest.raises(ValueError, match="specification"):
        with provider.sample(handle, 5, replace(spec, shape=(1,))):
            pytest.fail("invalid posterior accepted")
    for index in (8, 2, 5):
        with provider.sample(handle, index, spec) as generator:
            assert torch.equal(draw(generator), expected[index])
    with pytest.raises(RuntimeError, match="retry"):
        with provider.sample(handle, 2, spec) as generator:
            draw(generator)
            raise RuntimeError("retry")
    for index in (2, 8):
        with provider.sample(handle, index, spec) as generator:
            assert torch.equal(draw(generator), expected[index])
    assert torch.equal(provider.streams[handle].checkpoints[3], reference_generator.get_state())
