# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for final pixel assembly; device/full-model gates are separate."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

source = Path(__file__).resolve().parents[3] / "vllm_omni"
package = ModuleType("sp8_pixels_cpu")
package.__path__ = [str(source / "diffusion/layers")]
sys.modules[package.__name__] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pixels_module = load("sp8_pixels_cpu.lingbot_pixel_output", source / "diffusion/layers/lingbot_pixel_output.py")
PixelOutputPolicy = pixels_module.PixelOutputPolicy
current_pixel_output_policy = pixels_module.current_pixel_output_policy
exact_planar_pixels = pixels_module.exact_planar_pixels
gather_final_pixels = pixels_module.gather_final_pixels
pixel_output_context = pixels_module.pixel_output_context
validate_mode = pixels_module.validate_mode
WanStreamingDecoder = load(
    "sp8_streaming_cpu", source / "experimental/ar_diffusion/streaming_decode.py"
).WanStreamingDecoder


class PixelOutputTests(unittest.TestCase):
    def test_allgather_preserves_source_order_and_trim(self):
        for dtype in (torch.bfloat16, torch.float32):
            shards = [torch.linspace(-2, 2, 36).reshape(1, 3, 2, 3, 2).to(dtype) + r / 32 for r in range(8)]
            snapshots = [x.clone() for x in shards]
            reference = exact_planar_pixels(torch.cat(shards, dim=4))[:, :, :, :14]

            def allgather(target, source, group):
                for out, shard in zip(target, shards):
                    out.copy_(exact_planar_pixels(shard))

            for rank in range(8):
                policy = PixelOutputPolicy("root_uint8_allgather", object(), rank, 8)
                with patch("torch.distributed.all_gather", allgather):
                    out = gather_final_pixels(shards[rank], policy=policy, split_dim="width", expected_extent=14)
                self.assertTrue(torch.equal(out, reference) if rank == 0 else out.numel() == 0)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(shards, snapshots)))

    def test_root_batch_receive_order_and_wait(self):
        shards = [torch.full((1, 3, 2, 3, 2), r / 8, dtype=torch.bfloat16) for r in range(8)]
        ordered = []

        class Work:
            def wait(self):
                ordered.append("wait")

        def op(fn, tensor, peer, group):
            ordered.append(peer)
            tensor.copy_(exact_planar_pixels(shards[peer - 10]))
            return object()

        policy = PixelOutputPolicy("root_uint8_p2p", object(), 0, 8)
        with (
            patch("torch.distributed.get_global_rank", lambda group, rank: rank + 10),
            patch("torch.distributed.P2POp", op),
            patch("torch.distributed.batch_isend_irecv", lambda ops: [Work() for _ in ops]),
        ):
            out = gather_final_pixels(shards[0], policy=policy, split_dim="width", expected_extent=16)
        self.assertEqual(ordered, list(range(11, 18)) + ["wait"] * 7)
        self.assertTrue(torch.equal(out, exact_planar_pixels(torch.cat(shards, dim=4))))

    def test_exhaustive_bf16_pixel_rounding(self):
        source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 1, 256, 256)
        source = source.expand(1, 3, 1, 256, 256).clone()
        expected = source.clamp(-1, 1)
        expected = ((expected[0] / 2 + 0.5).clamp(0, 1).float() * 255).round().to(torch.uint8)
        self.assertTrue(torch.equal(expected, exact_planar_pixels(source)))

    def test_policy_context_resets_after_failure(self):
        policy = PixelOutputPolicy("root_float", object(), 0, 8)
        with self.assertRaises(RuntimeError):
            with pixel_output_context(policy):
                self.assertIs(current_pixel_output_policy(), policy)
                raise RuntimeError("controlled")
        self.assertIsNone(current_pixel_output_policy())
        with self.assertRaises(ValueError):
            validate_mode("root")

    def test_nonprimary_keeps_temporal_counters_and_cache(self):
        class VAE:
            config = SimpleNamespace(patch_size=None)
            _cached_conv_counts = {"decoder": 1}

            def post_quant_conv(self, value):
                return value

            def decoder(self, value, *, feat_cache, feat_idx, first_chunk):
                feat_cache[0] = value.clone()
                feat_idx[0] += 1
                return value.new_empty(0)

        decoder = WanStreamingDecoder(VAE())
        state = decoder.new_decode_state("s")
        for i in range(2):
            output = decoder.decode_chunk(torch.ones(1, 4, 3, 2, 2), state, produce_output=False)
            self.assertEqual(output.numel(), 0)
            self.assertEqual((state.frames_decoded, state.chunks_decoded), (3 * (i + 1), i + 1))
            self.assertIsNotNone(state.feat_map[0])

    def test_planar_temporal_concat_bypasses_float_clamp(self):
        class VAE:
            config = SimpleNamespace(patch_size=None)
            _cached_conv_counts = {"decoder": 0}

            def post_quant_conv(self, value):
                return value

            def decoder(self, value, **kwargs):
                return torch.full((3, 1 if kwargs["first_chunk"] else 4, 2, 2), 200, dtype=torch.uint8)

        decoder = WanStreamingDecoder(VAE())
        state = decoder.new_decode_state("s")
        output = decoder.decode_chunk(torch.ones(1, 4, 3, 2, 2), state, output_format="planar_uint8")
        self.assertEqual(output.shape, (3, 9, 2, 2))
        self.assertTrue(torch.all(output == 200))


if __name__ == "__main__":
    unittest.main()
