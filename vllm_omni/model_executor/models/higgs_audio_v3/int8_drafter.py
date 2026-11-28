# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from vllm import _custom_ops as ops
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_empty,
    marlin_make_workspace_new,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_scales,
)
from vllm.scalar_type import scalar_types
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from vllm_omni.platforms import current_omni_platform


class Int8DraftLinear:
    def __init__(self, weight: torch.Tensor):
        self.n, self.k = weight.shape
        group_size = 128
        groups = weight.detach().float().view(self.n, -1, group_size)
        scale = (groups.abs().amax(-1) / 127).clamp_min(1e-12).to(torch.bfloat16)
        codes = (groups / scale.float().unsqueeze(-1)).round().clamp_(-127, 127)
        words = (codes.view(self.n, self.k).T.contiguous().to(torch.int32) + 128).view(-1, 4, self.n)
        packed = words[:, 0] | (words[:, 1] << 8) | (words[:, 2] << 16) | (words[:, 3] << 24)
        n, k = marlin_padded_nk(self.n, self.k, group_size)
        packed = marlin_pad_qweight(packed, self.n, self.k, n, k)
        self.weight = ops.gptq_marlin_repack(packed, size_k=k, size_n=n, num_bits=8)
        scale = marlin_pad_scales(scale.T.contiguous(), self.n, self.k, n, k, group_size)
        self.scale = marlin_permute_scales(scale, size_k=k, size_n=n, group_size=group_size)
        self.zero = marlin_make_empty(weight.device)
        self.workspace = marlin_make_workspace_new(weight.device, max_blocks_per_sm=4)

    def apply(self, layer, inputs: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return apply_gptq_marlin_linear(
            inputs, self.weight, self.scale, self.zero, self.workspace, scalar_types.uint8b128, self.n, self.k, bias
        )


class HiggsInt8Drafter:
    def __init__(self, model, vllm_config):
        self.model = model
        self.vllm_config = vllm_config
        self.linears = [
            (layer, layer.quant_method, Int8DraftLinear(layer.weight))
            for block in model.model.layers
            for layer in block.modules()
            if isinstance(layer, LinearBase)
        ]
        self.graphs = {}

    def forward(self, batch_rows, starts, positions, feedback, max_context):
        num_reqs = len(batch_rows)
        tables = self.block_tables
        slots = tables.slot_mappings[0].index_select(0, starts)
        if 8 < num_reqs < 16 and self.vllm_config.scheduler_config.max_num_seqs >= 16:
            padding = 16 - num_reqs
            batch_rows = torch.cat((batch_rows, batch_rows[-1:].expand(padding)))
            positions = torch.cat((positions, positions[-1:].expand(padding)))
            feedback = torch.cat((feedback, feedback[-1:].expand(padding, -1)))
            # Padded rows reuse valid attention inputs without writing to KV.
            slots = torch.cat((slots, slots.new_full((padding,), -1)))
        n = len(batch_rows)
        block_table = tables.input_block_tables[0].index_select(0, batch_rows)
        key = n, max_context
        entry = self.graphs.get(key)
        if entry is None:
            graph = None
            static_positions = torch.empty_like(positions)
            static_feedback = torch.empty_like(feedback)
            meta = FlashAttentionMetadata(
                n,
                1,
                torch.arange(n + 1, dtype=torch.int32, device=positions.device),
                max_context,
                torch.empty(n, dtype=torch.int32, device=positions.device),
                torch.empty_like(block_table),
                torch.empty_like(slots),
                num_decode_reqs=n,
                num_decode_tokens=n,
            )
        else:
            graph, output, static_positions, static_feedback, meta = entry
        static_positions.copy_(positions)
        static_feedback.copy_(feedback)
        meta.block_table.copy_(block_table)
        meta.slot_mapping.copy_(slots)
        meta.seq_lens.copy_(positions + 1)
        if graph is None:
            metadata = dict.fromkeys(self.layer_names, meta)
            mappings = dict.fromkeys(self.layer_names, meta.slot_mapping)
            output = None

            def capture():
                nonlocal output
                with set_forward_context(
                    metadata, self.vllm_config, num_tokens=n, slot_mapping=mappings, skip_compiled=True
                ):
                    hidden = self.model._run_qwen3_layers_eager(static_positions, static_feedback.clone())
                    output = self.model.modality_head.generate(hidden)

            for layer, _, draft in self.linears:
                layer.quant_method = draft
            try:
                capture()
                graph = current_omni_platform.capture_graph(capture)
            finally:
                for layer, target, _ in self.linears:
                    layer.quant_method = target
            assert graph is not None
            # Replay references every metadata tensor, including query_start_loc.
            self.graphs[key] = graph, output, static_positions, static_feedback, meta
        graph.replay()
        return output[:num_reqs].clone()
