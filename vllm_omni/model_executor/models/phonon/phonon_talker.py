# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Phonon AR backbone and stateless flow sampling."""

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.layers.attention import Attention

from .flow import PhononFlowHead


class PhononAttention(nn.Module):
    def __init__(self, config, *, cache_config=None, prefix=""):
        super().__init__()
        width = config["d_model"]
        self.heads = config["num_heads"]
        self.head_dim = width // self.heads
        self.in_proj = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.register_buffer(
            "frequencies",
            config["max_period"] ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
            persistent=False,
        )
        self.attention = Attention(
            self.heads,
            self.head_dim,
            self.head_dim**-0.5,
            cache_config=cache_config,
            prefix=prefix + ".attn",
        )

    def forward(self, inputs, positions):
        packed = self.in_proj(inputs).view(-1, 3, self.heads, self.head_dim)
        query, key, value = packed.unbind(1)
        angles = positions[:, None, None].float() * self.frequencies
        cosine, sine = angles.cos(), angles.sin()

        def rotate(tensor):
            even, odd = tensor[..., 0::2], tensor[..., 1::2]
            rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1)
            return rotated.flatten(-2).to(tensor.dtype)

        query, key = rotate(query), rotate(key)
        output = self.attention(query.flatten(1), key.flatten(1), value.flatten(1))
        return self.out_proj(output)


class PhononLayer(nn.Module):
    def __init__(self, config, *, cache_config=None, prefix=""):
        super().__init__()
        width = config["d_model"]
        self.self_attn = PhononAttention(config, cache_config=cache_config, prefix=prefix)
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.linear1 = nn.Linear(width, config["dim_feedforward"], bias=False)
        self.linear2 = nn.Linear(config["dim_feedforward"], width, bias=False)

    def forward(self, inputs, positions):
        inputs = inputs + self.self_attn(self.norm1(inputs), positions)
        return inputs + self.linear2(F.gelu(self.linear1(self.norm2(inputs))))


class PhononBackbone(nn.Module):
    def __init__(self, config, *, cache_config=None, prefix="flow_lm.transformer"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PhononLayer(config, cache_config=cache_config, prefix=f"{prefix}.layers.{index}.self_attn")
                for index in range(config["num_layers"])
            ]
        )

    def forward(self, inputs, positions):
        for layer in self.layers:
            inputs = layer(inputs, positions)
        return inputs


class PhononFlowLM(nn.Module):
    def __init__(self, config, *, cache_config=None):
        super().__init__()
        cfg = config["flow_lm"]
        width, latent = cfg["d_model"], cfg["ldim"]
        self.bos_emb = nn.Parameter(torch.zeros(latent))
        self.emb_mean = nn.Parameter(torch.zeros(latent), requires_grad=False)
        self.emb_std = nn.Parameter(torch.ones(latent), requires_grad=False)
        self.input_linear = nn.Linear(latent, width, bias=False)
        self.transformer = PhononBackbone(cfg, cache_config=cache_config)
        self.out_norm = nn.LayerNorm(width, eps=1e-5)
        self.out_eos = nn.Linear(width, 1)
        self.flow_net = PhononFlowHead(latent, cfg["flow_dim"], width, cfg["flow_depth"])
        self.text_embedding = nn.Embedding(cfg["n_bins"] + 1, width)
        self.speaker_projection = nn.Linear(width, width, bias=False)
        self.speaker_embedding = nn.Embedding(32, 16)
        self.num_speakers_projection = nn.Linear(16, width, bias=False)
        self.duration_projection = nn.Linear(128, width, bias=False)
        self.config = config

    def conditioning(self, num_speakers=1, duration_delta=0.0):
        device = self.bos_emb.device
        frequencies = 10000.0 ** (-torch.arange(64, device=device).float() / 63)
        angles = duration_delta * 1000 * frequencies
        duration = torch.cat((angles.cos(), angles.sin())).to(self.bos_emb.dtype)
        speaker = self.speaker_embedding(torch.tensor(num_speakers, device=device))
        return self.num_speakers_projection(speaker) + self.duration_projection(duration)

    def sample(self, hidden, noise):
        normalized = self.out_norm(hidden)
        return self.flow_net(normalized, noise, steps=self.config["lsd_decode_steps"]), self.out_eos(normalized)

    def load_official_weights(self, weights):
        mapped = {}
        remap = {
            "condition_provider.conditioners.transcript_in_segment.embed.": "text_embedding.",
            "condition_provider.conditioners.speaker_wavs.output_proj.": "speaker_projection.",
            "condition_provider.conditioners.num_speakers.embed.": "speaker_embedding.",
            "condition_provider.conditioners.num_speakers.output_proj.": "num_speakers_projection.",
            "condition_provider.conditioners.duration_delta.output_proj.": "duration_projection.",
        }
        for name, tensor in weights.items():
            if not name.startswith("flow_lm."):
                continue
            target = name.removeprefix("flow_lm.")
            if target.startswith("flow.w_s_t.") or target.endswith(".learnt_padding"):
                continue
            for source, destination in remap.items():
                target = target.replace(source, destination)
            mapped[target] = tensor
        # Backend buffers are initialized by vLLM, outside the official checkpoint.
        for index, layer in enumerate(self.transformer.layers):
            attention = layer.self_attn.attention
            for name, buffer in attention.named_buffers():
                mapped[f"transformer.layers.{index}.self_attn.attention.{name}"] = buffer
        self.load_state_dict(mapped, strict=True)


from vllm.compilation.decorators import support_torch_compile  # noqa: E402


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0})
class PhononTalkerForConditionalGeneration(nn.Module):
    has_preprocess = True
    has_postprocess = True
    have_multimodal_outputs = True
    gpu_resident_buffer_keys = set()

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Phonon initial implementation requires tensor_parallel_size=1")
        self.config = vllm_config.model_config.hf_config
        self.eos_token_id = self.config.eos_token_id
        self.continue_token_id = self.config.audio_continue_token_id
        self.phonon_config = self.config.to_dict()
        self.flow_lm = PhononFlowLM(self.phonon_config, cache_config=vllm_config.cache_config)
        from vllm.utils.torch_utils import set_default_torch_dtype

        from .mimi_decoder import MimiStreamingDecoder

        with set_default_torch_dtype(torch.float32):
            self.mimi = MimiStreamingDecoder(self.phonon_config["mimi"], vllm_config.scheduler_config.max_num_seqs)
        self._continue = None

    def get_input_embeddings(self, input_ids):
        return self.flow_lm.text_embedding(input_ids.long())

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids)

    def preprocess(self, input_ids, input_embeds, **kwargs):
        return input_ids, input_embeds, {}

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        inputs = self.get_input_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        return self.flow_lm.transformer(inputs, positions)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        logits = hidden_states.new_full((hidden_states.shape[0], self.config.vocab_size), float("-inf"))
        if self._continue is None:
            logits[:, self.continue_token_id] = 0
        else:
            logits[:, self.continue_token_id] = torch.where(self._continue, 0.0, float("-inf"))
            logits[:, self.eos_token_id] = torch.where(self._continue, float("-inf"), 0.0)
        return logits

    def create_mrv2_model_state(self, vllm_config, encoder_cache, device):
        from .model_state import PhononModelState

        return PhononModelState(vllm_config, self, encoder_cache, device)

    def load_weights(self, weights):
        weights = dict(weights)
        self.flow_lm.load_official_weights(weights)
        self.mimi.load_official_weights(weights)
        return set(dict(self.named_parameters()))
