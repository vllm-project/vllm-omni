# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from MoonshotAI/Kimi-Audio (MIT), revision
# 349251e1d8f4f98d58fda59246381faecd7392e0, kimia_infer/models/detokenizer/.
# See vllm_omni/model_executor/models/kimi_audio/NOTICE for the upstream license.

import logging
import time

import torch
import yaml

from .flow_matching.model import DiTPrefix
from .flow_matching.ode_wrapper import StreamingODEWrapperForPrefix
from .flow_matching.scheduler import StreamingFlowMatchingScheduler

logger = logging.getLogger(__name__)


class StreamingSemanticFMWrapper:
    def __init__(
        self,
        speech_model: DiTPrefix,
        max_kv_cache_tokens=900,
        use_cfg=False,
        normalize_mel=False,
        mel_mean=None,
        mel_std=None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if use_cfg:
            raise ValueError("CFG is not supported by Kimi's streaming acoustic decoder")
        self.dtype = torch.bfloat16
        self.speech_model = speech_model.to(device).to(self.dtype)
        self.speech_model = self.speech_model.eval()
        self.device = device
        self.normalize_mel = normalize_mel
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        self.condition_cache = {"previous_seqlen": 0}

        self.scheduler = StreamingFlowMatchingScheduler()
        self.ode_wrapper = StreamingODEWrapperForPrefix(net=self.speech_model)

        self.max_kv_cache_tokens = max_kv_cache_tokens

    @torch.inference_mode()
    def infer_chunk(
        self,
        xt_chunk,
        semantic_tokens_chunk,
        start_position_id,
        cache=None,
        look_ahead_tokens=0,
        ode_steps=15,
        verbose=False,
        ode_solver="neural_ode_euler",
    ):
        """
        semantic_tokens: [T_1], torch.LongTensor
        xt: [T_2, 80], torch.Tensor, DO NOT normalize it outside
        ode_steps: int, number of ode steps, default 15
        verbose: bool, default False
        ode_solver: str, ode solver, expected in ("neural_ode_euler", "naive_euler"), default "neural_ode_euler"
        """
        self.scheduler.set_timesteps(ode_steps)

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(self.device)
        xt_chunk = xt_chunk.unsqueeze(0).to(self.device).to(self.dtype)

        t_span = torch.linspace(0, 1, self.scheduler.timesteps)

        cache_ret = self.ode_wrapper.set_conditions(
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        if verbose:
            t_start = time.time()
        if ode_solver == "neural_ode_euler":
            x_t = self.scheduler.sample_by_neuralode(self.ode_wrapper, time_steps=t_span, xt=xt_chunk, verbose=False)
        elif ode_solver == "naive_euler":
            x_t = self.scheduler.sample(
                ode_wrapper=self.ode_wrapper,
                time_steps=t_span,
                xt=xt_chunk,
                verbose=False,
            )
        else:
            raise NotImplementedError("ode_solver should be in ('neural_ode_euler', 'naive_euler')")

        if look_ahead_tokens > 0:
            semantic_tokens_left = semantic_tokens_chunk.view(-1)[-look_ahead_tokens:]
            cache["semantic_token"] = semantic_tokens_left
            x_t_ret = x_t[:, :-look_ahead_tokens, :]
        else:
            x_t_ret = x_t

        if look_ahead_tokens > 0:
            self.condition_cache = self.ode_wrapper.set_conditions(
                x_cond=semantic_tokens_chunk[:, :-look_ahead_tokens],
                start_position_id=start_position_id,
                cache=self.condition_cache,
            )
            self.ode_wrapper(torch.Tensor([0.999]).to(x_t_ret.device), x_t_ret)
        else:
            self.condition_cache = cache_ret

        if verbose:
            t_end = time.time()
            logger.info(f"[ODE Chunk] Time cost: {t_end - t_start}")

        if self.normalize_mel:
            x_t_ret = x_t_ret * self.mel_std + self.mel_mean
        return x_t_ret.squeeze(0)

    def clear_all_states(self):
        self.start_position_id = 0
        self.condition_cache = {"previous_seqlen": 0}
        self.ode_wrapper.clear_all_states()

    def update_incremental_state(self):
        self.ode_wrapper.update_incremental_state(
            max_kv_cache_tokens=self.max_kv_cache_tokens,
            condition_cache=self.condition_cache,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_config,
        ckpt_path,
        device,
        max_kv_cache_tokens=900,
        use_cfg=False,
    ):
        # open yaml file
        with open(model_config) as f:
            config = yaml.safe_load(f)
        model_config = config["model"]["dit"]
        dit = DiTPrefix(
            input_size=model_config["input_size"],
            semantic_vocab_size=model_config["semantic_vocab_size"] + 1,
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            ffn_type=model_config.get("ffn_type", "conv1d_conv1d"),
            ffn_gated_glu=model_config.get("ffn_gated_glu", True),
            ffn_act_layer=model_config.get("ffn_act_layer", "gelu"),
            ffn_conv_kernel_size=model_config.get("ffn_conv_kernel_size", 5),
            use_rope=model_config.get("use_rope", False),
            rope_params=model_config.get(
                "rope_params",
                {
                    "max_position_embeddings": 4096,
                    "rope_base": 10000,
                    "rope_interpolation_factor": 1,
                },
            ),
            position_embedding_type=model_config["position_embedding_type"],
            max_seq_len=model_config["max_seq_len"],
            output_size=model_config["input_size"],
            prompt_cfg_dropout=0,
        )

        # load state_dict
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        speech_model_params = {k.replace("speech_model.", ""): v for k, v in state_dict.items() if "speech_model" in k}
        dit.load_state_dict(speech_model_params, strict=True)
        logger.info(f">>> Loaded checkpoint from {ckpt_path}")

        return cls(
            speech_model=dit,
            device=device,
            normalize_mel=config["normalize_mel"],
            mel_mean=config["mel_mean"],
            mel_std=config["mel_std"],
            max_kv_cache_tokens=max_kv_cache_tokens,
            use_cfg=use_cfg,
        )
