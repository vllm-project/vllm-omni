"""
CUDA Graph wrapper for AcousticTransformer in VoxtralTTS.

Currently it support FlowMatching implementation only.

Captures the AcousticTransformer forward pass (semantic logit +
n-step Euler ODE with CFG) into CUDA graphs for fixed batch sizes,
eliminating kernel launch overhead on every decode step.
"""

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
    AudioSpecialTokens,
)

logger = init_logger(__name__)


class CUDAGraphAcousticTransformerWrapper:
    """
    CUDA Graph wrapper for the acoustic transformer.

    Replaces the vLLM Sampler path (which has Python-level branching and dynamic
    tensor allocation) with a CUDA-graph-compatible path using torch.argmax
    (equivalent to top_k=1 greedy sampling) and pre-allocated buffers.
    """

    DEFAULT_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(
        self,
        model,
        capture_sizes: list[int] | None = None,
    ):
        self.model = model
        self.acoustic_transformer = model.acoustic_transformer

        self.capture_sizes = sorted(capture_sizes or self.DEFAULT_CAPTURE_SIZES)

        # Pre-compute constants from the acoustic transformer
        self.empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)
        self.end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self.semantic_mask_start = len(AudioSpecialTokens) + self.acoustic_transformer.model_args.semantic_codebook_size
        self.n_acoustic_codebook = self.acoustic_transformer.model_args.n_acoustic_codebook
        self.acoustic_embeddings_levels = self.acoustic_transformer.acoustic_embeddings_levels

        self.cfg_alpha = 1.2
        self.n_steps = 8

        # Graph storage
        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_noise: dict[int, torch.Tensor] = {}
        self.static_fake_eos: dict[int, torch.Tensor] = {}
        self.static_audio_codes: dict[int, torch.Tensor] = {}

        self.enabled = False
        self._warmed_up = False

    def _warmup_and_capture(self, device: torch.device, dtype: torch.dtype, hidden_dim: int):
        """Perform eager warmup and CUDA graph capture for all bucket sizes."""
        if self._warmed_up:
            logger.warning("CUDAGraphAcousticTransformerWrapper already warmed up, skipping")
            return

        logger.info(
            "CUDAGraphAcousticTransformerWrapper: starting warmup and capture for sizes %s",
            self.capture_sizes,
        )

        # Pre-create persistent buffers
        self.timesteps = torch.linspace(0, 1, self.n_steps, device=device, dtype=dtype)
        self.fake_eos_one = torch.tensor(1.0, dtype=dtype, device=device)
        self.fake_eos_zero = torch.tensor(0.0, dtype=dtype, device=device)

        # Phase 1: Eager warmup for ALL capture sizes
        for size in self.capture_sizes:
            dummy = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
            with torch.no_grad():
                self._forward_cudagraph_compatible(dummy)

        torch.cuda.synchronize(device)

        # Phase 2: Capture graphs
        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype, hidden_dim)
                logger.info("  Captured CUDA Graph for batch_size=%d", size)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDA Graph for batch_size=%d",
                    size,
                    exc_info=True,
                )

        self.enabled = True
        self._warmed_up = True
        logger.info(
            "CUDAGraphAcousticTransformerWrapper warmup complete. Captured %d/%d graphs.",
            len(self.graphs),
            len(self.capture_sizes),
        )

    def _forward_cudagraph_compatible(self, hidden_states: torch.Tensor, noise: torch.Tensor | None = None):
        """
        The actual computation captured by the CUDA graph.

        This replaces the full compute_mm_logits -> acoustic_transformer.forward() path
        with a graph-compatible version:
        - Uses argmax instead of vLLM Sampler (equivalent for top_k=1)
        - Uses pre-created timesteps buffer instead of torch.linspace
        - Uses pre-created scalar tensors for torch.where
        - Calls _predict_velocity directly
        - Uses a pre-allocated noise buffer to avoid baking random state
          into the CUDA graph
        """
        at = self.acoustic_transformer
        B = hidden_states.shape[0]

        # --- Semantic logits via linear projection ---
        semantic_logit = at.semantic_codebook_output(hidden_states).float()
        semantic_logit[:, self.empty_audio_token_id] = -float("inf")
        semantic_logit[:, self.semantic_mask_start :] = -float("inf")

        # argmax == top_k=1 greedy sampling
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)  # (B, 1)

        # --- Flow matching: Euler ODE ---
        should_decode = semantic_code.squeeze(1) != self.end_audio_token_id

        if noise is not None:
            x = noise
        else:
            x = torch.randn(B, self.n_acoustic_codebook, device=hidden_states.device, dtype=hidden_states.dtype)

        # Pre-compute zero hidden states for unconditional CFG branch
        hidden_states_zero = torch.zeros_like(hidden_states)

        timesteps = self.timesteps
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            # Batch conditional + unconditional velocity in a single forward pass
            t_emb = at.time_embedding(t.view(-1, 1).repeat(B, 1)).to(hidden_states.dtype)
            x_batched = torch.cat([x, x], dim=0)  # (2B, C)
            llm_batched = torch.cat([hidden_states, hidden_states_zero], dim=0)  # (2B, D)
            t_emb_batched = t_emb.repeat(2, 1)  # (2B, D)

            v_all = at._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]

            # CFG combination
            v_t = self.cfg_alpha * v_t + (1 - self.cfg_alpha) * uncond_v_t

            x = x + v_t * dt

        # --- Quantize ---
        sampled = torch.clamp(x, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self.empty_audio_token_id
        acoustic_codes = output_codes + len(AudioSpecialTokens)

        # --- Combine semantic + acoustic ---
        audio_codes = torch.cat([semantic_code, acoustic_codes], dim=1)  # (B, 1 + n_acoustic)

        # --- Compute fake_eos ---
        fake_eos = torch.where(
            audio_codes[:, 0] == self.end_audio_token_id,
            self.fake_eos_one,
            self.fake_eos_zero,
        )

        return fake_eos, audio_codes

    def _capture_graph_for_size(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
    ):
        """Capture a CUDA graph for a specific batch size."""
        static_input = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
        static_noise = torch.randn(size, self.n_acoustic_codebook, device=device, dtype=dtype)

        # Stabilizing eager run
        with torch.no_grad():
            _ = self._forward_cudagraph_compatible(static_input, noise=static_noise)

        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_fake_eos, static_audio_codes = self._forward_cudagraph_compatible(
                    static_input, noise=static_noise
                )

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_noise[size] = static_noise
        self.static_fake_eos[size] = static_fake_eos
        self.static_audio_codes[size] = static_audio_codes

    def _get_padded_size(self, actual_size: int) -> int | None:
        """Round up to the nearest captured bucket size."""
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def __call__(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]] | None]:
        """
        Drop-in replacement for model.compute_mm_logits().

        Falls back to eager execution if:
        - CUDA graph is not enabled/warmed up
        - Batch size exceeds largest captured size
        """
        actual_size = hidden_states.shape[0]

        if not self.enabled or not self._warmed_up:
            return self.model.compute_mm_logits(hidden_states)

        padded_size = self._get_padded_size(actual_size)
        if padded_size is None or padded_size not in self.graphs:
            return self.model.compute_mm_logits(hidden_states)

        # Zero static input, then copy actual data
        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:actual_size] = hidden_states

        # Fill noise buffer with fresh random values before replay so the
        # flow-matching ODE starts from different initial noise each time.
        self.static_noise[padded_size].normal_()

        # Replay captured graph
        self.graphs[padded_size].replay()

        # Clone and slice outputs for actual batch size
        fake_eos = self.static_fake_eos[padded_size][:actual_size].clone()
        audio_codes = self.static_audio_codes[padded_size][:actual_size].clone()

        # Package into expected format: (fake_eos, {"audio": [list of tensors]})
        audio_list = list(torch.split(audio_codes.unsqueeze(1), 1, dim=0))
        mm_tokens = {"audio": audio_list}

        return fake_eos, mm_tokens
