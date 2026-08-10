# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import urlparse

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer
from vllm.logger import init_logger
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.media.audio import load_audio

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.model_executor.models.omni_diffusion.audio_tokenizer import (
    OmniDiffusionAudioTokenizer,
)
from vllm_omni.model_executor.models.omni_diffusion.chat_template import (
    OMNI_DIFFUSION_CHAT_TEMPLATE,
    normalize_chat_template_token_ids,
)
from vllm_omni.model_executor.models.omni_diffusion.component_paths import (
    OMNI_DIFFUSION_IMAGE_TOKENIZER_REPO_ID,
    OMNI_DIFFUSION_SENSEVOICE_REPO_ID,
    resolve_omni_diffusion_component_path,
)
from vllm_omni.model_executor.models.omni_diffusion.dream_compat import (
    ensure_default_rope_init_function,
    ensure_dream_generation_config_fields,
    ensure_dream_rope_parameters,
    initialize_dream_generation_config,
    patch_legacy_dream_generation_config_validate,
    patch_remote_dream_generation_config_validate,
    repair_default_dream_rope_buffers,
)
from vllm_omni.model_executor.models.omni_diffusion.image_tokenizer import (
    OmniDiffusionImageTokenizer,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE,
    OMNI_DIFFUSION_IMAGE_START_TOKEN,
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
    set_generation_seed,
)

logger = init_logger(__name__)

_S2I_DUMMY_AUDIO_SAMPLE_RATE = 16000
_S2I_IMAGE_TOKEN_COUNT = 256


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(
            f"Expected Omni-Diffusion S2I decoded image shape to be (1, 3, H, W), got {tuple(image.shape)}."
        )

    image = image[0].detach().float().cpu().permute(1, 2, 0)
    image = (image * 255).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(image.numpy(), mode="RGB")


def _decode_audio_source(audio_source: str) -> tuple[torch.Tensor, int]:
    if audio_source.startswith("data:audio"):
        audio, sample_rate = MediaConnector().fetch_audio(audio_source)
        return torch.as_tensor(audio).contiguous(), int(sample_rate)

    parsed = urlparse(audio_source)
    if parsed.scheme == "file":
        audio_source = parsed.path
    elif parsed.scheme in {"http", "https"}:
        # Network media must be fetched by the serving layer so its configured
        # domain allowlist and timeout policy are enforced. The diffusion chat
        # path currently forwards the raw URL instead, so reject it here rather
        # than downloading it without those safeguards.
        raise ValueError(
            "Omni-Diffusion S2I currently supports data:audio URLs, file:// URLs, "
            "or local audio paths. HTTP(S) audio URLs should be decoded before reaching the pipeline."
        )

    if not os.path.exists(audio_source):
        raise FileNotFoundError(f"Omni-Diffusion S2I audio source does not exist: {audio_source}")
    audio, sample_rate = load_audio(audio_source, sr=None, mono=False)
    return torch.as_tensor(audio).contiguous(), int(sample_rate)


def _get_prompt_text(prompt: OmniPromptType) -> str:
    """Extract the text field from an S2I diffusion prompt."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Mapping):
        value = prompt.get("prompt")
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        raise TypeError(f"Expected prompt['prompt'] to be a string, got {type(value)!r}.")
    raise TypeError(f"Expected S2I prompt to be a string or mapping, got {type(prompt)!r}.")


def _get_audio_source(req: DiffusionRequestBatch) -> str:
    extra_args = req.sampling_params.extra_args or {}
    for key in ("audio_path", "audio_url"):
        audio_source = extra_args.get(key)
        if audio_source is None:
            continue
        if not isinstance(audio_source, str):
            raise TypeError(f"Expected extra_args[{key!r}] to be a string, got {type(audio_source)!r}.")
        if audio_source:
            return audio_source

    for prompt in req.prompts:
        if isinstance(prompt, Mapping):
            mm_data = prompt.get("multi_modal_data")
            if isinstance(mm_data, Mapping) and "audio" in mm_data:
                audio = mm_data["audio"]
                if isinstance(audio, str):
                    return audio
                raise TypeError(
                    "Omni-Diffusion S2I diffusion pipeline expects audio_url/audio_path "
                    "or a string-valued multi_modal_data['audio']."
                )

    raise ValueError("Omni-Diffusion S2I requires an audio input.")


@dataclass(frozen=True)
class OmniDiffusionS2IPipelineConfig:
    """Model-specific configuration for the Omni-Diffusion S2I pipeline."""

    # MagVIT-v2 checkpoint used to decode generated image codebook IDs.
    image_tokenizer_path: str
    # SenseVoiceSmall checkpoint used to encode input speech features.
    sensevoice_path: str
    # Hugging Face attention implementation used when loading DreamModel.
    attn_implementation: str
    # Whether DreamModel should suppress non-text output modalities.
    output_text_only: bool
    # Omni-Diffusion task name forwarded to DreamModel.generate().
    task: str
    # Number of mask-diffusion refinement steps.
    steps: int
    # Maximum number of tokens generated after the input prompt.
    max_new_tokens: int
    # Dream token-remasking and selection algorithm.
    alg: str
    # Classifier-free guidance scale used during S2I generation.
    cfg: float
    # Token sampling temperature.
    temperature: float
    # Nucleus-sampling probability threshold.
    top_p: float
    # Optional top-k token sampling limit.
    top_k: int | None
    # Whether DreamModel should add a beginning-of-audio token.
    add_boa_token: int
    # Penalty applied according to a token's generated position.
    max_position_penalty: float
    # Penalty applied to repeated generated tokens.
    repeat_penalty: float
    # Optional random seed used to make generation reproducible.
    seed: int | None

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any] | None) -> OmniDiffusionS2IPipelineConfig:
        """Parse the S2I ``model_config`` once during pipeline startup."""

        if model_config is None:
            model_config = {}
        elif not isinstance(model_config, Mapping):
            raise TypeError(f"Omni-Diffusion S2I model_config must be a mapping, got {type(model_config)!r}.")

        def value_or_default(key: str, default: Any) -> Any:
            value = model_config.get(key)
            return default if value is None else value

        image_tokenizer_path = resolve_omni_diffusion_component_path(
            model_config.get("image_tokenizer_path"),
            config_key="model_config.image_tokenizer_path",
            default_repo_id=OMNI_DIFFUSION_IMAGE_TOKENIZER_REPO_ID,
        )
        sensevoice_path = resolve_omni_diffusion_component_path(
            model_config.get("sensevoice_path"),
            config_key="model_config.sensevoice_path",
            default_repo_id=OMNI_DIFFUSION_SENSEVOICE_REPO_ID,
        )
        attn_implementation = str(value_or_default("attn_implementation", "flash_attention_2"))
        output_text_only = bool(value_or_default("output_text_only", False))
        task = str(value_or_default("task", "S2I"))
        steps = int(value_or_default("steps", 260))
        max_new_tokens = int(value_or_default("max_new_tokens", 260))
        alg = str(value_or_default("alg", "entropy-penalty"))
        cfg = float(value_or_default("cfg", 2.0))
        temperature = float(value_or_default("temperature", 0.0))
        top_p = float(value_or_default("top_p", 0.9))
        raw_top_k = model_config.get("top_k")
        top_k = int(raw_top_k) if raw_top_k is not None else None
        add_boa_token = int(value_or_default("add_boa_token", 0))
        max_position_penalty = float(value_or_default("max_position_penalty", 2.0))
        repeat_penalty = float(value_or_default("repeat_penalty", 1.2))
        raw_seed = model_config.get("seed")
        seed = int(raw_seed) if raw_seed is not None else None

        config = cls(
            image_tokenizer_path=image_tokenizer_path,
            sensevoice_path=sensevoice_path,
            attn_implementation=attn_implementation,
            output_text_only=output_text_only,
            task=task,
            steps=steps,
            max_new_tokens=max_new_tokens,
            alg=alg,
            cfg=cfg,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            add_boa_token=add_boa_token,
            max_position_penalty=max_position_penalty,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )
        logger.info("Omni-Diffusion S2I pipeline config initialized: %s", config)
        return config


class OmniDiffusionS2IPipeline(nn.Module):
    """Diffusion-stage wrapper for Omni-Diffusion speech-to-image.

    Omni-Diffusion generates image tokens through its DreamModel, but the
    OpenAI image-output chat route passes audio inputs to diffusion stages as
    ``sampling_params.extra_args['audio_path']``.  This wrapper keeps that
    contract local to Omni-Diffusion instead of teaching shared serving or
    scheduler code about this model's LLM-style multimodal prompt format.
    """

    support_audio_input: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 1
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset({"audio_path", "audio_url"})

    def __init__(self, *, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = str(od_config.model)
        self.pipeline_config = OmniDiffusionS2IPipelineConfig.from_model_config(od_config.model_config)

        # OmniDiffusionConfig normalizes the deploy-level dtype to torch.dtype.
        self.model_torch_dtype = od_config.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=od_config.trust_remote_code,
            chat_template=OMNI_DIFFUSION_CHAT_TEMPLATE,
        )
        self.tokenizer_base_data = OmniDiffusionTokenizerBaseData(self.tokenizer)
        hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=od_config.trust_remote_code,
        )
        ensure_dream_rope_parameters(hf_config)
        ensure_default_rope_init_function()
        patch_remote_dream_generation_config_validate(
            self.model_path,
            od_config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            config=hf_config,
            trust_remote_code=od_config.trust_remote_code,
            torch_dtype=self.model_torch_dtype,
            attn_implementation=self.pipeline_config.attn_implementation,
        ).to(self.device)
        self.model.eval()
        repair_default_dream_rope_buffers(self.model)

        initialize_dream_generation_config(
            model=self.model,
            tokenizer=self.tokenizer,
            model_path=self.model_path,
            trust_remote_code=od_config.trust_remote_code,
            top_k=self.pipeline_config.top_k,
        )

        self.image_tokenizer = OmniDiffusionImageTokenizer(
            model_path=self.pipeline_config.image_tokenizer_path,
            device=self.device,
            image_size=512,
        )
        self.audio_tokenizer = OmniDiffusionAudioTokenizer(
            sensevoice_path=self.pipeline_config.sensevoice_path,
            device=self.device,
        )

        first_parameter = next(self.model.parameters(), None)
        logger.info(
            "Omni-Diffusion S2I pipeline load config: model_path=%s device=%s "
            "loaded_parameter_device=%s torch_dtype=%s loaded_parameter_dtype=%s "
            "attn_implementation=%s trust_remote_code=%s model_class=%s.%s model_source=%s",
            self.model_path,
            self.device,
            first_parameter.device if first_parameter is not None else None,
            self.model_torch_dtype,
            first_parameter.dtype if first_parameter is not None else None,
            self.pipeline_config.attn_implementation,
            self.od_config.trust_remote_code,
            type(self.model).__module__,
            type(self.model).__name__,
            inspect.getsourcefile(type(self.model)),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        # The HF DreamModel and side tokenizers are loaded directly in __init__.
        # Tell the generic diffusion loader that all parameters are accounted for.
        return {name for name, _ in self.named_parameters()}

    def _prepare_text_audio_inputs(
        self,
        prompt: str,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        audio_placeholder = OmniDiffusionModelSpecialTokens.AUD_TAG.value
        if audio_placeholder not in prompt:
            prompt = f"{prompt}\n{audio_placeholder}" if prompt else audio_placeholder
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        input_ids = normalize_chat_template_token_ids(rendered)
        input_ids, audios, audio_indices = self.audio_tokenizer.prepare_contiguous_audio_inputs(
            input_ids=input_ids,
            omni_audios=audio,
            omni_audio_sample_rates=sample_rate,
            tokenizer_base_data=self.tokenizer_base_data,
        )
        return torch.tensor([input_ids], dtype=torch.long, device=self.device), audios, audio_indices

    def _extract_image_codebook_ids(self, generated_token_ids: torch.Tensor) -> torch.Tensor:
        image_offset = self.tokenizer.convert_tokens_to_ids(OMNI_DIFFUSION_IMAGE_START_TOKEN)
        image_mask = (generated_token_ids >= image_offset) & (
            generated_token_ids < image_offset + OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE
        )
        return generated_token_ids[image_mask] - image_offset

    @staticmethod
    def _get_dummy_audio(req: DiffusionRequestBatch) -> tuple[torch.Tensor, int]:
        """Read the in-memory audio created by DiffusionEngine warmup."""
        prompt = req.prompts[0]
        if isinstance(prompt, Mapping):
            mm_data = prompt.get("multi_modal_data")
            if isinstance(mm_data, Mapping) and "audio" in mm_data:
                audio = torch.as_tensor(mm_data["audio"]).float().contiguous()
                return audio, _S2I_DUMMY_AUDIO_SAMPLE_RATE
        raise ValueError("Omni-Diffusion S2I dummy run requires in-memory audio input.")

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        is_dummy_run = req.is_dummy_run()
        prompt = _get_prompt_text(req.prompts[0]) if req.prompts else ""
        if is_dummy_run:
            audio, sample_rate = self._get_dummy_audio(req)
        else:
            audio_source = _get_audio_source(req)
            audio, sample_rate = _decode_audio_source(audio_source)
        input_ids, audios, audio_indices = self._prepare_text_audio_inputs(prompt, audio, sample_rate)

        set_generation_seed(self.pipeline_config.seed)
        patch_legacy_dream_generation_config_validate(self.model.generation_config)
        ensure_dream_generation_config_fields(
            self.model.generation_config,
            self.model.config,
            self.tokenizer,
        )
        outputs, histories = self.model.generate(
            input_ids,
            generation_config=self.model.generation_config,
            audios=audios,
            audio_indices=audio_indices,
            temperature=self.pipeline_config.temperature,
            top_p=self.pipeline_config.top_p,
            steps=(req.sampling_params.num_inference_steps or 1) if is_dummy_run else self.pipeline_config.steps,
            max_new_tokens=self.pipeline_config.max_new_tokens,
            alg=self.pipeline_config.alg,
            cfg=self.pipeline_config.cfg,
            tokenizer=self.tokenizer,
            add_boa_token=self.pipeline_config.add_boa_token,
            max_position_penalty=self.pipeline_config.max_position_penalty,
            repeat_penalty=self.pipeline_config.repeat_penalty,
            output_text_only=self.pipeline_config.output_text_only,
            task=self.pipeline_config.task,
        )
        del histories

        if is_dummy_run:
            # A one-step warmup is not expected to produce a valid image block.
            # Decode a fixed token grid to warm the MagVIT path as well.
            image_token_ids = torch.zeros(
                _S2I_IMAGE_TOKEN_COUNT,
                dtype=torch.long,
                device=self.device,
            )
            decoded_image = self.image_tokenizer.decode(image_token_ids)
            return DiffusionOutput(output=_image_tensor_to_pil(decoded_image))

        generated_token_ids = outputs[0][input_ids.shape[1] :]
        image_token_ids = self._extract_image_codebook_ids(generated_token_ids)
        decoded_image = self.image_tokenizer.decode(image_token_ids)
        return DiffusionOutput(output=_image_tensor_to_pil(decoded_image))
