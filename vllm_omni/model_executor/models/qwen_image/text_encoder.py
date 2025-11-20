
import torch
import torch.nn as nn

from typing import Any, Optional

from dataclasses import dataclass

from vllm_omni.inputs.data import OmniDiffusionRequest


@dataclass
class QwenImageTextEncoderOutput():
    batch_size: int
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor

class QwenImageTextEncoder():
    def __init__(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2Tokenizer
        # Detect device once and move model to it so tensors are on the same device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen-Image", subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.eval()
        # Disable gradients for inference
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen-Image", subfolder="tokenizer"
        )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        # Normalize prompt: accept None, str, or list[str]
        if prompt is None:
            prompt = [""]
        else:
            prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask 
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result
    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] | None= None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # Use provided device or the encoder's device
        device = device or getattr(self, "device", torch.device("cpu"))
        dtype = dtype or next(self.text_encoder.parameters()).dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        # Safely format prompts, treating None as empty string
        txt = [template.format(e if e is not None else "") for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        )
        # Move token tensors to the target device
        txt_tokens = {k: v.to(device) for k, v in txt_tokens.items()}
        # Ensure attention_mask exists; fall back to ones if tokenizer omitted it
        attention_mask = txt_tokens.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(txt_tokens["input_ids"], device=device)

        # Run model in inference mode without tracking gradients
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(
                input_ids=txt_tokens["input_ids"],
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = encoder_attention_mask.to(device=device)
        return prompt_embeds, encoder_attention_mask

    def generate(self, req: list[OmniDiffusionRequest])-> list[OmniDiffusionRequest]:
        #TODO: maybe len(req) always 1
        for r in req:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(r.prompt)
            r.prompt_embeds = prompt_embeds
            r.prompt_embeds_mask = prompt_embeds_mask
        return req