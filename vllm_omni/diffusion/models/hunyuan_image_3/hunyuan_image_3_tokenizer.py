# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.utils.outputs import BaseOutput
from transformers import AutoTokenizer

from .hunyuan_image_3_transformer import ImageInfo, JointImageInfo, default


class TokenizerEncodeOutput(BaseOutput):
    tokens: torch.Tensor | None = None
    timestep_scatter_index: torch.Tensor | None = None
    guidance_scatter_index: torch.Tensor | None = None
    text_slices: list[slice] | None = None
    gen_image_slices: list[slice] | None = None
    joint_image_slices: list[slice] | None = None
    cond_vae_image_slices: list[slice] | None = None
    cond_vit_image_slices: list[slice] | None = None
    text_mask: torch.Tensor | None = None
    gen_image_mask: torch.Tensor | None = None
    cond_vae_image_mask: torch.Tensor | None = None
    cond_vit_image_mask: torch.Tensor | None = None
    real_pos: torch.Tensor | None = None
    all_image_slices: list[slice] | None = None
    cond_timestep_scatter_index: torch.Tensor | None = None
    gen_timestep_scatter_index: torch.Tensor | None = None


class Conversation:
    roles: list[str] = ["User", "Assistant"]
    sep: str = "\n\n"


class TokenizerWrapper:
    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Define short names
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<boi>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<eoi>")
        self.img_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.cfg_token_id = self.tokenizer.convert_tokens_to_ids("<cfg>")
        self.end_answer_token_id = self.tokenizer.convert_tokens_to_ids("</answer>")
        self.end_recaption_token_id = self.tokenizer.convert_tokens_to_ids("</recaption>")
        self.ratio_token_offset = self.tokenizer.convert_tokens_to_ids("<img_ratio_0>")
        self.special_token_map = self.tokenizer.added_tokens_encoder

    def pad(self, tensor_list, dim=0, pad_val=None):
        if pad_val is None:
            pad_val = self.pad_token_id
        max_len = max([t.shape[dim] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            if t.shape[dim] < max_len:
                assert pad_val is not False, "Not allowed pad."
                t = F.pad(t, (0, max_len - t.shape[dim]), value=pad_val)
            padded_tensor_list.append(t)
        return padded_tensor_list

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def encode_text(
        self,
        *texts,
        uncond_enabled: bool | list[bool] | None = None,
        uncond_p: float | None = None,
        max_length: int | None = None,
        pad: str | None = None,
        return_lengths: bool = False,
    ) -> list[torch.Tensor]:
        r"""
        Encode text and image for AR-like model training of the text-to-image/instruction tuning tasks.
        Support encode multiple texts at once. Each text can be separately conditioned or unconditioned
        based on the uncond_flags and a uniform uncond_p.
        **<bos> token is always prepended to the text tokens.**

        Args:
            texts (`str` or `List[str]`):
                List of texts to be encoded.
            uncond_enabled (`bool` or `List[bool]`):
                List of flags to indicate whether the text should be unconditioned.
                If False, the text will never be unconditioned.
                If True, the text will be unconditioned with uncond_p.
            uncond_p (`float`):
                Probability to the unconditional text. Only works when uncond_enabled is True.
            max_length (`int`):
                Maximum length of the encoded text.
            pad (`str`, *optional*):
                Padding method. Can be 'left' or 'right'.
            return_lengths (`bool`):
                Whether to return the length of each encoded text.

        Returns:
            `tuple[torch.Tensor, List[int]]` or `torch.Tensor`:
                If `return_lengths` is True, returns a tuple of (encoded_tokens, lengths).
                If `return_lengths` is False, returns only the encoded_tokens.
        """
        if pad is not None:
            assert max_length is not None, "max_length should be provided when pad is not None."

        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        if len(uncond_enabled) != len(texts):
            print(uncond_enabled, texts)
        assert len(uncond_enabled) == len(texts), (
            f"Length of uncond_flags should be equal to the number of texts, "
            f"but got {len(uncond_enabled)} and {len(texts)}."
        )

        # Prepare text/uncond tokens
        # TODO: If len(texts) > 1, such as instruction + prompt in inpainting, we need to determine how to do uncond.
        # Now all texts will be cond or uncond at the same time.
        do_uncond_drop = (uncond_p is not None) and (random.random() < uncond_p)
        text_tokens, lengths = [], []
        cum_length = 0
        for text, uncond_flag in zip(texts, uncond_enabled):
            # If reach the max_length and there still have unencoded texts, give a warning message and break the loop.
            if max_length is not None and cum_length >= max_length:
                warnings.warn(
                    f"Text length exceeds the max_length({max_length}). The remaining texts will be ignored: "
                    f"{text[:80]}..."
                )
                break
            # Set add_special_tokens=False to avoid adding <bos> token in some LLMs.
            if isinstance(text, str):
                text_token = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                text_token = text
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            # Cutoff the text by max_length if necessary
            if max_length is not None and (cum_length + len(text_token)) > max_length:
                text_token = text_token[: max_length - cum_length]
            text_tokens.extend(text_token)
            lengths.append(len(text_token))
            cum_length += len(text_token)

        # Prepend/Append <pad> tokens if applicable
        if pad is not None and (pad_length := max_length - len(text_tokens)) > 0:
            if pad == "left":
                text_tokens = [self.pad_token_id] * pad_length + text_tokens
            elif pad == "right":
                text_tokens = text_tokens + [self.pad_token_id] * pad_length
            else:
                raise ValueError(f"Unsupported padding method: {pad}.")

        if return_lengths:
            return text_tokens, lengths
        return text_tokens

    @staticmethod
    def _check_key_number_matched(keys, data):
        # Assert keys and token_source are matched
        assert set(keys) == set(data.keys()), (
            f"Keys in the template and token source should be matched, but got {set(keys)} and {list(data.keys())}."
        )
        key_counts = {k: 0 for k in keys}
        for key in keys:
            key_counts[key] += 1
        for key, count in key_counts.items():
            assert len(data[key]) == count, (
                f"Number of `{key}` in the token source should be matched with the template, but got "
                f"{data[key]}({len(data[key])}) and {count}."
            )

    def _add_image_meta_info_token(
        self,
        token_seq,
        token_count,
        extra_token_pos,
        add_timestep_token=False,
        add_image_shape_token=False,
        base_size=None,
        ratio_idx=None,
        image_type=None,
        add_guidance_token=False,
    ):
        if add_image_shape_token:
            token_seq.extend(
                [self.special_token_map[f"<img_size_{base_size}>"], self.special_token_map[f"<img_ratio_{ratio_idx}>"]]
            )
            token_count += 2
        if add_timestep_token:
            token_seq.extend([self.special_token_map["<timestep>"]])
            extra_token_pos["timestep"].append(token_count)
            if image_type is not None:
                if image_type == "gen_image":
                    extra_token_pos["gen_timestep"].append(token_count)
                elif image_type in ["joint_image"]:
                    extra_token_pos["cond_timestep"].append(token_count)
                else:
                    raise ValueError(f"Unsupported image type: {image_type}.")
            token_count += 1
        if add_guidance_token:
            token_seq.extend([self.special_token_map["<guidance>"]])
            extra_token_pos["guidance"].append(token_count)
            token_count += 1
        return token_count

    def encode_sequence(
        self,
        template: str,
        token_source: dict[str, list],
        total_length=None,
        add_timestep_token=False,
        add_guidance_token=False,
        last_key_only_prefix=False,
        add_eos=True,
        use_front_boi_token=True,
        add_pad=True,
        add_bos=True,
        drop_last: str | bool = "auto",
        add_image_shape_token=False,
    ) -> tuple[list, dict[str, list]]:
        r"""
        Encode a sequence based on the template (e.g., `text-image` for t2i, `text-image-image` for instruction tuning)
        and token source.

        Args:
            template (`str`):
                Template of the sequence. E.g., "text-gen_image" means the sequence is composed of text and an image.
                "text-text-gen_image" means the sequence is composed of two sections of text and an image.
            token_source (`Dict[str, List]`):
                Token source for each key in the template, in order.
                - text: List[Dict].
                - gen_image: List[Dict].
                - joint_image: List[Dict].
            total_length (`int`):
                Total length of the encoded sequence, include padding tokens.
            add_timestep_token (`bool`):
                Whether to add timestep token before the image tokens.
                (Right after the <img_ratio_*><img_size_*> tokens)
            add_guidance_token (`bool`):
                Whether to add guidance token before the image tokens.
            last_key_only_prefix (`bool`):
                Whether to only use the modal prefix in the last key.
            add_eos (`bool` or `'auto'`):
                Whether to add eos token at the end of the sequence. If True, always add eos token. If 'auto',
                add eos token only when the total_length is not reached and the last token is not <eos>.
            use_front_boi_token (`bool`):
                Whether to put the <boi> token at the front of iw, ih and timestep tokens.
            add_pad (`bool` or `'auto'`):
                Whether to add padding tokens to the sequence.
            add_bos (`bool`):
                Whether to add bos token at the beginning of the sequence.
            drop_last (`bool` or `'auto'`):
                - If auto, drop last tokens exceeding the total_length if the total_length is provided. If cut point is
                    in the middle of the image tokens, an error will raised.
                - If True, drop last tokens exceeding the total_length.
                - If False, keep the last tokens exceeding the total_length, even if the total_length is reached.
            add_image_shape_token (`bool`):
                Whether to add image shape token before the image tokens. (Right before the <timestep> token)

        Returns:
            `tuple[list, dict]`: A tuple containing:
                - token_seq (`list`): Encoded token sequence.
                - extra_token_pos (`dict`): Positions of extra tokens.
        """
        if last_key_only_prefix:
            assert add_eos is not True, "add_eos should not be True when last_key_only_prefix is True."
        if drop_last is True and total_length is None:
            raise ValueError("total_length should be provided when drop_last is True.")

        keys = template.split("-")
        modal_length = len(keys)
        index_indicator = {k: 0 for k in token_source}
        for k, v in token_source.items():
            assert isinstance(v, (list, tuple)), (
                f"Value of `{k}` in the token source should be a list or tuple, but got {type(v)}."
            )
        self._check_key_number_matched(keys, token_source)

        token_seq = []
        token_count = 0
        extra_token_pos = defaultdict(list)
        if add_bos:
            token_seq.append(self.bos_token_id)
            token_count += 1
        # If drop_last is True, we check the token_count on the fly and exit the loop if the total_length is reached.
        # This check is only applied to the block tokens. Block tokens mean the tokens that are unsplittable, like
        # image tokens. Text tokens are splittable, so we don't need to check the token_count for text.
        # If the loop is broken by drop_last, we don't add the eos token at the end because the sequence is not
        # complete.
        drop_last_break = False
        for i, key in enumerate(keys):
            source = token_source[key][index_indicator[key]]
            if key == "text":
                token_seq.extend(source)  # text token sequence
                extra_token_pos["<text>_start"].append(token_count)
                token_count += len(source)
                extra_token_pos["<text>_end"].append(token_count - 1)

            elif key == "gen_image":
                if isinstance(source, int):
                    source = {"length": source}
                extra_count = (
                    2
                    + (1 if source.get("timestep", add_timestep_token) else 0)
                    + (1 if source.get("guidance", add_guidance_token) else 0)
                    + (2 if source.get("image_shape", add_image_shape_token) else 0)
                )
                if drop_last is True and token_count + extra_count + source["length"] > total_length:
                    drop_last_break = True
                    break
                if source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source.get("timestep", add_timestep_token),
                    add_guidance_token=source.get("guidance", add_guidance_token),
                    add_image_shape_token=source.get("image_shape", add_image_shape_token),
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                if not source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                if last_key_only_prefix and i == modal_length - 1:
                    pass  # for AR inference
                else:
                    token_seq.extend(
                        [self.img_token_id] * source["length"]  # token number
                        + [self.eoi_token_id]
                    )
                    extra_token_pos["<img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source["length"]
                    extra_token_pos["<img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)
                    extra_token_pos["eoi"].append(token_count)
                    token_count += 1  # <eoi>

            elif key == "joint_image":
                assert isinstance(source["length"], list) and len(source["length"]) == 2, (
                    "joint_image length should be a list of two integers"
                )
                extra_count = (
                    2
                    + 1
                    + (  # boi, eoi, joint_img_sep
                        1 if source.get("timestep", add_timestep_token) else 0
                    )
                    + (2 if source.get("image_shape", add_image_shape_token) else 0)
                )
                if drop_last is True and token_count + extra_count + sum(source["length"]) > total_length:
                    drop_last_break = True
                    break
                if source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)  # Use patched boi for Janus, otherwise useing default <boi>
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source.get("timestep", add_timestep_token),
                    add_image_shape_token=source.get("image_shape", add_image_shape_token),
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                if not source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                if last_key_only_prefix and i == modal_length - 1:
                    pass  # for AR inference
                else:
                    token_seq.extend([self.img_token_id] * source["length"][0])
                    extra_token_pos["<vae_img>_start"].append(token_count)
                    extra_token_pos["<joint_img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source["length"][0]
                    extra_token_pos["<vae_img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)

                    token_seq.extend([self.special_token_map["<joint_img_sep>"]])
                    extra_token_pos["joint_img_sep"].append(token_count)
                    token_count += 1

                    token_seq.extend([self.img_token_id] * source["length"][1])
                    extra_token_pos["<vit_img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source["length"][1]
                    extra_token_pos["<vit_img>_end"].append(token_count - 1)
                    extra_token_pos["<joint_img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)

                    token_seq.extend([self.eoi_token_id])
                    extra_token_pos["eoi"].append(token_count)
                    token_count += 1  # <eoi>

            else:
                raise ValueError(f"Not supported key: {key}")
            index_indicator[key] += 1

        if add_eos is True and not drop_last_break:
            # Typically used for t2i task.
            token_seq.append(self.eos_token_id)
            extra_token_pos["eos"].append(token_count)
            token_count += 1
        elif add_eos == "auto" and not drop_last_break:
            # Typically used for lm and mmu task.
            if token_seq[-1] != self.eos_token_id and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token_id)
                extra_token_pos["eos"].append(token_count)
                token_count += 1

        if total_length:
            # Check token count and clip sequence if necessary
            if token_count > total_length and drop_last:
                # Assert clip position is not in the middle of the block-wise tokens (gen_image, joint_image)
                for start_key, end_key in [
                    ("<img>_start", "<img>_end"),
                    ("<joint_img>_start", "<joint_img>_end"),
                    ("<vae_img>_start", "<vae_img>_end"),
                    ("<vit_img>_start", "<vit_img>_end"),
                ]:
                    if start_key in extra_token_pos and end_key in extra_token_pos:
                        assert all(
                            (start > total_length or end + 1 < total_length)
                            for start, end in zip(extra_token_pos[start_key], extra_token_pos[end_key])
                        ), (
                            "Clip position should not be in the middle of the image tokens.\n"
                            f"Below is the text:\n{self._shorten_text(self.tokenizer.decode(token_seq))}"
                        )
                token_seq = token_seq[:total_length]

            # Pad the sequence if necessary
            pad_num = max(0, total_length - len(token_seq))
            if add_pad and pad_num:
                token_seq.extend([self.pad_token_id] * pad_num)
                extra_token_pos["first_pad"].append(token_count)

        return token_seq, extra_token_pos

    def batch_gen_infer(
        self,
        infer_fn,
        prompt_list: list,
        negative_prompt_list: list | None = None,
        infer_fn_kwargs_list: list[dict[str, int]] | None = None,
        do_classifier_free_guidance=False,
        condition_repeat_times: int = 1,
        uncondition_repeat_times: int = 1,
    ):
        r"""
        Batch inference for the AR-like model training of the text-to-image/instruction tuning tasks.

        Args:
            infer_fn (`callable`):
                Inference function to encode the prompt.
            prompt_list (`list`):
                List of prompts. Each element can be a single prompt or a list of prompts passed to the infer_fn.
            negative_prompt_list (`list`, *optional*):
                List of negative prompts. Only used when do_classifier_free_guidance is True. If None, will use <cfg>
                token sequence as negative prompt.
            infer_fn_kwargs_list (`List[Dict[str, int]]`, *optional*):
                List of keyword arguments for the infer_fn.
            do_classifier_free_guidance (`bool`):
                Whether to do classifier-free guidance.
            condition_repeat_times (`int`):
                Support multi-condition.
            uncondition_repeat_times (`int`):
                Support multi-uncondition.
        """
        if infer_fn_kwargs_list is None:
            infer_fn_kwargs_list = [{} for _ in prompt_list]

        # [n_output, bsz]
        cond_results_list = None
        uncond_results_list = None
        output_type_list = []

        for prompt_idx, (prompt, infer_fn_kwargs) in enumerate(zip(prompt_list, infer_fn_kwargs_list)):
            if not isinstance(prompt, (list, tuple)):
                prompt = [prompt]
            cond_kwargs = {"uncond_p": 0.0} if do_classifier_free_guidance else {}
            results = infer_fn(
                *prompt,
                **infer_fn_kwargs,
                **cond_kwargs,
            )
            output_type_list.append((type(results), len(results) if isinstance(results, (list, tuple)) else 1))
            if isinstance(results, dict):
                raise ValueError("Make batch on dict is not supported. Please return list or tuple for infer_fn.")
            if not isinstance(results, (list, tuple)):
                results = (results,)
            if cond_results_list is None:
                cond_results_list = [[] for _ in results]
                uncond_results_list = [[] for _ in results]
            for i, result in enumerate(results):
                cond_results_list[i].append(result)

            if do_classifier_free_guidance:
                if negative_prompt_list is None:
                    uncond_kwargs = {"uncond_p": 1.0}
                    uncond_results = infer_fn(
                        *prompt,
                        **infer_fn_kwargs,
                        **uncond_kwargs,
                    )
                else:
                    negative_prompt = negative_prompt_list[prompt_idx]
                    if not isinstance(negative_prompt, (list, tuple)):
                        negative_prompt = [negative_prompt]
                    uncond_results = infer_fn(
                        *negative_prompt,
                        **infer_fn_kwargs,
                    )
                if isinstance(uncond_results, TokenizerEncodeOutput):
                    uncond_results_list.append(uncond_results)
                else:
                    for i, result in enumerate(uncond_results):
                        uncond_results_list[i].append(result)

        assert all(output_type_list[0] == n for n in output_type_list), (
            f"Number of outputs should be equal for all samples, but got {output_type_list}."
        )
        output_type, output_num = output_type_list[0]

        def make_batch(batch_cond_item, batch_uncond_item):
            # Process each output item to make batch
            first = batch_cond_item[0]  # The first element in the batch
            if isinstance(first, torch.Tensor):
                stacked_item = torch.stack(
                    self.pad(
                        batch_cond_item * condition_repeat_times + batch_uncond_item * uncondition_repeat_times,
                    )
                )

            elif first is None:
                assert all(item is None for item in batch_cond_item + batch_uncond_item), (
                    f"The first cond item is None, but some items are not None:\n\n"
                    f"condition: {batch_cond_item}\n\n"
                    f"uncondition: {batch_uncond_item}"
                )
                stacked_item = None

            elif isinstance(first, (list, tuple)):
                # If the output item is a list or tuple, we treat it as a whole, and won't make nested batch any more.
                stacked_item = batch_cond_item * condition_repeat_times + batch_uncond_item * uncondition_repeat_times

            elif isinstance(first, TokenizerEncodeOutput):
                stacked_item = {}
                # Traverse not-None attributes
                for key in list(first.keys()):
                    merged_list = [cond_item[key] for cond_item in batch_cond_item] * condition_repeat_times + [
                        uncond_item[key] for uncond_item in batch_uncond_item
                    ] * uncondition_repeat_times
                    if isinstance(first[key], torch.Tensor):
                        if "mask" in key:
                            pad_val = 0.0
                        elif key == "tokens":
                            pad_val = self.special_token_map["<pad>"]
                        else:
                            pad_val = False  # Should not pad for other tensors
                        stacked_item[key] = torch.stack(self.pad(merged_list, pad_val=pad_val), dim=0)
                    elif isinstance(first[key], list):
                        stacked_item[key] = merged_list
                    elif first[key] is None:
                        pass
                    else:
                        raise ValueError(f"Unsupported type of {key}: {type(first[key])}.")
                stacked_item = TokenizerEncodeOutput(stacked_item)

            else:
                raise TypeError(f"Making batch on type {type(first)} is not supported.")

            return stacked_item

        stacked_outputs = []
        for cond_results, uncond_results in zip(cond_results_list, uncond_results_list):
            stacked_outputs.append(make_batch(cond_results, uncond_results))

        if output_type is list:
            return stacked_outputs
        elif output_type is tuple:
            return tuple(stacked_outputs)
        elif output_num == 1:
            return stacked_outputs[0]
        else:
            raise ValueError(f"Unsupported output type: {output_type}.")

    @staticmethod
    def parse_extra_token_pos(extra_token_pos, prefix, tokens, rng=None):
        if rng is None:
            rng = slice(None)
        image_slices = (
            [
                slice(start, end + 1)
                for start, end in zip(
                    extra_token_pos[f"<{prefix}>_start"][rng], extra_token_pos[f"<{prefix}>_end"][rng]
                )
            ]
            if f"<{prefix}>_start" in extra_token_pos and f"<{prefix}>_end" in extra_token_pos
            else []
        )
        if image_slices:
            image_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for image_slice in image_slices:
                image_mask[image_slice] = True
        else:
            image_mask = None
        return image_slices, image_mask

    def encode_general(
        self,
        sections: list[dict[str, Any]] | None = None,
        max_token_length: int | None = None,
        add_eos="auto",
        use_text_mask=True,
        add_pad="auto",
        add_bos=True,
        drop_last="auto",
    ) -> TokenizerEncodeOutput:
        r"""
        General encode function to encode a sequence with multiple sections of text and images.
        Each section is a dict with a `type` key and other keys depending on the type.

        Supported section types:

        - text: dict with keys:
            - text (`str` or `List[int]`): Text to be encoded. Either `text` or `tokens` should be provided.
            - tokens (`List[int]`): Pre-encoded text tokens. Either `text` or `tokens` should be provided.
            - uncond_enabled (`bool`): Whether to enable uncondition for this text section.
            - uncond_p (`float`): Probability to drop the text section for uncondition.
            - max_length (`int`): Maximum length of the text section.
            - ignore (`bool`): Whether to ignore this text section in the text mask.
            - start_offset (`int`): Start offset of the text mask.
            - end_offset (`int`): End offset of the text mask.

        - gen_image: dict with keys:
            - token_length (`int`): Number of image tokens.
            - add_timestep_token (`bool`): Whether to add timestep token before the image tokens.
            - add_guidance_token (`bool`): Whether to add guidance token before the image tokens.
            - use_front_boi_token (`bool`): Whether to put the <boi> token.
            - add_image_shape_token (`bool`): Whether to add image shape token before the image tokens.
            - base_size (`int`): Base size of the image.
            - ratio_idx (`int`): Ratio index of the image.

        - joint_image: dict with keys:
            - token_length (`List[int]`): Number of image tokens for the two images.
            - add_timestep_token (`bool`): Whether to add timestep token before the image tokens.
            - use_front_boi_token (`bool`): Whether to put the <boi> token.
            - add_image_shape_token (`bool`): Whether to add image shape token before the image tokens.
            - base_size (`int`): Base size of the image.
            - ratio_idx (`int`): Ratio index of the image.

        Args:
            sections (`List[Dict[str, Any]]`):
                List of sections to be encoded.
            max_token_length (`int`):
                Maximum length of the encoded token sequence.
            add_eos (`bool` or `'auto'`):
                Whether to add eos token at the end of the sequence.
            use_text_mask (`bool`):
                Whether to generate text mask.
            add_pad (`bool` or `'auto'`):
                Whether to add padding tokens to the sequence. If True and total_length is not reached,
                add padding tokens.
            add_bos (`bool`):
                Whether to add bos token at the beginning of the sequence.
            drop_last (`bool` or `'auto'`):
                - If auto, drop last tokens exceeding the total_length if the total_length is provided.
                If cut point is in the middle of the image tokens, an error will raised.
                - If True, drop last tokens exceeding the total_length. If cut point is in the
                middle of the image tokens, all the successive image tokens will be dropped.
                - If False, keep the last tokens exceeding the total_length, even if the total_length
                is reached.

        Returns:
            `TokenizerEncodeOutput`: Encoded token sequence and extra information.
        """
        if sections is None:
            raise ValueError("sections must be provided.")
        template = "-".join([section["type"] for section in sections])

        sections = deepcopy(sections)
        token_source = defaultdict(list)
        text_mask_specs = []
        for section in sections:
            if section["type"] == "text":
                text = self.encode_text(
                    section["text"] if "text" in section else section["tokens"],
                    uncond_enabled=section.get("uncond_enabled"),
                    uncond_p=section.get("uncond_p"),
                    max_length=section.get("max_length"),
                )
                token_source["text"].append(text)
                text_mask_specs.append(
                    dict(
                        ignore=section.get("ignore", False),
                        start_offset=section.get("start_offset", 0),
                        end_offset=section.get("end_offset", 0),
                    )
                )
            elif section["type"] == "gen_image":
                token_source["gen_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", False),
                        guidance=section.get("add_guidance_token", False),
                        front_boi=section.get("use_front_boi_token", False),
                        image_shape=section.get("add_image_shape_token", False),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            elif section["type"] == "joint_image":
                token_source["joint_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", False),
                        front_boi=section.get("use_front_boi_token", False),
                        image_shape=section.get("add_image_shape_token", False),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            else:
                raise ValueError(f"Invalid section type: {section['type']}")

        # Combine text and image tokens
        full_token_seq, extra_token_pos = self.encode_sequence(
            template=template,
            token_source=dict(token_source),
            total_length=max_token_length,
            add_eos=add_eos,
            add_pad=add_pad,
            add_bos=add_bos,
            drop_last=drop_last,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)

        timestep_scatter_index = (
            torch.tensor(extra_token_pos["timestep"], dtype=torch.long) if "timestep" in extra_token_pos else None
        )
        guidance_scatter_index = (
            torch.tensor(extra_token_pos["guidance"], dtype=torch.long) if "guidance" in extra_token_pos else None
        )
        cond_timestep_scatter_index = (
            torch.tensor(extra_token_pos["cond_timestep"], dtype=torch.long)
            if "cond_timestep" in extra_token_pos
            else None
        )
        gen_timestep_scatter_index = (
            torch.tensor(extra_token_pos["gen_timestep"], dtype=torch.long)
            if "gen_timestep" in extra_token_pos
            else None
        )

        # Gen image mask
        gen_image_slices, gen_image_mask = self.parse_extra_token_pos(extra_token_pos, "img", full_seq_token_tensor)
        # Joint image
        joint_image_slices, _ = self.parse_extra_token_pos(extra_token_pos, "joint_img", full_seq_token_tensor)
        # Conditional vae image
        cond_vae_image_slices, cond_vae_image_mask = self.parse_extra_token_pos(
            extra_token_pos, "vae_img", full_seq_token_tensor
        )
        # Conditional vit image
        cond_vit_image_slices, cond_vit_image_mask = self.parse_extra_token_pos(
            extra_token_pos, "vit_img", full_seq_token_tensor
        )
        # All image slices (gen_image, joint_image)
        all_image_slices = (
            [
                slice(start, end + 1)
                for start, end in zip(extra_token_pos["<all_img>_start"], extra_token_pos["<all_img>_end"])
            ]
            if "<all_img>_start" in extra_token_pos and "<all_img>_end" in extra_token_pos
            else []
        )

        # Text mask
        text_slices = (
            [
                slice(start, end + 1)
                for start, end in zip(extra_token_pos["<text>_start"], extra_token_pos["<text>_end"])
            ]
            if "<text>_start" in extra_token_pos and "<text>_end" in extra_token_pos
            else []
        )
        assert len(text_slices) <= len(text_mask_specs), (
            f"Number of text slices ({len(text_slices)}) should be less than or equal to "
            f"number of text mask specs ({len(text_mask_specs)})"
        )
        if use_text_mask:
            text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
            for text_slice, mask_spec in zip(text_slices, text_mask_specs):
                if not mask_spec["ignore"]:
                    real_slice = slice(
                        text_slice.start + mask_spec["start_offset"], text_slice.stop + mask_spec["end_offset"]
                    )
                    text_mask[real_slice] = 1.0
        else:
            text_mask = None

        # real_pos is the first position of the <pad> token
        real_pos = torch.tensor(extra_token_pos.get("first_pad", [full_seq_token_tensor.shape[0]]), dtype=torch.long)

        return TokenizerEncodeOutput(
            tokens=full_seq_token_tensor,
            timestep_scatter_index=timestep_scatter_index,
            guidance_scatter_index=guidance_scatter_index,
            text_slices=text_slices,
            gen_image_slices=gen_image_slices,
            joint_image_slices=joint_image_slices,
            cond_vae_image_slices=cond_vae_image_slices,
            cond_vit_image_slices=cond_vit_image_slices,
            text_mask=text_mask,
            gen_image_mask=gen_image_mask,
            cond_vae_image_mask=cond_vae_image_mask,
            cond_vit_image_mask=cond_vit_image_mask,
            real_pos=real_pos,
            all_image_slices=all_image_slices,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )

    def get_cot_sections(self, cot_text, uncond_kwargs, cot_max_length=None, drop_think=False):
        if not cot_text:  # None or empty
            return []
        if "<think>" in cot_text and "</think>" in cot_text:
            before_think_sec = cot_text.split("<think>")[0]
            after_think_sec = cot_text.split("</think>")[1]
            think_sec = cot_text.split("<think>")[1].split("</think>")[0]
            return (
                self.get_cot_sections(before_think_sec, uncond_kwargs, drop_think=drop_think)
                + (
                    [
                        dict(type="text", text="<think>"),
                        dict(type="text", text=think_sec, max_length=cot_max_length, **uncond_kwargs),
                        dict(type="text", text="</think>"),
                    ]
                    if not drop_think
                    else []
                )
                + self.get_cot_sections(after_think_sec, uncond_kwargs, drop_think=drop_think)
            )

        if "<recaption>" in cot_text and "</recaption>" in cot_text:
            before_recaption_sec = cot_text.split("<recaption>")[0]
            after_recaption_sec = cot_text.split("</recaption>")[1]
            recaption_sec = cot_text.split("<recaption>")[1].split("</recaption>")[0]
            return (
                self.get_cot_sections(before_recaption_sec, uncond_kwargs, drop_think=drop_think)
                + [
                    dict(type="text", text="<recaption>"),
                    dict(type="text", text=recaption_sec, max_length=cot_max_length, **uncond_kwargs),
                    dict(type="text", text="</recaption>"),
                ]
                + self.get_cot_sections(after_recaption_sec, uncond_kwargs, drop_think=drop_think)
            )

        return [
            dict(type="text", text=cot_text, **uncond_kwargs),
        ]

    def apply_general_template(
        self,
        message_list,
        max_length=None,
        add_assistant_prefix=False,
        answer="auto",
        bot_task="auto",
        sequence_template="instruct",
        uncond_p=0.0,
        cfg_factor=1,
        batchify=False,
        image_base_size=1024,
        drop_think=False,
    ):
        # If cfg_factor > 1, we need to repeat the unconditioned part
        if batchify:
            assert isinstance(message_list[0], list), (
                f"When batchify is True, message_list should be a list of list, but got [{type(message_list[0])}, ...]."
            )
            return self.batch_gen_infer(
                infer_fn=self.apply_general_template,
                prompt_list=[[]],
                infer_fn_kwargs_list=[
                    dict(
                        message_list=message_list_i,
                        max_length=max_length,
                        add_assistant_prefix=add_assistant_prefix,
                        answer=answer,
                        bot_task=bot_task,
                        sequence_template=sequence_template,
                        image_base_size=image_base_size,
                        drop_think=drop_think,
                    )
                    for message_list_i in message_list
                ],
                do_classifier_free_guidance=cfg_factor > 1,
                condition_repeat_times=1,
                uncondition_repeat_times=cfg_factor - 1,
            )

        conv = Conversation()
        uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)

        def process_successive_message(
            _message_list, _cur_message_idx, role, prefix, suffix, answer_prefix="", answer_suffix=""
        ):
            _sub_sections = []
            while _cur_message_idx < len(message_list) and _message_list[_cur_message_idx]["role"] == role:
                message = _message_list[_cur_message_idx]
                if message["type"] == "text":
                    text = message["content"]
                    if role == "system":
                        _sub_sections.append(dict(type="text", text=text))
                    elif role == "assistant":
                        if ("<recaption>" in text and "</recaption>" in text) or (
                            "<think>" in text and "</think>" in text
                        ):
                            _sub_sections.extend(self.get_cot_sections(text, uncond_kwargs, drop_think=drop_think))
                        else:
                            _sub_sections.append(dict(type="text", text=text, **uncond_kwargs))
                    else:
                        _sub_sections.append(
                            dict(type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs)
                        )
                elif message["type"] == "gen_image":
                    info = message["content"]
                    assert isinstance(info, ImageInfo), f"Expected ImageInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message["type"], **info.meta_info))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message["type"] == "joint_image":
                    info = message["content"]
                    assert isinstance(info, JointImageInfo), f"Expected JointImageInfo, but got {type(info)}"
                    _sub_sections.append(dict(type=message["type"], **info.meta_info))
                else:
                    raise ValueError(f"Unknown message type: {message['type']}")
                _cur_message_idx += 1
            if len(_sub_sections) > 0:
                # Add role prefix and suffix
                _sub_sections.insert(0, dict(type="text", text=prefix))
                _sub_sections.append(dict(type="text", text=suffix))
            return _sub_sections, _cur_message_idx

        # Define assistant prefix and suffix
        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = "<answer>", "</answer>"
        else:
            answer_prefix, answer_suffix = "", ""
        if sequence_template == "pretrain":
            system_suffix = ""
            user_prefix = ""
            user_suffix = ""
            bot_prefix = ""
            bot_suffix = ""
        else:
            system_suffix = f"{conv.sep}"
            user_prefix = f"{conv.roles[0]}: "
            user_suffix = f"{conv.sep}"
            bot_prefix = f"{conv.roles[1]}: "
            bot_suffix = f"{conv.sep}"

        # Process successive user and assistant messages
        sections = []
        cur_message_idx = 0
        final_role = None
        while cur_message_idx < len(message_list):
            # Process successive system messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="system", prefix="", suffix=system_suffix
            )
            # Add to the template and sections
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "system"

            # Process successive user messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="user", prefix=user_prefix, suffix=user_suffix
            )
            # Add to the template and sections
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "user"

            # Process successive assistant messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list,
                cur_message_idx,
                role="assistant",
                prefix=bot_prefix,
                suffix=bot_suffix,
                answer_prefix=answer_prefix,
                answer_suffix=answer_suffix,
            )
            # Add to the template and sections
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "assistant"

        if add_assistant_prefix:
            if final_role == "assistant":
                # Avoid adding prefix twice
                _bot_prefix = ""
                # Remove the final bot_suffix
                if len(sections) > 0 and sections[-1]["type"] == "text" and sections[-1]["text"] == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            # We can add special tokens for the bot latest message according to different tasks
            bot_response_prefix = dict(
                auto=_bot_prefix,
                image="",
                think=f"{_bot_prefix}<think>",
                recaption=f"{_bot_prefix}<recaption>",
                img_ratio=f"{_bot_prefix}{answer_prefix}<boi><img_size_{image_base_size}>",
            )[bot_task]
            sections.append(dict(type="text", text=bot_response_prefix))

        output = self.encode_general(
            sections=sections,
            use_text_mask=False,
            add_eos=False,
            add_pad=False,
        )

        if max_length is not None:
            if output.tokens.shape[-1] > max_length:
                raise ValueError(
                    f"Encoded token length {output.tokens.shape[-1]} exceeds max_length {max_length}.\n"
                    f"Please set a larger max_length or check the input messages:\n{message_list}"
                )

        return output, sections

    def apply_chat_template(
        self,
        batch_prompt: list[str] | None = None,
        batch_message_list: list[list[dict[str, Any]]] | None = None,
        mode: str = "gen_text",
        batch_gen_image_info: list[ImageInfo] | None = None,
        batch_cond_image_info: list[JointImageInfo] | list[list[JointImageInfo]] | None = None,
        batch_system_prompt: list[str] | None = None,
        batch_cot_text: list[str] | None = None,
        max_length: int | None = None,
        bot_task: str = "auto",  # auto/image/think/recaption/img_ratio
        image_base_size: int = 1024,
        sequence_template: str = "pretrain",
        cfg_factor: int = 1,
        add_assistant_prefix: bool | None = None,
        drop_think: bool = False,
    ) -> dict[str, Any]:
        assert bot_task in ["image", "auto", "think", "recaption", "img_ratio"], (
            f"bot_task should be one of ['image', 'auto', 'think', 'recaption', 'img_ratio'], but got {bot_task}."
        )

        if batch_message_list is None:
            # Simple text-to-image or text-cot-to-image task
            batch_size = len(batch_prompt)

            # Batchify inputs
            if not isinstance(batch_system_prompt, list):
                batch_system_prompt = [batch_system_prompt] * batch_size
            if not isinstance(batch_gen_image_info, list):
                batch_gen_image_info = [batch_gen_image_info] * batch_size
            if batch_cot_text is not None:
                assert len(batch_cot_text) == batch_size, (
                    f"batch_cot_text should have the same length as batch_size ({batch_size}), "
                    f"but got {len(batch_cot_text)}."
                )
            else:
                batch_cot_text = [None] * batch_size
            if batch_cond_image_info is not None:
                assert len(batch_cond_image_info) == batch_size, (
                    f"batch_cond_image_info should have the same length as batch_size ({batch_size}), "
                    f"but got {len(batch_cond_image_info)}."
                )
                batch_cond_image_info = [
                    cond_image_info if isinstance(cond_image_info, list) else [cond_image_info]
                    for cond_image_info in batch_cond_image_info
                ]
            else:
                batch_cond_image_info = [[] for _ in range(batch_size)]

            # Convert single round materials into standard message list
            batch_message_list = []
            for prompt, system_prompt, cot_text, gen_image_info, cond_image_info_list in zip(
                batch_prompt,
                batch_system_prompt,
                batch_cot_text,
                batch_gen_image_info,
                batch_cond_image_info,
            ):
                message_list = []
                # 1. system prompt section
                if system_prompt:
                    message_list.append(dict(role="system", type="text", content=system_prompt, context_type="str"))
                # 2. user inputs sections
                #   2.1 image inputs
                if len(cond_image_info_list) > 0:
                    message_list.extend(
                        [
                            dict(role="user", type="joint_image", content=cond_image_info, context_type="image_info")
                            for cond_image_info in cond_image_info_list
                        ]
                    )
                #   2.2 text inputs
                message_list.append(dict(role="user", type="text", content=prompt, context_type="str"))
                # 3. assistant answer sections
                if cot_text is not None:
                    message_list.append(dict(role="assistant", type="text", content=cot_text, context_type="str"))
                if mode == "gen_image":
                    message_list.append(
                        dict(role="assistant", type="gen_image", content=gen_image_info, context_type="image_info")
                    )
                batch_message_list.append(message_list)

        output, sections = self.apply_general_template(
            message_list=batch_message_list,
            max_length=max_length,
            add_assistant_prefix=default(add_assistant_prefix, mode != "gen_image"),
            bot_task=bot_task,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor,
            batchify=True,
            image_base_size=image_base_size,
            drop_think=drop_think,
        )
        return dict(output=output, sections=sections)
