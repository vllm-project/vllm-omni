# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BAGEL image-block positions on Model Runner V2."""

from vllm.v1.worker.gpu.mm.rope import RopeState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class BagelModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # BAGEL uses 1D RoPE with compressed image blocks. Reuse the native
        # prompt-position storage and decode offsets without declaring 3D RoPE.
        self.rope_state = RopeState(
            num_dims=1,
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.model_config.max_model_len,
            device=device,
        )
        # Layouts have the same lifetime and identity as the encoder outputs.
        # A cached feature can omit its pixel data in later requests.
        self._edit_layouts: dict[str, tuple[int, int, int, int]] = {}

    def execute_mm_encoder(self, scheduled_encoder_inputs):
        super().execute_mm_encoder(scheduled_encoder_inputs)
        cache = self.encoder_cache
        self._edit_layouts = {key: layout for key, layout in self._edit_layouts.items() if key in cache.encoder_outputs}
        for req_id, input_ids in scheduled_encoder_inputs.items():
            for input_id in input_ids:
                feature = cache.mm_features[req_id][input_id]
                if feature.modality != "img2img" or feature.data is None:
                    continue
                pixels = feature.data["pixel_values_img2img"].data
                height, width = pixels.shape[-2:]
                vae, vit = feature.mm_position.extract_embeds_range()
                self._edit_layouts[feature.identifier] = (
                    vae[1] - vae[0] + 1,
                    vit[1] - vit[0] + 1,
                    height,
                    width,
                )

    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        inputs["positions"] = inputs["positions"][0]
        # The multimodal runner otherwise supplies embeddings without IDs.
        # Img2img layout routing also needs the actual placeholder tokens.
        inputs["input_ids"] = input_batch.input_ids
        layouts = []
        for row, req_id in enumerate(input_batch.req_ids):
            start = int(input_batch.num_computed_tokens_np[row])
            end = start + int(input_batch.num_scheduled_tokens[row])
            layout = None
            for feature in self.encoder_cache.mm_features.get(req_id, ()):
                position = feature.mm_position
                if (
                    feature.modality == "img2img"
                    and start <= position.offset
                    and position.offset + position.length <= end
                ):
                    layout = self._edit_layouts[feature.identifier]
                    break
            layouts.append(layout)
        inputs["img2img_layouts"] = layouts
        return inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        inputs["positions"] = inputs["positions"][0]
        return inputs
