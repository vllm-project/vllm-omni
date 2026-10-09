# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from vllm_omni.model_executor.models.chatterbox.s3gen_core.attr_dict import AttrDict

CFM_PARAMS = AttrDict(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
        "reg_loss_type": "l1",
    }
)
