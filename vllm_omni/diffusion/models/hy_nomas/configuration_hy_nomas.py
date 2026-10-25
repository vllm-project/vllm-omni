# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

MODEL_ID = "tencent/Hy-Nomas"
DIT_FILE = "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
DECODER_FILE = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
REQUIRED_FILES = (DIT_FILE, DECODER_FILE)
DEFER_DOWNLOAD = True
