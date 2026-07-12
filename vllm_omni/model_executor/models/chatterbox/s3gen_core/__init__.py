"""Vendored S3Gen sources for vllm-omni.

Copied from ``chatterbox.models.s3gen`` with the minimum edits required to
(a) make the package self-contained and (b) enable batched inference
(batch size > 1): the three B>1 fixes in ``s3gen.py`` (drop the B==1 assert
in ``drop_invalid_tokens``; broadcast CFM noise and the HiFi-GAN cache_source
to B), plus reusing ``common.snake_activation.Snake`` in ``hifigan.py``.
"""

from .const import S3GEN_SR
from .s3gen import S3Token2Wav

__all__ = ["S3Token2Wav", "S3GEN_SR"]
