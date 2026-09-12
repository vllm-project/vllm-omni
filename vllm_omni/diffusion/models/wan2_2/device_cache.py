# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-path device-cache release for the Wan2.2 pipelines."""

from vllm_omni.platforms.interface import OmniPlatform


def release_request_cache(platform: OmniPlatform) -> None:
    """Return cached allocator blocks to the driver between pipeline stages.

    The Wan2.2 pipelines call this on the request path (after denoising and
    before VAE decoding, and between clips in S2V) to lower the peak before
    the decoder allocates. Returning the blocks to the driver is not what
    makes the freed memory reusable -- the caching allocator already hands
    those blocks to the next allocation -- so on platforms where the call is
    harmful it can be skipped without changing what the decoder can allocate.

    It is harmful on XPU. Once the blocks are returned, the decoder's tensors
    land at new device addresses, and the XPU collective backend maps the
    peer segments those addresses belong to anew on every request without
    ever unmapping the old ones. Measured on a 4-rank Wan2.2-TI2V-5B service
    (SP4, no offload, 832x480x81 frames, 50 steps): host memory outside every
    user-space and cgroup counter fell by 10.3 GB per request, linearly and
    without a plateau, and came back only when the process exited. With this
    single call skipped, seven consecutive requests held host memory flat
    (227 -> 226 GB) at unchanged latency; a single-rank service never leaked,
    and reusing the collectives' receive buffers alone did not help, because
    the address churn comes from the cache release itself.

    Every other platform keeps the previous behaviour byte for byte.

    ``platform`` is passed in (normally ``current_omni_platform``) so each
    pipeline module keeps owning its platform binding.

    The ``is_available()`` check is kept from four of the five call sites and
    now also applies to the fifth (the S2V offload branch, which called
    ``empty_cache()`` unguarded); that is an extra gate on that one path, not a
    removed one.
    """
    if not platform.is_available():
        return
    if platform.is_xpu():
        return
    platform.empty_cache()
