# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""RNG helpers for diffusion pipelines."""

import contextlib
from collections.abc import Iterator

import torch


@contextlib.contextmanager
def seeded_global_rng(
    generator: torch.Generator | list[torch.Generator] | None,
    device: torch.device | str | None = None,
) -> Iterator[None]:
    """Seed the global torch RNG from ``generator`` for the duration of the block.

    ``transformers.GenerationMixin.generate()`` samples with ``torch.multinomial``
    off the global RNG, and rejects unknown keyword arguments outright
    (``_validate_model_kwargs`` raises ``ValueError``), so a request's
    ``torch.Generator`` cannot be handed to it directly. Seeding the global RNG
    from that generator instead is what makes prompt upsampling reproducible for
    a fixed ``SamplingParams.seed``.

    The seed is *read* with ``Generator.initial_seed()`` rather than drawn from
    ``generator``, so the caller's stream is left untouched and the latent noise
    sampled from it afterwards is unchanged. ``fork_rng`` restores the previous
    global state on exit, so nothing outside the block observes the reseed.

    Args:
        generator: The request's generator, or ``None`` to leave the RNG alone.
            A list (one generator per prompt, as ``diffusers`` allows) uses the
            first entry, since ``generate()`` has a single global RNG to seed
            regardless of batch size.
        device: Device generation runs on, whose RNG state has to be forked.
            Defaults to ``generator.device``.
    """
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is None:
        yield
        return

    dev = torch.device(device) if device is not None else generator.device
    # fork_rng() always saves the CPU state; naming the accelerator device
    # explicitly keeps it from saving (and warning about) every visible one.
    devices = [] if dev.type == "cpu" else [dev]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(generator.initial_seed())
        yield
