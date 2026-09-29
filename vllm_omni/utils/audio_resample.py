# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Streaming resampling, shared by the engine and the entrypoints.

Lives here rather than beside the OpenAI audio mixin because the duplex
engine needs it too, and engine code must not import the entrypoints layer.
Pure numpy on purpose: importing this must not pull in torch or the API
server.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["StreamingAudioResampler"]


class StreamingAudioResampler:
    """Stateful polyphase resampler for streaming mono float audio.

    Retains filter state so output is invariant to input chunk boundaries.
    """

    _half_filter_width = 10
    _kaiser_beta = 5.0
    max_polyphase_factor = 2_048

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Audio sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        rate_gcd = math.gcd(source_rate, target_rate)
        self._up = target_rate // rate_gcd
        self._down = source_rate // rate_gcd
        if max(self._up, self._down) > self.max_polyphase_factor:
            raise ValueError("source and target sample rates have an unsupported resampling ratio")
        self._half_len, self._phase_kernels = self._design_polyphase_filter()
        self._history_samples = self._phase_kernels.shape[1] - 1
        self.reset()

    def _design_polyphase_filter(self) -> tuple[int, np.ndarray]:
        if self._up == self._down:
            return 0, np.ones((1, 1), dtype=np.float32)

        max_rate = max(self._up, self._down)
        half_len = self._half_filter_width * max_rate
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64)
        cutoff = 1.0 / max_rate
        taps = cutoff * np.sinc(cutoff * offsets)
        taps *= np.kaiser(taps.size, self._kaiser_beta)
        taps /= np.sum(taps)
        taps *= self._up
        taps = np.ascontiguousarray(taps, dtype=np.float32)

        phases = [taps[phase :: self._up] for phase in range(self._up)]
        phase_width = max(phase.size for phase in phases)
        kernels = np.zeros((self._up, phase_width), dtype=np.float32)
        for phase_index, phase in enumerate(phases):
            kernels[phase_index, -phase.size :] = phase[::-1]
        return half_len, kernels

    @property
    def scratch_bytes(self) -> int:
        pending_samples = self._ceil_div(self._input_samples * self._up, self._down) - self._output_samples
        return max(0, pending_samples) * np.dtype(np.float32).itemsize

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return -(-numerator // denominator)

    def _render_outputs(
        self,
        combined: np.ndarray,
        *,
        first_output: int,
        output_count: int,
        input_start: int,
    ) -> np.ndarray:
        if output_count <= 0:
            return np.empty(0, dtype=np.float32)

        output_indexes = np.arange(first_output, first_output + output_count, dtype=np.int64)
        filter_indexes = output_indexes * self._down + self._half_len
        source_indexes = filter_indexes // self._up
        phases = filter_indexes % self._up
        windows = np.lib.stride_tricks.sliding_window_view(
            combined,
            self._phase_kernels.shape[1],
        )
        selected_windows = windows[source_indexes - input_start]
        return np.sum(
            selected_windows * self._phase_kernels[phases],
            axis=1,
            dtype=np.float32,
        )

    def _push(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.empty(0, dtype=np.float32)
        if self._flushed:
            raise RuntimeError("cannot process audio after the resampler has been finalized; call reset first")

        combined = np.concatenate((self._history, chunk))
        next_input_samples = self._input_samples + chunk.size
        stable_output_samples = max(
            0,
            self._ceil_div(next_input_samples * self._up - self._half_len, self._down),
        )
        output = self._render_outputs(
            combined,
            first_output=self._output_samples,
            output_count=stable_output_samples - self._output_samples,
            input_start=self._input_samples,
        )
        if self._history_samples:
            self._history = combined[-self._history_samples :].copy()
        self._input_samples = next_input_samples
        self._output_samples = stable_output_samples
        return output

    def _flush(self) -> np.ndarray:
        if self._flushed:
            return np.empty(0, dtype=np.float32)

        total_output_samples = self._ceil_div(self._input_samples * self._up, self._down)
        output_count = total_output_samples - self._output_samples
        if output_count:
            last_output = total_output_samples - 1
            last_source_index = (last_output * self._down + self._half_len) // self._up
            right_padding = max(0, last_source_index - self._input_samples + 1)
            combined = np.concatenate((self._history, np.zeros(right_padding, dtype=np.float32)))
            output = self._render_outputs(
                combined,
                first_output=self._output_samples,
                output_count=output_count,
                input_start=self._input_samples,
            )
        else:
            output = np.empty(0, dtype=np.float32)
        self._output_samples = total_output_samples
        self._flushed = True
        return output

    def process(self, audio: np.ndarray, *, final: bool = False) -> np.ndarray:
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"Streaming audio resampling only supports mono audio, got shape {chunk.shape}")
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        output = self._push(chunk)
        if not final:
            return output
        tail = self._flush()
        if not output.size:
            return tail
        if not tail.size:
            return output
        return np.concatenate((output, tail))

    def reset(self) -> None:
        self._history = np.zeros(self._history_samples, dtype=np.float32)
        self._input_samples = 0
        self._output_samples = 0
        self._flushed = False
