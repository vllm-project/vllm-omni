from typing import Callable

import torch


class RungeKutta4ODESolver:
    def __init__(self, function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], initial_value: torch.Tensor):
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def reset_ode_solver(self, function, initial_value):
        self.function = function
        self.initial_value = initial_value

    def _rk4_step(
        self,
        function,
        time_start,
        time_step,
        time_end,
        value_start,
        function_value_start=None,
    ):
        k1 = function_value_start if function_value_start is not None else function(time_start, value_start)
        k2 = function(
            time_start + time_step * self._one_third,
            value_start + time_step * k1 * self._one_third,
        )
        k3 = function(
            time_start + time_step * self._two_thirds,
            value_start + time_step * (k2 - k1 * self._one_third),
        )
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        function_value_start = function(time_start, value_start)
        return (
            self._rk4_step(
                function,
                time_start,
                time_step,
                time_end,
                value_start,
                function_value_start=function_value_start,
            ),
            function_value_start,
        )

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points: torch.Tensor) -> torch.Tensor:
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            delta_value, _ = self._compute_step(self.function, time_start, time_step, time_end, current_value)
            next_value = current_value + delta_value

            while current_index < len(time_points) and time_end >= time_points[current_index]:
                solution[current_index] = self._linear_interpolation(
                    time_start,
                    time_end,
                    current_value,
                    next_value,
                    time_points[current_index],
                )
                current_index += 1

            current_value = next_value

        return solution
