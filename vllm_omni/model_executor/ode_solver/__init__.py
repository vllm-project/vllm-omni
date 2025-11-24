from vllm_omni.model_executor.ode_solver.runger_kutta_4 import RungeKutta4ODESolver

_ODE_SOLVERS = {
    "RungeKutta4ODESolver": RungeKutta4ODESolver,
}


def get_ode_solver_class(class_name: str):
    solver = _ODE_SOLVERS.get(class_name)
    if solver is None:
        raise ValueError(f"ODE solver class '{class_name}' not found.")
    return solver


__all__ = ["RungeKutta4ODESolver"]
