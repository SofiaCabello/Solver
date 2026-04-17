from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np

from solver.models import IntegerModel


@dataclass
class GreedySolution:
    status: str
    objective: Optional[float]
    x: Optional[np.ndarray]
    steps: int
    message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class GreedyIPSolver:
    """Constructive greedy solver with step-level exploration accounting.

    The solver is designed as a fast baseline/control method rather than an exact method.
    It incrementally adjusts integer variables and prioritizes constraint-violation reduction,
    then objective improvement.
    """

    def __init__(
        self,
        epsilon: float = 1e-9,
        max_steps: int = 20_000,
        default_upper_bound: int = 8,
        record_steps: bool = True,
        max_record_steps: int = 3_000,
    ) -> None:
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)
        self.default_upper_bound = max(1, int(default_upper_bound))
        self.record_steps = bool(record_steps)
        self.max_record_steps = int(max_record_steps)

    def solve(self, model: IntegerModel, objective_sense: str = "max") -> GreedySolution:
        sense = objective_sense.lower()
        if sense not in ("max", "min"):
            raise ValueError("objective_sense must be one of: max, min")

        start = perf_counter()

        A = np.asarray(model.A, dtype=float)
        b = np.asarray(model.b, dtype=float)
        c = np.asarray(model.c, dtype=float)

        n = c.shape[0]
        x = np.zeros(n, dtype=int)
        upper_bounds = self._infer_upper_bounds(A, b, n)
        int_idx = list(model.integer_indices)

        step_logs: List[Dict[str, Any]] = []
        candidate_evals = 0
        steps = 0

        viol = self._total_violation(A, b, x.astype(float))
        obj = float(c @ x)

        while steps < self.max_steps:
            best = None
            best_key = None

            for idx in int_idx:
                if x[idx] >= upper_bounds[idx]:
                    continue

                x_try = x.copy()
                x_try[idx] += 1

                v_try = self._total_violation(A, b, x_try.astype(float))
                o_try = float(c @ x_try)
                candidate_evals += 1

                # Primary: reduce violation. Secondary: improve objective in required sense.
                if sense == "max":
                    key = (v_try, -o_try, idx)
                else:
                    key = (v_try, o_try, idx)

                if best is None or key < best_key:
                    best = (idx, v_try, o_try)
                    best_key = key

            if best is None:
                break

            idx, best_v, best_o = best
            improve_violation = best_v < viol - self.epsilon
            improve_objective = self._objective_better(best_o, obj, sense)

            # If currently infeasible, we accept moves that reduce violation.
            # Once feasible, we require objective improvement while staying feasible.
            accept = False
            if viol > self.epsilon:
                accept = improve_violation
            else:
                accept = best_v <= self.epsilon and improve_objective

            if not accept:
                break

            x[idx] += 1
            viol = best_v
            obj = best_o
            steps += 1

            if self.record_steps and len(step_logs) < self.max_record_steps:
                step_logs.append(
                    {
                        "step": steps,
                        "selected_var": int(idx),
                        "value_after": int(x[idx]),
                        "objective_after": obj,
                        "violation_after": viol,
                        "candidate_evaluations_so_far": candidate_evals,
                    }
                )

        duration_ms = (perf_counter() - start) * 1000.0
        feasible = viol <= self.epsilon
        status = "optimal_or_feasible" if feasible else "feasible_not_found"
        message = "" if feasible else "Greedy solver stopped before reaching full feasibility."

        return GreedySolution(
            status=status,
            objective=float(obj),
            x=x.astype(float),
            steps=steps,
            message=message,
            metadata={
                "objective_sense": sense,
                "steps": steps,
                "candidate_evaluations": candidate_evals,
                "final_violation": float(viol),
                "time_ms": float(duration_ms),
                "upper_bounds": upper_bounds.tolist(),
                "step_trace": step_logs,
            },
        )

    def _infer_upper_bounds(self, A: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
        ub = np.full(n, self.default_upper_bound, dtype=int)
        for j in range(n):
            candidates = []
            for i in range(A.shape[0]):
                a_ij = A[i, j]
                if a_ij > self.epsilon and b[i] >= 0.0:
                    candidates.append(int(np.floor(b[i] / a_ij)))
            if candidates:
                ub[j] = max(1, min(candidates))
        return ub

    def _total_violation(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        lhs = A @ x
        return float(np.maximum(lhs - b, 0.0).sum())

    def _objective_better(self, candidate: float, current: float, sense: str) -> bool:
        if sense == "max":
            return candidate > current + self.epsilon
        return candidate < current - self.epsilon
