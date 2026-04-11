from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from solver.config import SolverConfig
from solver.models import IntegerModel
from solver.results import LPSolution


@dataclass
class HeuristicSolution:
    feasible: bool
    x: Optional[np.ndarray]
    objective: Optional[float]


def rounding_heuristic(
    model: IntegerModel,
    lp_solution: LPSolution,
    extra_constraints: List[Tuple[np.ndarray, float]],
    config: SolverConfig,
) -> HeuristicSolution:
    if lp_solution.x is None:
        return HeuristicSolution(feasible=False, x=None, objective=None)

    x = lp_solution.x.copy()
    int_idx = model.integer_indices
    x[int_idx] = np.round(x[int_idx])
    x[x < 0.0] = 0.0

    # Attempt a lightweight feasibility repair by reducing selected variables.
    A_full, b_full = _compose_constraints(model, extra_constraints)
    max_steps = max(1, int(config.rounding_max_repair_steps))
    for _ in range(max_steps):
        violation = A_full @ x - b_full
        max_v = float(np.max(violation))
        if max_v <= config.epsilon:
            break

        row = int(np.argmax(violation))
        adjusted = _reduce_for_row(model.c, A_full[row], x, violation[row], config.epsilon)
        if not adjusted:
            return HeuristicSolution(feasible=False, x=None, objective=None)

    if not _is_feasible_integer(model, x, extra_constraints, config.epsilon):
        return HeuristicSolution(feasible=False, x=None, objective=None)

    obj = float(np.dot(model.c, x))
    return HeuristicSolution(feasible=True, x=x, objective=obj)


def _reduce_for_row(
    c: np.ndarray,
    a_row: np.ndarray,
    x: np.ndarray,
    violation: float,
    eps: float,
) -> bool:
    candidates = []
    for j, a_ij in enumerate(a_row):
        if a_ij > eps and x[j] > eps:
            cost_per_fix = c[j] / a_ij if a_ij > eps else float("inf")
            candidates.append((cost_per_fix, j, a_ij))

    if not candidates:
        return False

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, j, a_ij = candidates[0]
    delta = min(violation / a_ij, x[j])
    x[j] -= delta
    if abs(x[j]) <= eps:
        x[j] = 0.0
    return True


def _compose_constraints(
    model: IntegerModel,
    extra_constraints: List[Tuple[np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    A_blocks = [np.asarray(model.A, dtype=float)]
    b_blocks = [np.asarray(model.b, dtype=float)]
    if extra_constraints:
        A_extra = np.vstack([c for c, _ in extra_constraints])
        b_extra = np.array([rhs for _, rhs in extra_constraints], dtype=float)
        A_blocks.append(A_extra)
        b_blocks.append(b_extra)
    return np.vstack(A_blocks), np.concatenate(b_blocks)


def _is_feasible_integer(
    model: IntegerModel,
    x: np.ndarray,
    extra_constraints: List[Tuple[np.ndarray, float]],
    eps: float,
) -> bool:
    if np.any(x < -eps):
        return False

    A_full, b_full = _compose_constraints(model, extra_constraints)
    if np.any(A_full @ x - b_full > eps):
        return False

    for idx in model.integer_indices:
        if abs(x[idx] - round(float(x[idx]))) > eps:
            return False
    return True
