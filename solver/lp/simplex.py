from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from solver.config import SolverConfig
from solver.models import LPModel
from solver.results import LPSolution, SimplexState


@dataclass
class PivotChoice:
    row: int
    col: int


class LPSolver:
    def __init__(self, config: Optional[SolverConfig] = None) -> None:
        self.config = config or SolverConfig()

    def solve(self, model: LPModel, method: str = "primal", objective_sense: str = "max") -> LPSolution:
        sense = objective_sense.lower()
        if sense not in ("max", "min"):
            raise ValueError("objective_sense must be one of: max, min")

        effective_model = model
        sign = 1.0
        if sense == "min":
            sign = -1.0
            effective_model = LPModel(c=-np.asarray(model.c, dtype=float), A=model.A, b=model.b)

        state = self._build_initial_state(effective_model)
        if method == "primal":
            result = self._run_primal_simplex(state)
        elif method == "dual":
            result = self._run_dual_simplex(state)
        else:
            raise ValueError("method must be one of: primal, dual")

        if result.objective is not None:
            result.objective = float(sign * result.objective)
        return result

    def reoptimize_with_added_constraint(
        self,
        state: SimplexState,
        coeff: np.ndarray,
        rhs: float,
        method: str = "dual",
    ) -> LPSolution:
        new_state = self._add_constraint_to_state(state, coeff=coeff, rhs=rhs)
        if method == "dual":
            return self._run_dual_simplex(new_state)
        if method == "primal":
            return self._run_primal_simplex(new_state)
        raise ValueError("method must be one of: primal, dual")

    def reoptimize_with_added_full_constraint(
        self,
        state: SimplexState,
        coeff_full: np.ndarray,
        rhs: float,
        method: str = "dual",
    ) -> LPSolution:
        new_state = self._add_full_constraint_to_state(state, coeff_full=coeff_full, rhs=rhs)
        if method == "dual":
            return self._run_dual_simplex(new_state)
        if method == "primal":
            return self._run_primal_simplex(new_state)
        raise ValueError("method must be one of: primal, dual")

    def _build_initial_state(self, model: LPModel) -> SimplexState:
        A = np.asarray(model.A, dtype=float)
        b = np.asarray(model.b, dtype=float)
        c = np.asarray(model.c, dtype=float)

        m, n = A.shape
        tableau = np.zeros((m + 1, n + m + 1), dtype=float)
        tableau[:m, :n] = A
        tableau[:m, n : n + m] = np.eye(m)
        tableau[:m, -1] = b
        tableau[-1, :n] = -c

        basis = list(range(n, n + m))
        state = SimplexState(tableau=tableau, basis=basis, n_original=n)
        return state

    def _run_primal_simplex(self, state: SimplexState) -> LPSolution:
        tableau = state.tableau
        eps = self.config.epsilon

        if np.any(tableau[:-1, -1] < -eps):
            return LPSolution(
                status="infeasible",
                objective=None,
                x=None,
                state=state,
                message="Primal simplex requires nonnegative RHS for the initial basis.",
            )

        iterations = 0
        while iterations < self.config.max_iterations:
            entering = self._choose_entering_primal(tableau)
            if entering is None:
                return self._extract_solution(state, status="optimal", iterations=iterations)

            leaving = self._choose_leaving_primal(tableau, entering)
            if leaving is None:
                return LPSolution(
                    status="unbounded",
                    objective=None,
                    x=None,
                    state=state,
                    iterations=iterations,
                    message="No leaving row found; LP is unbounded.",
                )

            self._pivot(state, PivotChoice(row=leaving, col=entering))
            iterations += 1

        return LPSolution(
            status="iteration_limit",
            objective=None,
            x=None,
            state=state,
            iterations=iterations,
            message="Maximum iterations reached in primal simplex.",
        )

    def _run_dual_simplex(self, state: SimplexState) -> LPSolution:
        tableau = state.tableau
        eps = self.config.epsilon
        iterations = 0

        if np.any(tableau[-1, :-1] < -eps):
            return LPSolution(
                status="invalid_start",
                objective=None,
                x=None,
                state=state,
                message="Dual simplex requires dual-feasible reduced costs (>= 0).",
            )

        while iterations < self.config.max_iterations:
            leaving = self._choose_leaving_dual(tableau)
            if leaving is None:
                return self._extract_solution(state, status="optimal", iterations=iterations)

            entering = self._choose_entering_dual(tableau, leaving)
            if entering is None:
                return LPSolution(
                    status="infeasible",
                    objective=None,
                    x=None,
                    state=state,
                    iterations=iterations,
                    message="No valid entering variable found in dual simplex.",
                )

            self._pivot(state, PivotChoice(row=leaving, col=entering))
            iterations += 1

        return LPSolution(
            status="iteration_limit",
            objective=None,
            x=None,
            state=state,
            iterations=iterations,
            message="Maximum iterations reached in dual simplex.",
        )

    def _pivot(self, state: SimplexState, choice: PivotChoice) -> None:
        tableau = state.tableau
        r = choice.row
        c = choice.col
        pivot = tableau[r, c]
        if abs(pivot) < self.config.epsilon:
            raise ZeroDivisionError("Pivot value too small.")

        tableau[r, :] = tableau[r, :] / pivot
        n_rows = tableau.shape[0]
        for i in range(n_rows):
            if i == r:
                continue
            factor = tableau[i, c]
            if abs(factor) <= self.config.epsilon:
                continue
            tableau[i, :] -= factor * tableau[r, :]

        state.basis[r] = c

    def _choose_entering_primal(self, tableau: np.ndarray) -> Optional[int]:
        reduced_costs = tableau[-1, :-1]
        col = int(np.argmin(reduced_costs))
        if reduced_costs[col] >= -self.config.epsilon:
            return None
        return col

    def _choose_leaving_primal(self, tableau: np.ndarray, entering_col: int) -> Optional[int]:
        column = tableau[:-1, entering_col]
        rhs = tableau[:-1, -1]

        candidates: list[tuple[float, int]] = []
        for r, coeff in enumerate(column):
            if coeff > self.config.epsilon:
                candidates.append((rhs[r] / coeff, r))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1]

    def _choose_leaving_dual(self, tableau: np.ndarray) -> Optional[int]:
        rhs = tableau[:-1, -1]
        row = int(np.argmin(rhs))
        if rhs[row] >= -self.config.epsilon:
            return None
        return row

    def _choose_entering_dual(self, tableau: np.ndarray, leaving_row: int) -> Optional[int]:
        row = tableau[leaving_row, :-1]
        reduced_costs = tableau[-1, :-1]

        candidates: list[tuple[float, int]] = []
        for c, value in enumerate(row):
            if value < -self.config.epsilon:
                ratio = reduced_costs[c] / (-value)
                candidates.append((ratio, c))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1]

    def _extract_solution(self, state: SimplexState, status: str, iterations: int) -> LPSolution:
        tableau = state.tableau
        total_vars = tableau.shape[1] - 1
        x_all = np.zeros(total_vars, dtype=float)
        for r, col in enumerate(state.basis):
            x_all[col] = tableau[r, -1]

        x = x_all[: state.n_original]
        x[np.abs(x) <= self.config.epsilon] = 0.0
        objective = float(tableau[-1, -1])
        if abs(objective) <= self.config.epsilon:
            objective = 0.0

        return LPSolution(
            status=status,
            objective=objective,
            x=x,
            state=state,
            iterations=iterations,
        )

    def _add_constraint_to_state(self, state: SimplexState, coeff: np.ndarray, rhs: float) -> SimplexState:
        eps = self.config.epsilon
        parent = state.copy()
        tableau = parent.tableau
        m_plus_obj, old_cols_plus_rhs = tableau.shape
        m = m_plus_obj - 1
        old_cols = old_cols_plus_rhs - 1

        if coeff.shape[0] != parent.n_original:
            raise ValueError("Constraint coefficient length must match original variable count.")

        new_cols = old_cols + 1
        expanded = np.zeros((m_plus_obj, new_cols + 1), dtype=float)
        expanded[:, :old_cols] = tableau[:, :old_cols]
        expanded[:, -1] = tableau[:, -1]

        new_slack_col = old_cols
        row = np.zeros(new_cols + 1, dtype=float)
        row[: parent.n_original] = coeff
        row[new_slack_col] = 1.0
        row[-1] = rhs

        # Convert the added row to canonical form with respect to the current basis.
        for r, basic_col in enumerate(parent.basis):
            factor = row[basic_col]
            if abs(factor) <= eps:
                continue
            row -= factor * expanded[r, :]

        expanded_with_row = np.zeros((m_plus_obj + 1, new_cols + 1), dtype=float)
        expanded_with_row[:m, :] = expanded[:m, :]
        expanded_with_row[m, :] = row
        expanded_with_row[m + 1, :] = expanded[m, :]

        parent.tableau = expanded_with_row
        parent.basis.append(new_slack_col)
        return parent

    def _add_full_constraint_to_state(
        self,
        state: SimplexState,
        coeff_full: np.ndarray,
        rhs: float,
    ) -> SimplexState:
        eps = self.config.epsilon
        parent = state.copy()
        tableau = parent.tableau
        m_plus_obj, old_cols_plus_rhs = tableau.shape
        m = m_plus_obj - 1
        old_cols = old_cols_plus_rhs - 1

        coeff_arr = np.asarray(coeff_full, dtype=float)
        if coeff_arr.shape[0] != old_cols:
            raise ValueError("Full constraint length must match current tableau variable count.")

        new_cols = old_cols + 1
        expanded = np.zeros((m_plus_obj, new_cols + 1), dtype=float)
        expanded[:, :old_cols] = tableau[:, :old_cols]
        expanded[:, -1] = tableau[:, -1]

        new_slack_col = old_cols
        row = np.zeros(new_cols + 1, dtype=float)
        row[:old_cols] = coeff_arr
        row[new_slack_col] = 1.0
        row[-1] = rhs

        for r, basic_col in enumerate(parent.basis):
            factor = row[basic_col]
            if abs(factor) <= eps:
                continue
            row -= factor * expanded[r, :]

        expanded_with_row = np.zeros((m_plus_obj + 1, new_cols + 1), dtype=float)
        expanded_with_row[:m, :] = expanded[:m, :]
        expanded_with_row[m, :] = row
        expanded_with_row[m + 1, :] = expanded[m, :]

        parent.tableau = expanded_with_row
        parent.basis.append(new_slack_col)
        return parent
