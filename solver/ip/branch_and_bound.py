from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from solver.config import SolverConfig
from solver.ip.heuristics import rounding_heuristic
from solver.lp.simplex import LPSolver
from solver.models import IntegerModel
from solver.results import BnBResult, LPSolution, SimplexState


@dataclass
class Node:
    state: SimplexState
    lp_solution: LPSolution
    extra_constraints: List[Tuple[np.ndarray, float]]
    gomory_constraints: List[Tuple[np.ndarray, float]]
    depth: int
    node_id: int
    parent_id: Optional[int]
    branch_constraint: Optional[Tuple[np.ndarray, float]] = None


class BranchAndBoundSolver:
    def __init__(self, config: Optional[SolverConfig] = None) -> None:
        self.config = config or SolverConfig()
        self.lp_solver = LPSolver(config=self.config)

    def solve(self, model: IntegerModel, objective_sense: str = "max") -> BnBResult:
        sense = objective_sense.lower()
        if sense not in ("max", "min"):
            raise ValueError("objective_sense must be one of: max, min")

        if sense == "min":
            transformed = IntegerModel(
                c=-np.asarray(model.c, dtype=float),
                A=np.asarray(model.A, dtype=float),
                b=np.asarray(model.b, dtype=float),
                integer_indices=list(model.integer_indices),
            )
            result = self.solve(transformed, objective_sense="max")
            if result.objective is not None:
                result.objective = -float(result.objective)
            result.metadata = dict(result.metadata)
            result.metadata["objective_sense"] = "min"
            return result

        collect_trace = self.config.visualize and model.c.shape[0] == 2
        trace: List[Dict[str, Any]] = []
        branch_lines: List[Dict[str, Any]] = []
        gomory_lines: List[Dict[str, Any]] = []
        gomory_cuts_added = 0

        root = self.lp_solver.solve(model, method="primal")
        if root.status != "optimal" and np.any(np.asarray(model.b, dtype=float) < -self.config.epsilon):
            dual_root = self.lp_solver.solve(model, method="dual")
            if dual_root.status == "optimal":
                root = dual_root

        if root.status != "optimal" or root.state is None or root.x is None:
            return BnBResult(
                status="infeasible",
                objective=None,
                x=None,
                nodes_visited=1,
                incumbent_updates=0,
                message=f"Root LP solve failed with status={root.status}",
            )

        incumbent_x: Optional[np.ndarray] = None
        incumbent_obj = -float("inf")
        incumbent_updates = 0
        incumbent_constraints: List[Tuple[np.ndarray, float]] = []
        incumbent_gomory_constraints: List[Tuple[np.ndarray, float]] = []

        # Early incumbent from root rounding can aggressively prune the tree.
        if self.config.use_rounding_heuristic:
            rounded_root = rounding_heuristic(
                model=model,
                lp_solution=root,
                extra_constraints=[],
                config=self.config,
            )
            if rounded_root.feasible and rounded_root.objective is not None and rounded_root.x is not None:
                incumbent_obj = rounded_root.objective
                incumbent_x = rounded_root.x
                incumbent_updates += 1
                incumbent_constraints = []

        next_node_id = 1
        stack: list[Node] = [
            Node(
                state=root.state,
                lp_solution=root,
                extra_constraints=[],
                gomory_constraints=[],
                depth=0,
                node_id=0,
                parent_id=None,
            )
        ]

        nodes_visited = 0
        while stack and nodes_visited < self.config.max_nodes:
            node = stack.pop()
            nodes_visited += 1

            if collect_trace and len(trace) < self.config.max_trace_nodes:
                trace.append(
                    {
                        "node_id": node.node_id,
                        "parent_id": node.parent_id,
                        "depth": node.depth,
                        "lp_status": node.lp_solution.status,
                        "lp_objective": node.lp_solution.objective,
                        "lp_x": None
                        if node.lp_solution.x is None
                        else [float(node.lp_solution.x[0]), float(node.lp_solution.x[1])],
                        "num_path_constraints": len(node.extra_constraints),
                    }
                )

            lp = node.lp_solution
            if lp.status != "optimal" or lp.x is None or lp.objective is None:
                continue

            if lp.objective <= incumbent_obj + self.config.epsilon:
                continue

            if self._is_integral(lp.x, model.integer_indices):
                incumbent_obj = lp.objective
                incumbent_x = lp.x.copy()
                incumbent_updates += 1
                incumbent_constraints = list(node.extra_constraints)
                incumbent_gomory_constraints = list(node.gomory_constraints)
                continue

            if self.config.use_gomory_cuts:
                lp_after_cuts, cuts_added_here, cut_records = self._apply_gomory_cuts(model=model, lp_solution=lp)
                if cuts_added_here > 0 and lp_after_cuts is not None:
                    node.lp_solution = lp_after_cuts
                    node.state = lp_after_cuts.state if lp_after_cuts.state is not None else node.state
                    node.gomory_constraints.extend(cut_records)
                    lp = node.lp_solution
                    gomory_cuts_added += cuts_added_here
                    if collect_trace:
                        for coeff_full, rhs in cut_records:
                            gomory_lines.append(
                                {
                                    "node_id": node.node_id,
                                    "coeff": [float(v) for v in coeff_full.tolist()],
                                    "rhs": float(rhs),
                                }
                            )
                    if lp.status != "optimal" or lp.x is None or lp.objective is None:
                        continue

                    if lp.objective <= incumbent_obj + self.config.epsilon:
                        continue

                    if self._is_integral(lp.x, model.integer_indices):
                        incumbent_obj = lp.objective
                        incumbent_x = lp.x.copy()
                        incumbent_updates += 1
                        incumbent_constraints = list(node.extra_constraints)
                        incumbent_gomory_constraints = list(node.gomory_constraints)
                        continue

            branch_hint_x: Optional[np.ndarray] = None

            if self.config.use_rounding_heuristic:
                rounded = rounding_heuristic(
                    model=model,
                    lp_solution=lp,
                    extra_constraints=node.extra_constraints,
                    config=self.config,
                )
                if rounded.feasible and rounded.x is not None:
                    branch_hint_x = rounded.x
                    if rounded.objective is not None and rounded.objective > incumbent_obj:
                        incumbent_obj = rounded.objective
                        incumbent_x = rounded.x
                        incumbent_updates += 1
                        incumbent_constraints = list(node.extra_constraints)
                        incumbent_gomory_constraints = list(node.gomory_constraints)

            dived = self._diving_heuristic(model, node)
            if dived is not None and dived[1] > incumbent_obj + self.config.epsilon:
                incumbent_x = dived[0]
                incumbent_obj = dived[1]
                incumbent_updates += 1
                branch_hint_x = dived[0]
                incumbent_constraints = list(dived[2])
                incumbent_gomory_constraints = list(node.gomory_constraints)

            # Heuristic updates can tighten the bound enough to cut this node.
            if lp.objective <= incumbent_obj + self.config.epsilon:
                continue

            branch_index = self._select_branch_variable(lp.x, model.integer_indices)
            if branch_index is None:
                continue

            value = lp.x[branch_index]
            floor_v = float(np.floor(value))
            ceil_v = float(np.ceil(value))

            left_coeff = np.zeros(model.c.shape[0], dtype=float)
            left_coeff[branch_index] = 1.0
            left_rhs = floor_v

            right_coeff = np.zeros(model.c.shape[0], dtype=float)
            right_coeff[branch_index] = -1.0
            right_rhs = -ceil_v

            prefer_left = True
            if branch_hint_x is not None:
                prefer_left = branch_hint_x[branch_index] <= floor_v + self.config.epsilon

            if prefer_left:
                # DFS desired: left first -> push right then left.
                right_child = self._create_child_node(
                    model,
                    node,
                    right_coeff,
                    right_rhs,
                    node_id=next_node_id,
                )
                if right_child is not None:
                    stack.append(right_child)
                    next_node_id += 1
                    if collect_trace:
                        branch_lines.append(
                            {
                                "node_id": right_child.node_id,
                                "coeff": [float(right_coeff[0]), float(right_coeff[1])],
                                "rhs": float(right_rhs),
                            }
                        )

                left_child = self._create_child_node(
                    model,
                    node,
                    left_coeff,
                    left_rhs,
                    node_id=next_node_id,
                )
                if left_child is not None:
                    stack.append(left_child)
                    next_node_id += 1
                    if collect_trace:
                        branch_lines.append(
                            {
                                "node_id": left_child.node_id,
                                "coeff": [float(left_coeff[0]), float(left_coeff[1])],
                                "rhs": float(left_rhs),
                            }
                        )
            else:
                # DFS desired: right first -> push left then right.
                left_child = self._create_child_node(
                    model,
                    node,
                    left_coeff,
                    left_rhs,
                    node_id=next_node_id,
                )
                if left_child is not None:
                    stack.append(left_child)
                    next_node_id += 1
                    if collect_trace:
                        branch_lines.append(
                            {
                                "node_id": left_child.node_id,
                                "coeff": [float(left_coeff[0]), float(left_coeff[1])],
                                "rhs": float(left_rhs),
                            }
                        )

                right_child = self._create_child_node(
                    model,
                    node,
                    right_coeff,
                    right_rhs,
                    node_id=next_node_id,
                )
                if right_child is not None:
                    stack.append(right_child)
                    next_node_id += 1
                    if collect_trace:
                        branch_lines.append(
                            {
                                "node_id": right_child.node_id,
                                "coeff": [float(right_coeff[0]), float(right_coeff[1])],
                                "rhs": float(right_rhs),
                            }
                        )

        if incumbent_x is None:
            return BnBResult(
                status="infeasible",
                objective=None,
                x=None,
                nodes_visited=nodes_visited,
                incumbent_updates=incumbent_updates,
                message="No integer feasible solution found.",
            )

        return BnBResult(
            status="optimal_or_feasible",
            objective=incumbent_obj,
            x=incumbent_x,
            nodes_visited=nodes_visited,
            incumbent_updates=incumbent_updates,
            metadata={
                "max_nodes": self.config.max_nodes,
                "objective_sense": "max",
                "gomory_cuts_added": gomory_cuts_added,
                "trace": trace,
                "branch_lines": branch_lines,
                "gomory_lines": gomory_lines,
                "incumbent_constraints": [
                    {"coeff": [float(v) for v in c.tolist()], "rhs": float(rhs)}
                    for c, rhs in incumbent_constraints
                ],
                "incumbent_gomory_constraints": [
                    {"coeff": [float(v) for v in c.tolist()], "rhs": float(rhs)}
                    for c, rhs in incumbent_gomory_constraints
                ],
            },
        )

    def _create_child_node(
        self,
        model: IntegerModel,
        parent: Node,
        coeff: np.ndarray,
        rhs: float,
        node_id: int,
    ) -> Optional[Node]:
        if parent.state is None:
            return None

        child_lp = self.lp_solver.reoptimize_with_added_constraint(
            state=parent.state,
            coeff=coeff,
            rhs=rhs,
            method="dual",
        )

        if child_lp.status != "optimal" or child_lp.state is None:
            return None

        child_constraints = list(parent.extra_constraints)
        child_constraints.append((coeff, rhs))

        return Node(
            state=child_lp.state,
            lp_solution=child_lp,
            extra_constraints=child_constraints,
            gomory_constraints=list(parent.gomory_constraints),
            depth=parent.depth + 1,
            node_id=node_id,
            parent_id=parent.node_id,
            branch_constraint=(coeff.copy(), rhs),
        )

    def _is_integral(self, x: np.ndarray, indices: List[int]) -> bool:
        for idx in indices:
            if abs(x[idx] - round(float(x[idx]))) > self.config.epsilon:
                return False
        return True

    def _select_branch_variable(self, x: np.ndarray, indices: List[int]) -> Optional[int]:
        best_idx = None
        best_score = -1.0
        for idx in indices:
            frac = abs(x[idx] - round(float(x[idx])))
            if frac > best_score + self.config.epsilon:
                best_score = frac
                best_idx = idx

        if best_idx is None or best_score <= self.config.epsilon:
            return None
        return best_idx

    def _diving_heuristic(
        self,
        model: IntegerModel,
        node: Node,
    ) -> Optional[Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]]:
        best: Optional[Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]] = None
        tries = max(1, self.config.diving_max_tries)

        for trial in range(tries):
            state = node.state.copy()
            lp_solution = node.lp_solution
            constraints = list(node.extra_constraints)

            for _ in range(self.config.diving_max_depth):
                if lp_solution.x is None:
                    break
                if self._is_integral(lp_solution.x, model.integer_indices):
                    if lp_solution.objective is not None:
                        candidate = (lp_solution.x.copy(), lp_solution.objective)
                        if best is None or candidate[1] > best[1] + self.config.epsilon:
                            best = (candidate[0], candidate[1], list(constraints))
                    break

                idx = self._select_branch_variable(lp_solution.x, model.integer_indices)
                if idx is None:
                    break

                value = lp_solution.x[idx]
                frac = value - np.floor(value)
                prefer_floor = frac <= 0.5 if trial % 2 == 0 else frac > 0.5

                coeff = np.zeros(model.c.shape[0], dtype=float)
                if prefer_floor:
                    coeff[idx] = 1.0
                    rhs = float(np.floor(value))
                else:
                    coeff[idx] = -1.0
                    rhs = -float(np.ceil(value))

                lp_solution = self.lp_solver.reoptimize_with_added_constraint(
                    state=state,
                    coeff=coeff,
                    rhs=rhs,
                    method="dual",
                )
                if lp_solution.status != "optimal" or lp_solution.state is None:
                    break

                state = lp_solution.state
                constraints.append((coeff, rhs))

            rounded = rounding_heuristic(
                model=model,
                lp_solution=lp_solution,
                extra_constraints=constraints,
                config=self.config,
            )
            if rounded.feasible and rounded.objective is not None and rounded.x is not None:
                candidate = (rounded.x, rounded.objective)
                if best is None or candidate[1] > best[1] + self.config.epsilon:
                    best = (candidate[0], candidate[1], list(constraints))

        return best

    def _apply_gomory_cuts(
        self,
        model: IntegerModel,
        lp_solution: LPSolution,
    ) -> Tuple[Optional[LPSolution], int, List[Tuple[np.ndarray, float]]]:
        if lp_solution.state is None:
            return None, 0, []

        # Current implementation supports pure-integer models only.
        if sorted(model.integer_indices) != list(range(model.c.shape[0])):
            return None, 0, []

        updated_lp = lp_solution
        cuts_added = 0
        cut_records: List[Tuple[np.ndarray, float]] = []
        max_cuts = max(0, int(self.config.max_gomory_cuts_per_node))
        for _ in range(max_cuts):
            row_index = self._select_gomory_row(updated_lp.state.tableau)
            if row_index is None:
                break

            coeff_full, rhs = self._build_gomory_cut(updated_lp.state.tableau, row_index)
            if coeff_full is None:
                break

            cut_records.append((coeff_full.copy(), rhs))

            next_lp = self.lp_solver.reoptimize_with_added_full_constraint(
                state=updated_lp.state,
                coeff_full=coeff_full,
                rhs=rhs,
                method="dual",
            )
            if next_lp.status != "optimal" or next_lp.state is None:
                return next_lp, cuts_added + 1, cut_records
            updated_lp = next_lp
            cuts_added += 1

        if cuts_added == 0:
            return None, 0, []
        return updated_lp, cuts_added, cut_records

    def _select_gomory_row(self, tableau: np.ndarray) -> Optional[int]:
        rhs = tableau[:-1, -1]
        eps = self.config.epsilon
        best_row = None
        best_frac = eps
        for r, value in enumerate(rhs):
            frac = value - np.floor(value)
            frac = min(frac, 1.0 - frac)
            if frac > best_frac:
                best_frac = frac
                best_row = r
        return best_row

    def _build_gomory_cut(self, tableau: np.ndarray, row_idx: int) -> Tuple[Optional[np.ndarray], float]:
        eps = self.config.epsilon
        row = tableau[row_idx, :-1]
        rhs = float(tableau[row_idx, -1])

        frac_rhs = rhs - np.floor(rhs)
        if frac_rhs <= eps or 1.0 - frac_rhs <= eps:
            return None, 0.0

        frac_coeff = row - np.floor(row)
        frac_coeff[np.abs(frac_coeff) <= eps] = 0.0
        if np.all(np.abs(frac_coeff) <= eps):
            return None, 0.0

        cut_coeff = -frac_coeff
        cut_rhs = -frac_rhs
        return cut_coeff, float(cut_rhs)
