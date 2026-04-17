from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from solver.models import IntegerModel


@dataclass
class GASolution:
    status: str
    objective: Optional[float]
    x: Optional[np.ndarray]
    generations: int
    message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class GeneticAlgorithmIPSolver:
    def __init__(
        self,
        population_size: int = 180,
        generations: int = 320,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.12,
        elite_size: int = 6,
        penalty_weight: float = 10_000.0,
        seed: int = 42,
        default_upper_bound: int = 8,
        epsilon: float = 1e-9,
    ) -> None:
        self.population_size = max(10, int(population_size))
        self.generations = max(1, int(generations))
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elite_size = max(1, int(elite_size))
        self.penalty_weight = float(penalty_weight)
        self.seed = int(seed)
        self.default_upper_bound = max(1, int(default_upper_bound))
        self.epsilon = float(epsilon)

    def solve(self, model: IntegerModel, objective_sense: str = "max") -> GASolution:
        sense = objective_sense.lower()
        if sense not in ("max", "min"):
            raise ValueError("objective_sense must be one of: max, min")

        n = model.c.shape[0]
        A = np.asarray(model.A, dtype=float)
        b = np.asarray(model.b, dtype=float)
        c = np.asarray(model.c, dtype=float)

        rng = np.random.default_rng(self.seed)
        upper_bounds = self._infer_upper_bounds(A, b, n)

        population = self._init_population(rng, self.population_size, upper_bounds)
        best_x = None
        best_obj = None
        best_violation = float("inf")

        for gen in range(self.generations):
            fitness, objective, violations = self._evaluate_population(population, A, b, c, sense)

            feasible_idx = np.where(violations <= self.epsilon)[0]
            if feasible_idx.size > 0:
                idx = self._best_feasible_index(objective, feasible_idx, sense)
                obj = float(objective[idx])
                if self._is_better(obj, best_obj, sense):
                    best_obj = obj
                    best_x = population[idx].copy()

            min_v_idx = int(np.argmin(violations))
            if float(violations[min_v_idx]) < best_violation:
                best_violation = float(violations[min_v_idx])

            elite_idx = np.argsort(-fitness)[: self.elite_size]
            next_pop = [population[i].copy() for i in elite_idx]

            while len(next_pop) < self.population_size:
                p1 = population[self._tournament_select(rng, fitness)]
                p2 = population[self._tournament_select(rng, fitness)]
                c1, c2 = self._crossover(rng, p1, p2)
                self._mutate(rng, c1, upper_bounds)
                self._mutate(rng, c2, upper_bounds)
                next_pop.append(c1)
                if len(next_pop) < self.population_size:
                    next_pop.append(c2)

            population = np.asarray(next_pop, dtype=int)

        if best_x is not None and best_obj is not None:
            return GASolution(
                status="optimal_or_feasible",
                objective=float(best_obj),
                x=best_x.astype(float),
                generations=self.generations,
                metadata={
                    "objective_sense": sense,
                    "best_violation": 0.0,
                    "population_size": self.population_size,
                    "seed": self.seed,
                },
            )

        fitness, objective, violations = self._evaluate_population(population, A, b, c, sense)
        idx = int(np.argmin(violations))
        return GASolution(
            status="feasible_not_found",
            objective=float(objective[idx]),
            x=population[idx].astype(float),
            generations=self.generations,
            message="No strictly feasible integer solution found; returning least-violation individual.",
            metadata={
                "objective_sense": sense,
                "best_violation": float(violations[idx]),
                "population_size": self.population_size,
                "seed": self.seed,
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

    def _init_population(self, rng: np.random.Generator, pop_size: int, ub: np.ndarray) -> np.ndarray:
        n = ub.shape[0]
        pop = np.zeros((pop_size, n), dtype=int)
        for j in range(n):
            pop[:, j] = rng.integers(0, ub[j] + 1, size=pop_size)
        return pop

    def _evaluate_population(
        self,
        population: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        sense: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        objective = population @ c
        lhs = population @ A.T
        violations = np.maximum(lhs - b, 0.0).sum(axis=1)

        if sense == "max":
            base = objective
        else:
            base = -objective

        fitness = base - self.penalty_weight * violations
        return fitness.astype(float), objective.astype(float), violations.astype(float)

    def _best_feasible_index(self, objective: np.ndarray, feasible_idx: np.ndarray, sense: str) -> int:
        feasible_obj = objective[feasible_idx]
        if sense == "max":
            loc = int(np.argmax(feasible_obj))
        else:
            loc = int(np.argmin(feasible_obj))
        return int(feasible_idx[loc])

    def _is_better(self, obj: float, best_obj: Optional[float], sense: str) -> bool:
        if best_obj is None:
            return True
        if sense == "max":
            return obj > best_obj + self.epsilon
        return obj < best_obj - self.epsilon

    def _tournament_select(self, rng: np.random.Generator, fitness: np.ndarray, k: int = 3) -> int:
        idx = rng.integers(0, fitness.shape[0], size=max(2, k))
        return int(idx[np.argmax(fitness[idx])])

    def _crossover(
        self,
        rng: np.random.Generator,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        c1 = p1.copy()
        c2 = p2.copy()
        if rng.random() > self.crossover_rate:
            return c1, c2

        mask = rng.integers(0, 2, size=p1.shape[0]).astype(bool)
        c1[mask], c2[mask] = p2[mask], p1[mask]
        return c1, c2

    def _mutate(self, rng: np.random.Generator, x: np.ndarray, ub: np.ndarray) -> None:
        for j in range(x.shape[0]):
            if rng.random() >= self.mutation_rate:
                continue
            if rng.random() < 0.5:
                x[j] = int(rng.integers(0, ub[j] + 1))
            else:
                delta = int(rng.choice([-1, 1]))
                x[j] = int(np.clip(x[j] + delta, 0, ub[j]))
