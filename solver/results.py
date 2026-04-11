from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SimplexState:
    tableau: np.ndarray
    basis: list[int]
    n_original: int

    def copy(self) -> "SimplexState":
        return SimplexState(
            tableau=self.tableau.copy(),
            basis=list(self.basis),
            n_original=self.n_original,
        )


@dataclass
class LPSolution:
    status: str
    objective: Optional[float]
    x: Optional[np.ndarray]
    state: Optional[SimplexState] = None
    iterations: int = 0
    message: str = ""


@dataclass
class BnBResult:
    status: str
    objective: Optional[float]
    x: Optional[np.ndarray]
    nodes_visited: int
    incumbent_updates: int
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
