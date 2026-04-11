from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np


def _as_float_vector(values: Union[List[float], np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D vector.")
    return arr


def _as_float_matrix(values: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    return arr


@dataclass
class LPModel:
    c: Union[List[float], np.ndarray]
    A: Union[List[List[float]], np.ndarray]
    b: Union[List[float], np.ndarray]

    def __post_init__(self) -> None:
        self.c = _as_float_vector(self.c)
        self.A = _as_float_matrix(self.A)
        self.b = _as_float_vector(self.b)

        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A row count must match b length.")
        if self.A.shape[1] != self.c.shape[0]:
            raise ValueError("A column count must match c length.")


@dataclass
class IntegerModel(LPModel):
    integer_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        n = self.c.shape[0]
        for idx in self.integer_indices:
            if idx < 0 or idx >= n:
                raise ValueError(f"Integer index out of bounds: {idx}")
