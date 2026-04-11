from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BasisFactorization:
    method: str = "inverse"
    epsilon: float = 1e-12

    def solve(self, B: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if self.method == "inverse":
            inv = np.linalg.inv(B)
            return inv @ rhs
        if self.method == "lu":
            L, U, piv = self._lu_decompose(B)
            return self._lu_solve(L, U, piv, rhs)
        raise ValueError("method must be one of: inverse, lu")

    def _lu_decompose(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = A.shape[0]
        U = A.copy().astype(float)
        L = np.eye(n, dtype=float)
        piv = np.arange(n)

        for k in range(n - 1):
            pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
            if abs(U[pivot_row, k]) <= self.epsilon:
                raise np.linalg.LinAlgError("Singular matrix in LU decomposition.")

            if pivot_row != k:
                U[[k, pivot_row], :] = U[[pivot_row, k], :]
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
                piv[[k, pivot_row]] = piv[[pivot_row, k]]

            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]

        if abs(U[-1, -1]) <= self.epsilon:
            raise np.linalg.LinAlgError("Singular matrix in LU decomposition.")

        return L, U, piv

    def _lu_solve(self, L: np.ndarray, U: np.ndarray, piv: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        b = rhs[piv]
        n = b.shape[0]

        y = np.zeros(n, dtype=float)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])

        x = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) <= self.epsilon:
                raise np.linalg.LinAlgError("Singular upper triangular matrix.")
            x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

        return x
