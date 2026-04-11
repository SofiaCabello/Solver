from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Cut:
    coeff: np.ndarray
    rhs: float
    name: str = "generic_cut"


class CutGenerator:
    """Placeholder for future Gomory/MIR/etc. cut generation."""

    def generate(self, x_lp: np.ndarray) -> List[Cut]:
        _ = x_lp
        return []


def cuts_to_constraints(cuts: List[Cut]) -> List[Tuple[np.ndarray, float]]:
    return [(cut.coeff, cut.rhs) for cut in cuts]
