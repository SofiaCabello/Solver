from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import LPModel, LPSolver


def main() -> None:
    model = LPModel(
        c=[3, 2],
        A=[
            [2, 1],
            [2, 3],
            [3, 1],
        ],
        b=[18, 42, 24],
    )
    solver = LPSolver()
    result = solver.solve(model, method="primal")
    print("status:", result.status)
    print("objective:", result.objective)
    print("x:", result.x)


if __name__ == "__main__":
    main()
