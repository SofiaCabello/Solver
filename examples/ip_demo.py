from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import BranchAndBoundSolver, IntegerModel


def main() -> None:
    # A small integer LP where LP optimum is fractional.
    # max x + y
    # s.t. 2x + y <= 4
    #      x + 2y <= 4
    #      x, y >= 0, integer
    model = IntegerModel(
        c=[5, 3],
        A=[
            [1, 1],
            [15000, 10000],
        ],
        b=[5, 50000],
        integer_indices=[0, 1],
    )

    solver = BranchAndBoundSolver()
    result = solver.solve(model)
    print("status:", result.status)
    print("objective:", result.objective)
    print("x:", result.x)
    print("nodes:", result.nodes_visited)


if __name__ == "__main__":
    main()
