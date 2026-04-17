from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import GreedyIPSolver, IntegerModel, load_problem_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve integer model from YAML via greedy baseline")
    parser.add_argument("yaml_file", help="Path to YAML model")
    parser.add_argument("--max-steps", type=int, default=20000, help="Maximum greedy search steps")
    parser.add_argument("--ub", type=int, default=8, help="Default variable upper bound fallback")
    args = parser.parse_args()

    model, cfg, runtime = load_problem_from_yaml(args.yaml_file)
    if not isinstance(model, IntegerModel):
        raise ValueError("Greedy baseline currently supports integer models only.")

    greedy = GreedyIPSolver(
        epsilon=cfg.epsilon,
        max_steps=args.max_steps,
        default_upper_bound=args.ub,
        record_steps=True,
    )
    sense = str(runtime.get("objective_sense", "max"))
    result = greedy.solve(model, objective_sense=sense)

    payload = {
        "mode": "ip-greedy",
        "status": result.status,
        "objective_sense": sense,
        "objective": result.objective,
        "x": None if result.x is None else result.x.tolist(),
        "steps": result.steps,
        "message": result.message,
        "metadata": result.metadata,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
