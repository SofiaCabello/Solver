from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import GeneticAlgorithmIPSolver, IntegerModel, load_problem_from_yaml, solve_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact BnB solver vs GA baseline on the same YAML model")
    parser.add_argument("yaml_file", help="Path to YAML model")
    args = parser.parse_args()

    exact = solve_from_yaml(args.yaml_file)

    model, cfg, runtime = load_problem_from_yaml(args.yaml_file)
    if not isinstance(model, IntegerModel):
        raise ValueError("Comparison tool currently targets integer models.")

    ga = GeneticAlgorithmIPSolver(epsilon=cfg.epsilon)
    ga_result = ga.solve(model, objective_sense=str(runtime.get("objective_sense", "max")))

    payload = {
        "exact": exact,
        "ga": {
            "status": ga_result.status,
            "objective": ga_result.objective,
            "x": None if ga_result.x is None else ga_result.x.tolist(),
            "generations": ga_result.generations,
            "message": ga_result.message,
            "metadata": ga_result.metadata,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
