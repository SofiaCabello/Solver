from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import (
    GeneticAlgorithmIPSolver,
    GreedyIPSolver,
    IntegerModel,
    load_problem_from_yaml,
    solve_from_yaml,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact solver vs GA vs greedy on one YAML model")
    parser.add_argument("yaml_file", help="Path to YAML model")
    args = parser.parse_args()

    exact = solve_from_yaml(args.yaml_file)

    model, cfg, runtime = load_problem_from_yaml(args.yaml_file)
    if not isinstance(model, IntegerModel):
        raise ValueError("Comparison tool currently targets integer models.")

    sense = str(runtime.get("objective_sense", "max"))

    ga = GeneticAlgorithmIPSolver(epsilon=cfg.epsilon)
    ga_result = ga.solve(model, objective_sense=sense)

    greedy = GreedyIPSolver(epsilon=cfg.epsilon)
    greedy_result = greedy.solve(model, objective_sense=sense)

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
        "greedy": {
            "status": greedy_result.status,
            "objective": greedy_result.objective,
            "x": None if greedy_result.x is None else greedy_result.x.tolist(),
            "steps": greedy_result.steps,
            "message": greedy_result.message,
            "metadata": {
                "steps": greedy_result.metadata.get("steps"),
                "candidate_evaluations": greedy_result.metadata.get("candidate_evaluations"),
                "time_ms": greedy_result.metadata.get("time_ms"),
                "final_violation": greedy_result.metadata.get("final_violation"),
            },
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
