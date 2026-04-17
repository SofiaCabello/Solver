from pathlib import Path
import argparse
import json
import sys
import time

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

    exact_start = time.perf_counter()
    exact = solve_from_yaml(args.yaml_file)
    exact_elapsed_ms = (time.perf_counter() - exact_start) * 1000.0

    model, cfg, runtime = load_problem_from_yaml(args.yaml_file)
    if not isinstance(model, IntegerModel):
        raise ValueError("Comparison tool currently targets integer models.")

    sense = str(runtime.get("objective_sense", "max"))

    ga = GeneticAlgorithmIPSolver(epsilon=cfg.epsilon)
    ga_start = time.perf_counter()
    ga_result = ga.solve(model, objective_sense=sense)
    ga_elapsed_ms = (time.perf_counter() - ga_start) * 1000.0

    greedy = GreedyIPSolver(epsilon=cfg.epsilon)
    greedy_start = time.perf_counter()
    greedy_result = greedy.solve(model, objective_sense=sense)
    greedy_elapsed_ms = (time.perf_counter() - greedy_start) * 1000.0

    payload = {
        "exact": {
            **exact,
            "time_ms": exact_elapsed_ms,
        },
        "ga": {
            "status": ga_result.status,
            "objective": ga_result.objective,
            "x": None if ga_result.x is None else ga_result.x.tolist(),
            "generations": ga_result.generations,
            "message": ga_result.message,
            "metadata": {
                **ga_result.metadata,
                "compare_time_ms": ga_elapsed_ms,
            },
            "time_ms": ga_elapsed_ms,
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
                "compare_time_ms": greedy_elapsed_ms,
            },
            "time_ms": greedy_elapsed_ms,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
