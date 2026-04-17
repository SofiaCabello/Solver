from pathlib import Path
import argparse
import json
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import GeneticAlgorithmIPSolver, IntegerModel, load_problem_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve integer model from YAML via Genetic Algorithm baseline")
    parser.add_argument("yaml_file", help="Path to YAML model")
    parser.add_argument("--population", type=int, default=None, help="GA population size override")
    parser.add_argument("--generations", type=int, default=None, help="GA generations override")
    parser.add_argument("--seed", type=int, default=None, help="GA random seed override")
    args = parser.parse_args()

    model, cfg, runtime = load_problem_from_yaml(args.yaml_file)
    if not isinstance(model, IntegerModel):
        raise ValueError("GA baseline currently supports integer models only.")

    user_cfg = _read_config(args.yaml_file)
    ga = GeneticAlgorithmIPSolver(
        population_size=args.population or int(user_cfg.get("ga_population_size", 180)),
        generations=args.generations or int(user_cfg.get("ga_generations", 320)),
        mutation_rate=float(user_cfg.get("ga_mutation_rate", 0.12)),
        crossover_rate=float(user_cfg.get("ga_crossover_rate", 0.9)),
        elite_size=int(user_cfg.get("ga_elite_size", 6)),
        penalty_weight=float(user_cfg.get("ga_penalty_weight", 10_000.0)),
        seed=args.seed if args.seed is not None else int(user_cfg.get("ga_seed", 42)),
        epsilon=cfg.epsilon,
    )

    sense = str(runtime.get("objective_sense", "max"))
    result = ga.solve(model, objective_sense=sense)
    payload = {
        "mode": "ip-ga",
        "status": result.status,
        "objective_sense": sense,
        "objective": result.objective,
        "x": None if result.x is None else result.x.tolist(),
        "generations": result.generations,
        "message": result.message,
        "metadata": result.metadata,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _read_config(yaml_file: str) -> dict:
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and isinstance(data.get("config"), dict):
        return data["config"]
    return {}


if __name__ == "__main__":
    main()
