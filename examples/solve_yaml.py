from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import solve_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve LP/IP model from YAML file")
    parser.add_argument("yaml_file", help="Path to YAML model definition")
    args = parser.parse_args()

    start = time.perf_counter()
    result = solve_from_yaml(args.yaml_file)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if isinstance(result, dict):
        payload = {**result, "time_ms": elapsed_ms}
    else:
        payload = {"result": result, "time_ms": elapsed_ms}

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
