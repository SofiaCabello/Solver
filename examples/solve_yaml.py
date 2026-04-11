from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import solve_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve LP/IP model from YAML file")
    parser.add_argument("yaml_file", help="Path to YAML model definition")
    args = parser.parse_args()

    result = solve_from_yaml(args.yaml_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
