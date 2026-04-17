# Solver: LP/IP Optimization Engine in Python

This project implements a lightweight and extensible LP/IP solver with YAML-driven input, branch-and-bound, Gomory fractional cuts, and optional 2D search visualization.

## Features

### LP (Linear Programming)

- Primal Simplex
- Dual Simplex (warm-start re-optimization)
- Objective sense support: max and min
- Epsilon-based floating-point tolerance handling
- Basis operations utility:
  - Inverse solve (baseline)
  - LU solve (stability-oriented path)

### IP (Integer Programming)

- Branch and Bound (DFS node strategy)
- Objective sense support: max and min
- Primal heuristics:
  - Rounding heuristic
  - Diving heuristic
- Gomory Fractional Cut (current implementation targets pure-integer models)
- Genetic Algorithm baseline solver (for comparison/control experiments)
- Greedy baseline solver with exploration-step accounting

### Visualization (2D IP only)

When enabled, the solver stores traversal history and renders a GIF animation including:

- Node-by-node branch-and-bound traversal
- Branch constraints
- Gomory cut planes (x1-x2 projection)
- Final feasible region overlay
- Final incumbent marker
- Paper-style timeline figure (left-to-right temporal panels)

## Requirements

- Python 3.9+
- Dependencies listed in requirements.txt

Install:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from solver import LPModel, IntegerModel, LPSolver, BranchAndBoundSolver

lp = LPModel(
    c=[3, 2],
    A=[
        [2, 1],
        [2, 3],
        [3, 1],
    ],
    b=[18, 42, 24],
)
lp_result = LPSolver().solve(lp, method="primal")
print(lp_result.status, lp_result.objective, lp_result.x)

ip = IntegerModel(
    c=lp.c,
    A=lp.A,
    b=lp.b,
    integer_indices=[0, 1],
)
ip_result = BranchAndBoundSolver().solve(ip)
print(ip_result.status, ip_result.objective, ip_result.x)
```

## YAML I/O

Solve from YAML:

```bash
python examples/solve_yaml.py examples/problem_lp.yaml
python examples/solve_yaml.py examples/problem_ip.yaml
python examples/solve_yaml.py examples/problem_lp_6vars.yaml
python examples/solve_yaml.py examples/problem_ip_6vars.yaml
```

GA baseline solve from YAML:

```bash
python examples/solve_yaml_ga.py examples/problem_ip_stress_test.yaml
```

Greedy baseline solve from YAML:

```bash
python examples/solve_yaml_greedy.py examples/problem_ip_stress_test.yaml
```

Exact vs GA comparison on the same problem:

```bash
python examples/compare_exact_vs_ga.py examples/problem_ip_stress_test.yaml
```

Exact vs GA vs Greedy comparison:

```bash
python examples/compare_exact_ga_greedy.py examples/problem_ip_stress_test.yaml
```

Prepared benchmark/problem files:

- examples/problem_lp_6vars.yaml
- examples/problem_ip_6vars.yaml
- examples/problem_ip_vis.yaml

### YAML Schema (structured)

```yaml
problem:
  objective:
    sense: max  # max or min
    coefficients: [3, 2]
  constraints:
    - coefficients: [2, 1]
      sense: <=   # <=, >=, ==
      rhs: 18

config:
  is_integer: false
  integer_indices: [0, 1]
  lp_method: primal         # primal or dual
  epsilon: 1.0e-9
  max_iterations: 10000
  max_nodes: 50000
  diving_max_depth: 20
  diving_max_tries: 2
  use_rounding_heuristic: true
  rounding_max_repair_steps: 100
  use_gomory_cuts: true
  max_gomory_cuts_per_node: 1
  visualize: false
  visualization_output: outputs/bnb_animation.gif
  visualization_timeline_output: outputs/bnb_timeline.png
  visualization_generate_timeline: true
  visualization_timeline_panels: 6
  visualization_fps: 2
  visualization_grid_size: 160
  max_trace_nodes: 8000
```

### YAML Schema (compact)

```yaml
problem:
  c: [5, 3]
  A:
    - [1, 1]
    - [15000, 10000]
  b: [5, 50000]
config:
  is_integer: true
  integer_indices: [0, 1]
```

## Visualization Usage

```bash
python examples/solve_yaml.py examples/problem_ip_vis.yaml
```

Default output file:

- examples/outputs/problem_ip_vis_bnb.gif
- examples/outputs/problem_ip_vis_timeline.png

Note:

- Visualization is supported only when IP has exactly 2 decision variables.
- Gomory lines are shown as x1-x2 projection. Some high-dimensional cuts may appear weak in 2D view if they mainly act on slack/extended tableau variables.
- For minimization models, solver uses equivalent transformed maximization internally and maps objective back.
- When visualization_generate_timeline is true, solver emits a static paper-ready timeline figure with panels ordered left-to-right by step index.

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Current Scope

- Canonical internal form: max c^T x, Ax <= b, x >= 0.
- YAML parser accepts objective sense max/min and constraint senses <=, >=, == (normalized to internal <= form).
- Branch constraints are appended and re-optimized by dual simplex.
- Gomory fractional cuts are integrated for pure-integer models.
- Mixed-integer Gomory cuts and richer cut families are future extension points.

## GA Baseline Notes

- GA solver currently targets integer models and uses penalty-based constraint handling.
- Recommended for comparison experiments, stress tests, and heuristic benchmarking.
- For deterministic reproduction, set ga_seed in YAML config or pass --seed to solve_yaml_ga.py.

## Greedy Baseline Notes

- Greedy solver targets integer models and is intended as a fast control method.
- It records exploration consumption in metadata:
  - steps
  - candidate_evaluations
  - time_ms
  - final_violation
  - step_trace (step-by-step selection log)
