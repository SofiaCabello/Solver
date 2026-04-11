# Solver: LP/IP Optimization Engine in Python

This project implements a lightweight and extensible LP/IP solver with YAML-driven input, branch-and-bound, Gomory fractional cuts, and optional 2D search visualization.

## Features

### LP (Linear Programming)

- Primal Simplex
- Dual Simplex (warm-start re-optimization)
- Epsilon-based floating-point tolerance handling
- Basis operations utility:
  - Inverse solve (baseline)
  - LU solve (stability-oriented path)

### IP (Integer Programming)

- Branch and Bound (DFS node strategy)
- Primal heuristics:
  - Rounding heuristic
  - Diving heuristic
- Gomory Fractional Cut (current implementation targets pure-integer models)

### Visualization (2D IP only)

When enabled, the solver stores traversal history and renders a GIF animation including:

- Node-by-node branch-and-bound traversal
- Branch constraints
- Gomory cut planes (x1-x2 projection)
- Final feasible region overlay
- Final incumbent marker

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

Prepared benchmark/problem files:

- examples/problem_lp_6vars.yaml
- examples/problem_ip_6vars.yaml
- examples/problem_ip_vis.yaml

### YAML Schema (structured)

```yaml
problem:
  objective:
    sense: max
    coefficients: [3, 2]
  constraints:
    - coefficients: [2, 1]
      sense: <=
      rhs: 18

config:
  is_integer: false
  integer_indices: [0, 1]
  lp_method: primal
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

Note:

- Visualization is supported only when IP has exactly 2 decision variables.
- Gomory lines are shown as x1-x2 projection. Some high-dimensional cuts may appear weak in 2D view if they mainly act on slack/extended tableau variables.

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Current Scope

- Canonical model form: max c^T x, Ax <= b, x >= 0.
- Branch constraints are appended and re-optimized by dual simplex.
- Gomory fractional cuts are integrated for pure-integer models.
- Mixed-integer Gomory cuts and richer cut families are future extension points.
