from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from solver.config import SolverConfig
from solver.ip.branch_and_bound import BranchAndBoundSolver
from solver.lp.simplex import LPSolver
from solver.models import IntegerModel, LPModel
from solver.visualization import render_bnb_animation, render_bnb_timeline_figure


def load_problem_from_yaml(
    file_path: str,
) -> Tuple[Union[LPModel, IntegerModel], SolverConfig, Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping.")

    problem = payload.get("problem")
    if not isinstance(problem, dict):
        raise ValueError("Missing required 'problem' mapping.")

    c, A, b, objective_sense = _parse_problem(problem)
    runtime = _parse_runtime(payload.get("config"))
    runtime["objective_sense"] = objective_sense
    solver_config = _build_solver_config(runtime)

    is_integer = bool(runtime.get("is_integer", False))
    integer_indices = runtime.get("integer_indices")
    if is_integer:
        if integer_indices is None:
            integer_indices = list(range(len(c)))
        model = IntegerModel(c=c, A=A, b=b, integer_indices=integer_indices)
    else:
        model = LPModel(c=c, A=A, b=b)

    return model, solver_config, runtime


def solve_from_yaml(file_path: str) -> Dict[str, Any]:
    model, solver_config, runtime = load_problem_from_yaml(file_path)
    objective_sense = str(runtime.get("objective_sense", "max")).lower()
    if isinstance(model, IntegerModel):
        solver = BranchAndBoundSolver(config=solver_config)
        result = solver.solve(model, objective_sense=objective_sense)
        clean_x = None if result.x is None else _clean_vector(result.x.tolist(), solver_config.epsilon)
        clean_obj = None if result.objective is None else _clean_number(result.objective, solver_config.epsilon)
        output = {
            "mode": "ip",
            "status": result.status,
            "objective": clean_obj,
            "x": clean_x,
            "nodes_visited": result.nodes_visited,
            "incumbent_updates": result.incumbent_updates,
            "message": result.message,
            "metadata": result.metadata,
            "solver_config": asdict(solver_config),
            "objective_sense": objective_sense,
        }

        if solver_config.visualize:
            output["visualization"] = _build_visualization_output(
                yaml_path=file_path,
                model=model,
                result_metadata=result.metadata,
                clean_x=clean_x,
                solver_config=solver_config,
            )
        return output

    lp_method = runtime.get("lp_method", "primal")
    if lp_method == "primal" and any(float(v) < -solver_config.epsilon for v in model.b):
        lp_method = "dual"
    solver = LPSolver(config=solver_config)
    result = solver.solve(model, method=lp_method, objective_sense=objective_sense)
    clean_x = None if result.x is None else _clean_vector(result.x.tolist(), solver_config.epsilon)
    clean_obj = None if result.objective is None else _clean_number(result.objective, solver_config.epsilon)
    output = {
        "mode": "lp",
        "status": result.status,
        "objective": clean_obj,
        "x": clean_x,
        "iterations": result.iterations,
        "message": result.message,
        "solver_config": asdict(solver_config),
        "lp_method": lp_method,
        "objective_sense": objective_sense,
    }
    return output


def _parse_problem(problem: Dict[str, Any]) -> Tuple[List[float], List[List[float]], List[float], str]:
    if "objective" in problem:
        return _parse_structured_problem(problem)

    # Legacy compact format:
    # problem:
    #   c: [...]
    #   A: [[...], ...]
    #   b: [...]
    c = problem.get("c")
    A = problem.get("A")
    b = problem.get("b")
    if c is None or A is None or b is None:
        raise ValueError("Problem must define either objective/constraints or c/A/b.")
    compact_sense = str(problem.get("sense", "max")).lower()
    if compact_sense not in ("max", "min"):
        raise ValueError("problem.sense must be one of: max, min")
    return (
        _as_float_list(c, "problem.c"),
        _as_float_matrix(A, "problem.A"),
        _as_float_list(b, "problem.b"),
        compact_sense,
    )


def _parse_structured_problem(problem: Dict[str, Any]) -> Tuple[List[float], List[List[float]], List[float], str]:
    objective = problem.get("objective")
    constraints = problem.get("constraints")

    if not isinstance(objective, dict):
        raise ValueError("problem.objective must be a mapping.")
    if not isinstance(constraints, list) or len(constraints) == 0:
        raise ValueError("problem.constraints must be a non-empty list.")

    sense = str(objective.get("sense", "max")).lower()
    if sense not in ("max", "min"):
        raise ValueError("problem.objective.sense must be one of: max, min")

    c = _as_float_list(objective.get("coefficients"), "problem.objective.coefficients")
    n = len(c)

    A: List[List[float]] = []
    b: List[float] = []
    for i, row in enumerate(constraints):
        if not isinstance(row, dict):
            raise ValueError(f"problem.constraints[{i}] must be a mapping.")
        row_sense = str(row.get("sense", "<=")).strip()
        if row_sense not in ("<=", ">=", "==", "="):
            raise ValueError("Constraint sense must be one of: <=, >=, ==")

        coeff = _as_float_list(row.get("coefficients"), f"problem.constraints[{i}].coefficients")
        if len(coeff) != n:
            raise ValueError(
                f"Constraint {i} coefficient length mismatch: expected {n}, got {len(coeff)}."
            )
        rhs = float(row.get("rhs"))

        if row_sense == "<=":
            A.append(coeff)
            b.append(rhs)
        elif row_sense == ">=":
            A.append([-v for v in coeff])
            b.append(-rhs)
        else:
            A.append(coeff)
            b.append(rhs)
            A.append([-v for v in coeff])
            b.append(-rhs)

    return c, A, b, sense


def _parse_runtime(config_payload: Any) -> Dict[str, Any]:
    if config_payload is None:
        return {}
    if not isinstance(config_payload, dict):
        raise ValueError("config must be a mapping.")

    runtime: Dict[str, Any] = {}
    runtime["is_integer"] = bool(
        config_payload.get(
            "is_integer",
            config_payload.get("integer", config_payload.get("integer_optimization", False)),
        )
    )
    runtime["visualize"] = bool(
        config_payload.get(
            "visualize",
            config_payload.get("enable_visualization", config_payload.get("visualization", False)),
        )
    )

    integer_indices = config_payload.get("integer_indices")
    if integer_indices is not None:
        if not isinstance(integer_indices, list):
            raise ValueError("config.integer_indices must be a list of non-negative indices.")
        runtime["integer_indices"] = [int(idx) for idx in integer_indices]

    lp_method = str(config_payload.get("lp_method", "primal"))
    if lp_method not in ("primal", "dual"):
        raise ValueError("config.lp_method must be one of: primal, dual")
    runtime["lp_method"] = lp_method

    for name in (
        "epsilon",
        "max_iterations",
        "max_nodes",
        "diving_max_depth",
        "diving_max_tries",
        "use_rounding_heuristic",
        "rounding_max_repair_steps",
        "use_gomory_cuts",
        "max_gomory_cuts_per_node",
        "visualization_output",
        "visualization_timeline_output",
        "visualization_generate_timeline",
        "visualization_timeline_panels",
        "visualization_fps",
        "visualization_grid_size",
        "max_trace_nodes",
    ):
        if name in config_payload:
            runtime[name] = config_payload[name]

    return runtime


def _build_solver_config(runtime: Dict[str, Any]) -> SolverConfig:
    kwargs: Dict[str, Any] = {}
    if "epsilon" in runtime:
        kwargs["epsilon"] = float(runtime["epsilon"])
    if "max_iterations" in runtime:
        kwargs["max_iterations"] = int(runtime["max_iterations"])
    if "max_nodes" in runtime:
        kwargs["max_nodes"] = int(runtime["max_nodes"])
    if "diving_max_depth" in runtime:
        kwargs["diving_max_depth"] = int(runtime["diving_max_depth"])
    if "diving_max_tries" in runtime:
        kwargs["diving_max_tries"] = int(runtime["diving_max_tries"])
    if "use_rounding_heuristic" in runtime:
        kwargs["use_rounding_heuristic"] = bool(runtime["use_rounding_heuristic"])
    if "rounding_max_repair_steps" in runtime:
        kwargs["rounding_max_repair_steps"] = int(runtime["rounding_max_repair_steps"])
    if "use_gomory_cuts" in runtime:
        kwargs["use_gomory_cuts"] = bool(runtime["use_gomory_cuts"])
    if "max_gomory_cuts_per_node" in runtime:
        kwargs["max_gomory_cuts_per_node"] = int(runtime["max_gomory_cuts_per_node"])
    if "visualize" in runtime:
        kwargs["visualize"] = bool(runtime["visualize"])
    if "visualization_output" in runtime:
        kwargs["visualization_output"] = str(runtime["visualization_output"])
    if "visualization_timeline_output" in runtime:
        kwargs["visualization_timeline_output"] = str(runtime["visualization_timeline_output"])
    if "visualization_generate_timeline" in runtime:
        kwargs["visualization_generate_timeline"] = bool(runtime["visualization_generate_timeline"])
    if "visualization_timeline_panels" in runtime:
        kwargs["visualization_timeline_panels"] = int(runtime["visualization_timeline_panels"])
    if "visualization_fps" in runtime:
        kwargs["visualization_fps"] = int(runtime["visualization_fps"])
    if "visualization_grid_size" in runtime:
        kwargs["visualization_grid_size"] = int(runtime["visualization_grid_size"])
    if "max_trace_nodes" in runtime:
        kwargs["max_trace_nodes"] = int(runtime["max_trace_nodes"])
    return SolverConfig(**kwargs)


def _as_float_list(value: Any, path: str) -> List[float]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{path} must be a non-empty list.")
    return [float(v) for v in value]


def _as_float_matrix(value: Any, path: str) -> List[List[float]]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{path} must be a non-empty 2D list.")
    matrix: List[List[float]] = []
    width = None
    for i, row in enumerate(value):
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError(f"{path}[{i}] must be a non-empty list.")
        casted = [float(v) for v in row]
        if width is None:
            width = len(casted)
        elif len(casted) != width:
            raise ValueError(f"{path} must be rectangular.")
        matrix.append(casted)
    return matrix


def _clean_number(value: float, epsilon: float) -> float:
    if abs(value) <= epsilon:
        return 0.0

    nearest_int = round(value)
    if abs(value - nearest_int) <= epsilon:
        return float(nearest_int)

    return float(value)


def _clean_vector(values: List[float], epsilon: float) -> List[float]:
    return [_clean_number(v, epsilon) for v in values]


def _build_visualization_output(
    yaml_path: str,
    model: IntegerModel,
    result_metadata: Dict[str, Any],
    clean_x: Optional[List[float]],
    solver_config: SolverConfig,
) -> Dict[str, Any]:
    if model.c.shape[0] != 2:
        return {
            "enabled": False,
            "message": "Visualization is only supported for 2-variable problems.",
        }

    trace = result_metadata.get("trace", [])
    branch_lines = result_metadata.get("branch_lines", [])
    gomory_lines = result_metadata.get("gomory_lines", [])
    incumbent_constraints = result_metadata.get("incumbent_constraints", [])
    incumbent_gomory_constraints = result_metadata.get("incumbent_gomory_constraints", [])
    if not trace:
        return {
            "enabled": False,
            "message": "No branch-and-bound traversal trace available for visualization.",
        }

    output_path = _resolve_visualization_output_path(yaml_path, solver_config.visualization_output)
    timeline_output_path = _resolve_visualization_output_path(
        yaml_path,
        solver_config.visualization_timeline_output,
    )
    try:
        saved = render_bnb_animation(
            c=model.c.tolist(),
            A=model.A.tolist(),
            b=model.b.tolist(),
            trace=trace,
            branch_lines=branch_lines,
            gomory_lines=gomory_lines,
            incumbent_constraints=incumbent_constraints,
            incumbent_gomory_constraints=incumbent_gomory_constraints,
            incumbent_x=clean_x,
            output_path=output_path,
            fps=solver_config.visualization_fps,
            grid_size=solver_config.visualization_grid_size,
            epsilon=solver_config.epsilon,
        )
        timeline_saved = None
        if solver_config.visualization_generate_timeline:
            timeline_saved = render_bnb_timeline_figure(
                c=model.c.tolist(),
                A=model.A.tolist(),
                b=model.b.tolist(),
                trace=trace,
                branch_lines=branch_lines,
                gomory_lines=gomory_lines,
                incumbent_constraints=incumbent_constraints,
                incumbent_gomory_constraints=incumbent_gomory_constraints,
                incumbent_x=clean_x,
                output_path=timeline_output_path,
                panel_count=solver_config.visualization_timeline_panels,
                grid_size=solver_config.visualization_grid_size,
                epsilon=solver_config.epsilon,
            )
        return {
            "enabled": True,
            "output_file": saved,
            "timeline_output_file": timeline_saved,
            "frames": len(trace),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "message": f"Visualization generation failed: {exc}",
        }


def _resolve_visualization_output_path(yaml_path: str, configured_output: str) -> str:
    out_path = Path(configured_output)
    if out_path.is_absolute():
        return str(out_path)
    base = Path(yaml_path).resolve().parent
    return str(base / out_path)
