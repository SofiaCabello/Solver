from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def render_bnb_animation(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    trace: List[Dict[str, Any]],
    branch_lines: List[Dict[str, Any]],
    gomory_lines: List[Dict[str, Any]],
    incumbent_constraints: List[Dict[str, Any]],
    incumbent_gomory_constraints: List[Dict[str, Any]],
    incumbent_x: Optional[Sequence[float]],
    output_path: str,
    fps: int = 2,
    grid_size: int = 160,
    epsilon: float = 1e-9,
) -> str:
    if len(c) != 2:
        raise ValueError("Visualization supports 2 variables only.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    points = [t["lp_x"] for t in trace if t.get("lp_x") is not None]
    xlim, ylim = _infer_plot_bounds(A_np, b_np, points, epsilon)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    def update(frame_idx: int) -> None:
        ax.clear()
        ax.set_xlim(0.0, xlim)
        ax.set_ylim(0.0, ylim)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Branch-and-Bound Traversal (step {frame_idx + 1}/{len(trace)})")
        ax.grid(alpha=0.3)

        _draw_original_constraints(ax, A_np, b_np, xlim, epsilon)

        active_node_ids = set(t["node_id"] for t in trace[: frame_idx + 1])
        visible_branch_lines = [bl for bl in branch_lines if bl.get("node_id") in active_node_ids]
        _draw_branch_lines(ax, visible_branch_lines, xlim, epsilon)

        visible_gomory_lines = [gl for gl in gomory_lines if gl.get("node_id") in active_node_ids]
        _draw_gomory_lines(ax, visible_gomory_lines, xlim, epsilon)

        _draw_trace_points(ax, trace[: frame_idx + 1])

        if frame_idx == len(trace) - 1:
            _draw_final_feasible_region(
                ax=ax,
                A=A_np,
                b=b_np,
                incumbent_constraints=incumbent_constraints,
                incumbent_gomory_constraints=incumbent_gomory_constraints,
                xlim=xlim,
                ylim=ylim,
                grid_size=grid_size,
                epsilon=epsilon,
            )
            if incumbent_x is not None:
                ax.scatter([incumbent_x[0]], [incumbent_x[1]], c="#16a34a", s=120, marker="*", label="incumbent")

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")

    frames = max(1, len(trace))
    animation = FuncAnimation(fig, update, frames=frames, interval=max(100, int(1000 / max(1, fps))))
    animation.save(str(output), writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    return str(output)


def _infer_plot_bounds(
    A: np.ndarray,
    b: np.ndarray,
    points: List[Sequence[float]],
    eps: float,
) -> Tuple[float, float]:
    x_max = 1.0
    y_max = 1.0

    for row, rhs in zip(A, b):
        if row[0] > eps:
            x_max = max(x_max, rhs / row[0])
        if row[1] > eps:
            y_max = max(y_max, rhs / row[1])

    for pt in points:
        x_max = max(x_max, float(pt[0]))
        y_max = max(y_max, float(pt[1]))

    return max(1.0, x_max * 1.25 + 1.0), max(1.0, y_max * 1.25 + 1.0)


def _draw_original_constraints(ax: Any, A: np.ndarray, b: np.ndarray, xlim: float, eps: float) -> None:
    for i, (row, rhs) in enumerate(zip(A, b)):
        _plot_line(ax, row, float(rhs), xlim, color="#64748b", linestyle="-", linewidth=1.2, alpha=0.7)
        if i == 0:
            ax.plot([], [], color="#64748b", linestyle="-", linewidth=1.2, label="original constraints")


def _draw_branch_lines(ax: Any, branch_lines: List[Dict[str, Any]], xlim: float, eps: float) -> None:
    has_label = False
    for bl in branch_lines:
        coeff = np.asarray(bl["coeff"], dtype=float)
        rhs = float(bl["rhs"])
        _plot_line(
            ax,
            coeff,
            rhs,
            xlim,
            color="#f97316",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        if not has_label:
            ax.plot([], [], color="#f97316", linestyle="--", linewidth=1.0, label="branch constraints")
            has_label = True


def _draw_trace_points(ax: Any, trace: List[Dict[str, Any]]) -> None:
    pts = [t.get("lp_x") for t in trace if t.get("lp_x") is not None]
    if not pts:
        return

    arr = np.asarray(pts, dtype=float)
    ax.scatter(arr[:, 0], arr[:, 1], c="#2563eb", s=26, alpha=0.7, label="visited LP points")
    current = arr[-1]
    ax.scatter([current[0]], [current[1]], c="#dc2626", s=50, marker="o", label="current node")


def _draw_gomory_lines(ax: Any, gomory_lines: List[Dict[str, Any]], xlim: float, eps: float) -> None:
    has_label = False
    for gl in gomory_lines:
        coeff = np.asarray(gl["coeff"], dtype=float)
        if coeff.shape[0] < 2:
            continue
        rhs = float(gl["rhs"])
        _plot_line(
            ax,
            coeff[:2],
            rhs,
            xlim,
            color="#0f766e",
            linestyle=":",
            linewidth=1.3,
            alpha=0.95,
        )
        if not has_label:
            ax.plot([], [], color="#0f766e", linestyle=":", linewidth=1.3, label="gomory cuts (x1-x2 projection)")
            has_label = True


def _draw_final_feasible_region(
    ax: Any,
    A: np.ndarray,
    b: np.ndarray,
    incumbent_constraints: List[Dict[str, Any]],
    incumbent_gomory_constraints: List[Dict[str, Any]],
    xlim: float,
    ylim: float,
    grid_size: int,
    epsilon: float,
) -> None:
    x_vals = np.linspace(0.0, xlim, max(40, grid_size))
    y_vals = np.linspace(0.0, ylim, max(40, grid_size))
    X, Y = np.meshgrid(x_vals, y_vals)

    mask = np.ones_like(X, dtype=bool)
    for row, rhs in zip(A, b):
        mask &= (row[0] * X + row[1] * Y) <= (rhs + epsilon)

    for item in incumbent_constraints:
        coeff = np.asarray(item["coeff"], dtype=float)
        rhs = float(item["rhs"])
        if coeff.shape[0] >= 2:
            mask &= (coeff[0] * X + coeff[1] * Y) <= (rhs + epsilon)

    for item in incumbent_gomory_constraints:
        coeff = np.asarray(item["coeff"], dtype=float)
        rhs = float(item["rhs"])
        if coeff.shape[0] >= 2:
            # 2D view uses x1/x2 projection of cuts; ignore projection with no x1/x2 signal.
            if abs(coeff[0]) <= epsilon and abs(coeff[1]) <= epsilon:
                continue
            mask &= (coeff[0] * X + coeff[1] * Y) <= (rhs + epsilon)

    mask &= X >= -epsilon
    mask &= Y >= -epsilon

    overlay = np.where(mask, 1.0, np.nan)
    ax.contourf(X, Y, overlay, levels=[0.5, 1.5], colors=["#86efac"], alpha=0.28)
    ax.plot([], [], color="#16a34a", linewidth=6, alpha=0.28, label="final feasible region")


def _plot_line(
    ax: Any,
    coeff: np.ndarray,
    rhs: float,
    xlim: float,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
) -> None:
    a = float(coeff[0])
    b = float(coeff[1])
    eps = 1e-12

    if abs(b) > eps:
        xs = np.linspace(0.0, xlim, 200)
        ys = (rhs - a * xs) / b
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        return

    if abs(a) > eps:
        x_v = rhs / a
        ax.axvline(x=x_v, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
