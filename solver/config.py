from dataclasses import dataclass


@dataclass
class SolverConfig:
    epsilon: float = 1e-9
    max_iterations: int = 10_000
    max_nodes: int = 50_000
    diving_max_depth: int = 20
    diving_max_tries: int = 2
    use_rounding_heuristic: bool = True
    rounding_max_repair_steps: int = 100
    visualize: bool = False
    visualization_output: str = "outputs/bnb_animation.gif"
    visualization_timeline_output: str = "outputs/bnb_timeline.png"
    visualization_generate_timeline: bool = True
    visualization_timeline_panels: int = 6
    visualization_fps: int = 2
    visualization_grid_size: int = 160
    max_trace_nodes: int = 8_000
    use_gomory_cuts: bool = True
    max_gomory_cuts_per_node: int = 1
