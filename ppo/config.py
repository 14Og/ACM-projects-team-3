from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RobotConfig:
    """Physical parameters of the planar 3DOF manipulator."""

    link_lengths: Tuple[float, float, float] = (90.0, 70.0, 40.0)
    masses: Tuple[float, float, float] = (1.0, 0.7, 0.6)
    damping: float = 1.0


@dataclass
class BacksteppingConfig:
    """Tunable parameters for trajectory tracking."""

    # Use "ellipse" for end-effector tracking in XY, or "joint_sine" for
    # q_d = [A1 sin(wt), A2 cos(wt), A3 sin(2wt)].
    trajectory_mode: str = "ellipse"
    amplitudes: Tuple[float, float, float] = (0.5, 0.4, 0.3)
    omega: float = 0.7

    ellipse_center_xy: Tuple[float, float] = (60.0, 0.0)
    ellipse_radii_xy: Tuple[float, float] = (165.0, 45.0)
    ellipse_orientation: float = 0.0
    ellipse_elbow: int = 1
    fit_ellipse_to_workspace: bool = True
    project_ellipse_to_workspace: bool = True
    workspace_margin: float = 2.0

    # Backstepping gains. Change them here to tune the closed-loop response.
    k1: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    k2: Tuple[float, float, float] = (10.0, 10.0, 10.0)

    duration: float = 12.0
    dt: float = 0.002
    derivative_dt: float = 1e-4
    initial_q: Optional[Tuple[float, float, float]] = None
    initial_q_dot: Optional[Tuple[float, float, float]] = None
    initial_q_error: Tuple[float, float, float] = (0.15, -0.10, 0.08)
    initial_q_dot_error: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    output_dir: str = "outputs/backstepping_tracking"
    compensate_damping: bool = True
    torque_clip: Optional[Tuple[float, float, float]] = None
