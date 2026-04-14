"""Configuration dataclasses for the Lyapunov APF project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class APFConfig:
    """APF controller parameters."""

    # Attractive / tracking gains
    k_att: float = 5
    k_v: float = 3.6

    # Obstacle repulsion
    k_rep: float = 1.05
    influence_radius: float = 0.5  # distance from safety boundary where repulsion starts
    safe_margin: float = 0.1
    eps_dist: float = 1e-3

    # Control constraint
    constrain_control: bool = True
    u_max: float = 8.0


@dataclass
class SimConfig:
    """Simulation timing and integration parameters."""
    steps_per_episode: int = 700
    num_cycles: float = 2


@dataclass
class EnvConfig:
    """Environment: reference ellipse, plant geometry, spawn rules, obstacles."""

    # Reference ellipse  (cx + a*cos(theta),  cy + b*sin(theta))
    cx: float = 22.0
    cy: float = 20.0
    a: float = 14.0
    b: float = 10.0
    omega_range: Tuple[float, float] = (0.16, 0.32)

    # Plant geometry
    plant_radius: float = 1.8   # plant radius
    target_radius: float = 1.4

    # Plant spawn constraints
    plant_spawn_scale_range: Tuple[float, float] = (1.20, 1.85)
    plant_spawn_clearance: float = 1.0
    target_obstacle_clearance: float = 0.1
    min_target_plant_start_distance: float = 7.0


    # Obstacles: shared radius + base (x, y) positions.
    # Each episode positions are perturbed by a uniform draw in
    # [-obstacle_drift, +obstacle_drift] per axis; radius stays fixed.
    obstacle_radius: float = 1.8
    obstacle_bases: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (22.0, 28.0),   # inner – top of ellipse
            (22.0, 36.0),   # outer – top of ellipse
            (30.0, 22.0),   # inner – right of ellipse
            (42.0, 20.0),   # outer – right of ellipse
        ]
    )
    obstacle_drift: float = 2.0 # max per-axis position drift each episode


@dataclass
class EpisodeState:
    """Fully randomised state for one simulation episode."""

    theta0: float                               # target start angle on ellipse
    omega: float                                # angular velocity of target
    t_final: float                              # simulation horizon
    p0: Tuple[float, float]                     # plant initial position
    v0: Tuple[float, float]                     # plant initial velocity
    obstacles: List[Tuple[float, float, float]] # (x, y, radius) after drift
