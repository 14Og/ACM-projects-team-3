"""Configuration loading for the adaptive manipulator project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RobotConfig:
    base_xy: np.ndarray
    link_lengths: np.ndarray
    initial_angles: np.ndarray


@dataclass(frozen=True)
class DynamicsConfig:
    true_inertia: np.ndarray
    true_damping: np.ndarray
    link_masses: np.ndarray
    torque_limits: np.ndarray
    disturbance_constant: np.ndarray
    disturbance_amplitude: np.ndarray
    disturbance_frequency: np.ndarray


@dataclass(frozen=True)
class TargetConfig:
    center_xy: np.ndarray
    amplitude_xy: np.ndarray
    omega: float
    threshold: float


@dataclass(frozen=True)
class ObstacleConfig:
    radius: float
    base_centers: np.ndarray
    amplitudes: np.ndarray
    omegas: np.ndarray
    phases: np.ndarray


@dataclass(frozen=True)
class PlannerConfig:
    goal_gain: float
    max_task_speed: float
    damping: float
    max_joint_speed: float
    safe_margin: float
    repulsion_influence: float
    repulsion_gain: float
    repulsion_scale: float
    repulsion_joint_gain: float
    startup_ramp_time: float


@dataclass(frozen=True)
class AdaptiveControllerConfig:
    lambda_gain: float
    sliding_gain: np.ndarray
    gamma_inertia: np.ndarray
    gamma_damping: np.ndarray
    gamma_bias: np.ndarray
    initial_inertia_hat: np.ndarray
    initial_damping_hat: np.ndarray
    initial_bias_hat: np.ndarray
    inertia_bounds: tuple[float, float]
    damping_bounds: tuple[float, float]
    bias_bounds: tuple[float, float]


@dataclass(frozen=True)
class FixedLyapunovControllerConfig:
    lambda_gain: float
    sliding_gain: np.ndarray
    nominal_inertia: np.ndarray
    nominal_damping: np.ndarray


@dataclass(frozen=True)
class PDControllerConfig:
    kp: np.ndarray
    kd: np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    dt: float
    duration: float
    tail_window_seconds: float


@dataclass(frozen=True)
class OutputConfig:
    figures_dir: str
    animations_dir: str


@dataclass(frozen=True)
class ProjectConfig:
    robot: RobotConfig
    dynamics: DynamicsConfig
    target: TargetConfig
    obstacles: ObstacleConfig
    planner: PlannerConfig
    adaptive_controller: AdaptiveControllerConfig
    fixed_lyapunov_controller: FixedLyapunovControllerConfig
    pd_controller: PDControllerConfig
    simulation: SimulationConfig
    output: OutputConfig


def load_config(path: str | Path) -> ProjectConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    n_joints = len(raw["robot"]["link_lengths"])
    planner_raw = dict(raw["planner"])
    planner_raw.setdefault("startup_ramp_time", 0.0)
    return ProjectConfig(
        robot=RobotConfig(
            base_xy=_array(raw["robot"]["base_xy"]),
            link_lengths=_array(raw["robot"]["link_lengths"]),
            initial_angles=_array(raw["robot"]["initial_angles"]),
        ),
        dynamics=DynamicsConfig(
            true_inertia=_array(raw["dynamics"]["true_inertia"]),
            true_damping=_array(raw["dynamics"]["true_damping"]),
            link_masses=_array(raw["dynamics"].get("link_masses", [1.0] * n_joints)),
            torque_limits=_array(raw["dynamics"]["torque_limits"]),
            disturbance_constant=_array(
                raw["dynamics"].get("disturbance_constant", [0.0, 0.0, 0.0])
            ),
            disturbance_amplitude=_array(
                raw["dynamics"].get("disturbance_amplitude", [0.0, 0.0, 0.0])
            ),
            disturbance_frequency=_array(
                raw["dynamics"].get("disturbance_frequency", [0.0, 0.0, 0.0])
            ),
        ),
        target=TargetConfig(
            center_xy=_array(raw["target"]["center_xy"]),
            amplitude_xy=_array(raw["target"]["amplitude_xy"]),
            omega=float(raw["target"]["omega"]),
            threshold=float(raw["target"]["threshold"]),
        ),
        obstacles=ObstacleConfig(
            radius=float(raw["obstacles"]["radius"]),
            base_centers=_array(raw["obstacles"]["base_centers"]),
            amplitudes=_array(raw["obstacles"]["amplitudes"]),
            omegas=_array(raw["obstacles"]["omegas"]),
            phases=_array(raw["obstacles"]["phases"]),
        ),
        planner=PlannerConfig(**{k: float(v) for k, v in planner_raw.items()}),
        adaptive_controller=AdaptiveControllerConfig(
            lambda_gain=float(raw["adaptive_controller"]["lambda_gain"]),
            sliding_gain=_array(raw["adaptive_controller"]["sliding_gain"]),
            gamma_inertia=_array(raw["adaptive_controller"]["gamma_inertia"]),
            gamma_damping=_array(raw["adaptive_controller"]["gamma_damping"]),
            gamma_bias=_array(raw["adaptive_controller"].get("gamma_bias", [1.0, 1.0, 1.0])),
            initial_inertia_hat=_array(raw["adaptive_controller"]["initial_inertia_hat"]),
            initial_damping_hat=_array(raw["adaptive_controller"]["initial_damping_hat"]),
            initial_bias_hat=_array(raw["adaptive_controller"].get("initial_bias_hat", [0.0, 0.0, 0.0])),
            inertia_bounds=_bounds(raw["adaptive_controller"]["inertia_bounds"]),
            damping_bounds=_bounds(raw["adaptive_controller"]["damping_bounds"]),
            bias_bounds=_bounds(raw["adaptive_controller"].get("bias_bounds", [-100.0, 100.0])),
        ),
        fixed_lyapunov_controller=FixedLyapunovControllerConfig(
            lambda_gain=float(raw["fixed_lyapunov_controller"]["lambda_gain"]),
            sliding_gain=_array(raw["fixed_lyapunov_controller"]["sliding_gain"]),
            nominal_inertia=_array(raw["fixed_lyapunov_controller"]["nominal_inertia"]),
            nominal_damping=_array(raw["fixed_lyapunov_controller"]["nominal_damping"]),
        ),
        pd_controller=PDControllerConfig(
            kp=_array(raw["pd_controller"]["kp"]),
            kd=_array(raw["pd_controller"]["kd"]),
        ),
        simulation=SimulationConfig(
            dt=float(raw["simulation"]["dt"]),
            duration=float(raw["simulation"]["duration"]),
            tail_window_seconds=float(raw["simulation"]["tail_window_seconds"]),
        ),
        output=OutputConfig(**raw["output"]),
    )


def _array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _bounds(values: Any) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"Expected two bounds, got {values!r}")
    return float(values[0]), float(values[1])
