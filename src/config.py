"""Configuration loading for the robust manipulator project."""

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
    link_masses: np.ndarray
    joint_damping: np.ndarray
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
class PlannerConfig:
    goal_gain: float
    max_task_speed: float
    damping: float
    max_joint_speed: float
    startup_ramp_time: float


@dataclass(frozen=True)
class RobustControllerConfig:
    lambda_gain: float
    sliding_gain: np.ndarray
    reference_model_omega_n: np.ndarray
    reference_model_zeta: np.ndarray
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
    planner: PlannerConfig
    robust_controller: RobustControllerConfig
    simulation: SimulationConfig
    output: OutputConfig


def load_config(path: str | Path) -> ProjectConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    planner_raw = dict(raw["planner"])
    planner_raw.setdefault("startup_ramp_time", 0.0)
    return ProjectConfig(
        robot=RobotConfig(
            base_xy=_array(raw["robot"]["base_xy"]),
            link_lengths=_array(raw["robot"]["link_lengths"]),
            initial_angles=_array(raw["robot"]["initial_angles"]),
        ),
        dynamics=DynamicsConfig(
            link_masses=_array(raw["dynamics"]["link_masses"]),
            joint_damping=_array(raw["dynamics"]["joint_damping"]),
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
        planner=PlannerConfig(**{k: float(v) for k, v in planner_raw.items()}),
        robust_controller=RobustControllerConfig(
            lambda_gain=float(raw["robust_controller"]["lambda_gain"]),
            sliding_gain=_array(raw["robust_controller"]["sliding_gain"]),
            reference_model_omega_n=_array(
                raw["robust_controller"].get("reference_model_omega_n", [3.0, 3.0, 3.0])
            ),
            reference_model_zeta=_array(
                raw["robust_controller"].get("reference_model_zeta", [1.0, 1.0, 1.0])
            ),
            gamma_inertia=_array(raw["robust_controller"]["gamma_inertia"]),
            gamma_damping=_array(raw["robust_controller"]["gamma_damping"]),
            gamma_bias=_array(raw["robust_controller"].get("gamma_bias", [1.0, 1.0, 1.0])),
            initial_inertia_hat=_array(raw["robust_controller"]["initial_inertia_hat"]),
            initial_damping_hat=_array(raw["robust_controller"]["initial_damping_hat"]),
            initial_bias_hat=_array(raw["robust_controller"].get("initial_bias_hat", [0.0, 0.0, 0.0])),
            inertia_bounds=_bounds(raw["robust_controller"]["inertia_bounds"]),
            damping_bounds=_bounds(raw["robust_controller"]["damping_bounds"]),
            bias_bounds=_bounds(raw["robust_controller"].get("bias_bounds", [-100.0, 100.0])),
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
