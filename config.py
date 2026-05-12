"""Configuration loading for the adaptive manipulator project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CONTROLLER_TYPE = "all"  # "adaptive", "adaptive_simp", "backstepping_full", "backstepping_simp", or "all"
SIMULATE_PAYLOAD_ERROR = False
PAYLOAD_MULTIPLIER = 3.0


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
    gamma_mass: np.ndarray
    initial_inertia_hat: np.ndarray
    initial_damping_hat: np.ndarray
    initial_bias_hat: np.ndarray
    initial_mass_hat: np.ndarray
    inertia_bounds: tuple[float, float]
    damping_bounds: tuple[float, float]
    bias_bounds: tuple[float, float]
    mass_bounds: tuple[float, float]


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
class BacksteppingControllerConfig:
    k1: np.ndarray
    k2: np.ndarray
    assumed_link_masses: np.ndarray


@dataclass(frozen=True)
class ControllerSelectionConfig:
    controller_type: str
    simulate_payload_error: bool
    payload_multiplier: float


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
    backstepping_controller: BacksteppingControllerConfig
    controller_selection: ControllerSelectionConfig
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
            gamma_mass=_array(raw["adaptive_controller"].get("gamma_mass", [0.1, 0.1, 0.1])),
            initial_inertia_hat=_array(raw["adaptive_controller"]["initial_inertia_hat"]),
            initial_damping_hat=_array(raw["adaptive_controller"]["initial_damping_hat"]),
            initial_bias_hat=_array(raw["adaptive_controller"].get("initial_bias_hat", [0.0, 0.0, 0.0])),
            initial_mass_hat=_array(
                raw["adaptive_controller"].get(
                    "initial_mass_hat",
                    raw["dynamics"].get("link_masses", [1.0] * n_joints),
                )
            ),
            inertia_bounds=_bounds(raw["adaptive_controller"]["inertia_bounds"]),
            damping_bounds=_bounds(raw["adaptive_controller"]["damping_bounds"]),
            bias_bounds=_bounds(raw["adaptive_controller"].get("bias_bounds", [-100.0, 100.0])),
            mass_bounds=_bounds(raw["adaptive_controller"].get("mass_bounds", [0.05, 10.0])),
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
        backstepping_controller=BacksteppingControllerConfig(
            k1=_array(raw.get("backstepping_controller", {}).get("k1", [10.0] * n_joints)),
            k2=_array(raw.get("backstepping_controller", {}).get("k2", [10.0] * n_joints)),
            assumed_link_masses=_array(
                raw.get("backstepping_controller", {}).get(
                    "assumed_link_masses",
                    raw["dynamics"].get("link_masses", [1.0] * n_joints),
                )
            ),
        ),
        controller_selection=ControllerSelectionConfig(
            controller_type=str(raw.get("controller_selection", {}).get("controller_type", "all")),
            simulate_payload_error=bool(
                raw.get("controller_selection", {}).get("simulate_payload_error", False)
            ),
            payload_multiplier=float(
                raw.get("controller_selection", {}).get("payload_multiplier", 3.0)
            ),
        ),
        simulation=SimulationConfig(
            dt=float(raw["simulation"]["dt"]),
            duration=float(raw["simulation"]["duration"]),
            tail_window_seconds=float(raw["simulation"]["tail_window_seconds"]),
        ),
        output=OutputConfig(**raw["output"]),
    )


def default_config() -> ProjectConfig:
    """Default configuration for comparing adaptive and backstepping controllers."""
    n = 3
    link_lengths = np.array([90.0, 70.0, 40.0], dtype=float)
    link_masses = np.array([1.0, 0.7, 0.6], dtype=float)
    inertia = np.array([1.2, 0.8, 0.45], dtype=float)
    damping = np.array([0.08, 0.06, 0.05], dtype=float)
    torque_limits = np.array([80.0, 60.0, 40.0], dtype=float)

    return ProjectConfig(
        robot=RobotConfig(
            base_xy=np.array([0.0, 0.0], dtype=float),
            link_lengths=link_lengths,
            initial_angles=np.array([0.2, 0.1, -0.2], dtype=float),
        ),
        dynamics=DynamicsConfig(
            true_inertia=inertia,
            true_damping=damping,
            link_masses=link_masses,
            torque_limits=torque_limits,
            disturbance_constant=np.zeros(n, dtype=float),
            disturbance_amplitude=np.zeros(n, dtype=float),
            disturbance_frequency=np.zeros(n, dtype=float),
        ),
        target=TargetConfig(
            center_xy=np.array([60.0, 0.0], dtype=float),
            amplitude_xy=np.array([120.0, 35.0], dtype=float),
            omega=0.7,
            threshold=10.0,
        ),
        obstacles=ObstacleConfig(
            radius=0.0,
            base_centers=np.zeros((0, 2), dtype=float),
            amplitudes=np.zeros((0, 2), dtype=float),
            omegas=np.zeros(0, dtype=float),
            phases=np.zeros(0, dtype=float),
        ),
        planner=PlannerConfig(
            goal_gain=2.0,
            max_task_speed=120.0,
            damping=20.0,
            max_joint_speed=2.5,
            safe_margin=0.0,
            repulsion_influence=1.0,
            repulsion_gain=0.0,
            repulsion_scale=0.0,
            repulsion_joint_gain=0.0,
            startup_ramp_time=0.5,
        ),
        adaptive_controller=AdaptiveControllerConfig(
            lambda_gain=6.0,
            sliding_gain=np.array([18.0, 16.0, 12.0], dtype=float),
            gamma_inertia=np.array([0.8, 0.8, 0.8], dtype=float),
            gamma_damping=np.array([0.5, 0.5, 0.5], dtype=float),
            gamma_bias=np.array([0.4, 0.4, 0.4], dtype=float),
            gamma_mass=np.array([0.08, 0.08, 0.08], dtype=float),
            initial_inertia_hat=inertia.copy(),
            initial_damping_hat=damping.copy(),
            initial_bias_hat=np.zeros(n, dtype=float),
            initial_mass_hat=link_masses.copy(),
            inertia_bounds=(0.05, 8.0),
            damping_bounds=(0.0, 2.0),
            bias_bounds=(-100.0, 100.0),
            mass_bounds=(0.05, 3.0),
        ),
        fixed_lyapunov_controller=FixedLyapunovControllerConfig(
            lambda_gain=6.0,
            sliding_gain=np.array([18.0, 16.0, 12.0], dtype=float),
            nominal_inertia=inertia.copy(),
            nominal_damping=damping.copy(),
        ),
        pd_controller=PDControllerConfig(
            kp=np.array([20.0, 18.0, 14.0], dtype=float),
            kd=np.array([6.0, 5.0, 4.0], dtype=float),
        ),
        backstepping_controller=BacksteppingControllerConfig(
            k1=np.array([10.0, 10.0, 10.0], dtype=float),
            k2=np.array([10.0, 10.0, 10.0], dtype=float),
            assumed_link_masses=link_masses.copy(),
        ),
        controller_selection=ControllerSelectionConfig(
            controller_type=CONTROLLER_TYPE,
            simulate_payload_error=SIMULATE_PAYLOAD_ERROR,
            payload_multiplier=PAYLOAD_MULTIPLIER,
        ),
        simulation=SimulationConfig(
            dt=0.01,
            duration=12.0,
            tail_window_seconds=2.0,
        ),
        output=OutputConfig(
            figures_dir="outputs/comparison/figures",
            animations_dir="outputs/comparison/animations",
        ),
    )


def _array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _bounds(values: Any) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"Expected two bounds, got {values!r}")
    return float(values[0]), float(values[1])
