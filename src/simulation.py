"""Closed-loop simulation for adaptive manipulator control."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DynamicsConfig, ProjectConfig
from .controller import ObstacleAwareReferenceGenerator
from .system import JointSpacePlant, MovingObstacles, PlanarArm


class Controller(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def compute(self, q: np.ndarray, dq: np.ndarray, ref, dt: float):
        ...


@dataclass
class Rollout:
    label: str
    time: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    q_des: np.ndarray
    dq_des: np.ndarray
    dq_r: np.ndarray
    ddq_r: np.ndarray
    tau: np.ndarray
    tau_raw: np.ndarray
    disturbance_torque: np.ndarray
    inertia_hat: np.ndarray
    damping_hat: np.ndarray
    bias_hat: np.ndarray
    mass_hat: np.ndarray
    end_effector: np.ndarray
    target: np.ndarray
    obstacle_centers: np.ndarray
    target_error: np.ndarray
    q_error_norm: np.ndarray
    sliding_norm: np.ndarray
    tracking_lyapunov: np.ndarray
    augmented_lyapunov: np.ndarray
    clearance: np.ndarray
    reference_clearance: np.ndarray
    collision: np.ndarray
    saturated: np.ndarray


def run_rollout(
    config: ProjectConfig,
    controller: Controller,
    real_dynamics: DynamicsConfig | None = None,
) -> Rollout:
    real_dynamics = real_dynamics or config.dynamics
    arm = PlanarArm(config.robot)
    obstacles = MovingObstacles(config.obstacles)
    planner = ObstacleAwareReferenceGenerator(
        arm=arm,
        obstacles=obstacles,
        target_cfg=config.target,
        planner_cfg=config.planner,
        q0=config.robot.initial_angles,
        obstacle_radius=config.obstacles.radius,
    )
    plant = JointSpacePlant(config.robot.initial_angles, real_dynamics, config.robot)
    controller.reset()

    dt = config.simulation.dt
    n_steps = int(round(config.simulation.duration / dt)) + 1
    time = np.linspace(0.0, config.simulation.duration, n_steps)
    n_joints = arm.n_joints
    n_obstacles = int(config.obstacles.base_centers.shape[0])

    q = np.zeros((n_steps, n_joints))
    dq = np.zeros((n_steps, n_joints))
    q_des = np.zeros((n_steps, n_joints))
    dq_des = np.zeros((n_steps, n_joints))
    dq_r = np.zeros((n_steps, n_joints))
    ddq_r = np.zeros((n_steps, n_joints))
    tau = np.zeros((n_steps, n_joints))
    tau_raw = np.zeros((n_steps, n_joints))
    disturbance_torque = np.zeros((n_steps, n_joints))
    inertia_hat = np.zeros((n_steps, n_joints))
    damping_hat = np.zeros((n_steps, n_joints))
    bias_hat = np.zeros((n_steps, n_joints))
    mass_hat = np.zeros((n_steps, n_joints))
    end_effector = np.zeros((n_steps, 2))
    target = np.zeros((n_steps, 2))
    obstacle_centers = np.zeros((n_steps, n_obstacles, 2))
    target_error = np.zeros(n_steps)
    q_error_norm = np.zeros(n_steps)
    sliding_norm = np.zeros(n_steps)
    tracking_lyapunov = np.zeros(n_steps)
    augmented_lyapunov = np.zeros(n_steps)
    clearance = np.zeros(n_steps)
    reference_clearance = np.zeros(n_steps)
    collision = np.zeros(n_steps, dtype=bool)
    saturated = np.zeros(n_steps, dtype=bool)

    for index, t in enumerate(time):
        ref = planner.step(float(t), dt)
        q_now = plant.q.copy()
        dq_now = plant.dq.copy()
        info = controller.compute(q_now, dq_now, ref, dt)
        tau_now = info.tau.copy()
        disturbance_now = plant.disturbance(float(t))

        centers = ref.obstacle_centers
        clearance_result = arm.clearance(q_now, centers, config.obstacles.radius)
        ee = arm.end_effector(q_now)

        q[index] = q_now
        dq[index] = dq_now
        q_des[index] = ref.q
        dq_des[index] = ref.dq
        dq_r[index] = info.dq_r
        ddq_r[index] = info.ddq_r
        tau[index] = tau_now
        tau_raw[index] = info.tau_raw
        disturbance_torque[index] = disturbance_now
        inertia_hat[index] = info.inertia_hat
        damping_hat[index] = info.damping_hat
        bias_hat[index] = info.bias_hat
        mass_hat[index] = info.mass_hat
        end_effector[index] = ee
        target[index] = ref.target
        obstacle_centers[index] = centers
        target_error[index] = float(np.linalg.norm(ee - ref.target))
        q_error_norm[index] = float(np.linalg.norm(info.q_error))
        sliding_norm[index] = float(np.linalg.norm(info.sliding_error))
        tracking_lyapunov[index] = 0.5 * float(
            np.dot(info.q_error, info.q_error) + np.dot(info.sliding_error, info.sliding_error)
        )
        augmented_lyapunov[index] = _augmented_lyapunov(config, real_dynamics, controller.name, info)
        clearance[index] = clearance_result.min_clearance
        reference_clearance[index] = ref.reference_clearance
        collision[index] = clearance_result.collision
        saturated[index] = info.saturated

        if index < n_steps - 1:
            plant.step(info.tau, dt, float(t))

    return Rollout(
        label=controller.name,
        time=time,
        q=q,
        dq=dq,
        q_des=q_des,
        dq_des=dq_des,
        dq_r=dq_r,
        ddq_r=ddq_r,
        tau=tau,
        tau_raw=tau_raw,
        disturbance_torque=disturbance_torque,
        inertia_hat=inertia_hat,
        damping_hat=damping_hat,
        bias_hat=bias_hat,
        mass_hat=mass_hat,
        end_effector=end_effector,
        target=target,
        obstacle_centers=obstacle_centers,
        target_error=target_error,
        q_error_norm=q_error_norm,
        sliding_norm=sliding_norm,
        tracking_lyapunov=tracking_lyapunov,
        augmented_lyapunov=augmented_lyapunov,
        clearance=clearance,
        reference_clearance=reference_clearance,
        collision=collision,
        saturated=saturated,
    )


def summarize_rollout(
    config: ProjectConfig,
    rollout: Rollout,
    real_dynamics: DynamicsConfig | None = None,
) -> dict[str, object]:
    real_dynamics = real_dynamics or config.dynamics
    tail = max(1, int(round(config.simulation.tail_window_seconds / config.simulation.dt)))
    tail_error = rollout.target_error[-tail:]
    tail_sliding = rollout.sliding_norm[-tail:]
    finite_augmented = rollout.augmented_lyapunov[np.isfinite(rollout.augmented_lyapunov)]
    augmented_initial = float(finite_augmented[0]) if finite_augmented.size else None
    augmented_final = float(finite_augmented[-1]) if finite_augmented.size else None
    augmented_drop_fraction = (
        float((augmented_initial - augmented_final) / augmented_initial)
        if augmented_initial is not None
        and augmented_final is not None
        and abs(augmented_initial) > 1e-12
        else None
    )
    augmented_increments = np.diff(finite_augmented) if finite_augmented.size > 1 else np.array([])
    positive_augmented_increments = augmented_increments[augmented_increments > 1e-12]
    return {
        "final_target_error_px": float(rollout.target_error[-1]),
        "mean_target_error_px": float(np.mean(rollout.target_error)),
        "tail_mean_target_error_px": float(np.mean(tail_error)),
        "tail_max_target_error_px": float(np.max(tail_error)),
        "max_target_error_px": float(np.max(rollout.target_error)),
        "tail_success_fraction": float(np.mean(tail_error <= config.target.threshold)),
        "min_clearance_px": float(np.min(rollout.clearance)) if np.isfinite(np.min(rollout.clearance)) else 1e6,
        "collision_count": float(np.sum(rollout.collision)),
        "saturation_fraction": float(np.mean(rollout.saturated)),
        "final_q_error_norm_rad": float(rollout.q_error_norm[-1]),
        "tail_mean_q_error_norm_rad": float(np.mean(rollout.q_error_norm[-tail:])),
        "final_sliding_norm": float(rollout.sliding_norm[-1]),
        "tail_mean_sliding_norm": float(np.mean(tail_sliding)),
        "max_sliding_norm": float(np.max(rollout.sliding_norm)),
        "rms_torque": float(np.sqrt(np.mean(np.sum(rollout.tau * rollout.tau, axis=1)))),
        "rms_external_torque": float(
            np.sqrt(np.mean(np.sum(rollout.disturbance_torque * rollout.disturbance_torque, axis=1)))
        ),
        "max_external_torque": float(np.max(np.linalg.norm(rollout.disturbance_torque, axis=1))),
        "tracking_lyapunov_initial": float(rollout.tracking_lyapunov[0]),
        "tracking_lyapunov_final": float(rollout.tracking_lyapunov[-1]),
        "tracking_lyapunov_tail_mean": float(np.mean(rollout.tracking_lyapunov[-tail:])),
        "augmented_lyapunov_initial": augmented_initial,
        "augmented_lyapunov_final": augmented_final,
        "augmented_lyapunov_drop_fraction": augmented_drop_fraction,
        "augmented_lyapunov_positive_step_count": float(positive_augmented_increments.size),
        "augmented_lyapunov_max_positive_step": (
            float(np.max(positive_augmented_increments)) if positive_augmented_increments.size else 0.0
        ),
        "final_inertia_hat": _finite_list_or_none(rollout.inertia_hat[-1]),
        "final_damping_hat": _finite_list_or_none(rollout.damping_hat[-1]),
        "final_bias_hat": _finite_list_or_none(rollout.bias_hat[-1]),
        "final_mass_hat": _finite_list_or_none(rollout.mass_hat[-1]),
        "final_inertia_error_norm": _parameter_error_norm(rollout.inertia_hat[-1], real_dynamics.true_inertia),
        "final_damping_error_norm": _parameter_error_norm(rollout.damping_hat[-1], real_dynamics.true_damping),
        "final_bias_error_norm": _parameter_error_norm(
            rollout.bias_hat[-1],
            real_dynamics.disturbance_constant,
        ),
        "final_mass_error_norm": _parameter_error_norm(rollout.mass_hat[-1], real_dynamics.link_masses),
    }


def save_rollout_data_csv(rollouts: dict[str, Rollout], path: str | Path) -> Path:
    """Save synchronized rollout samples for report tables or external plotting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_rollout = next(iter(rollouts.values()))
    n_joints = int(first_rollout.q.shape[1])
    joint_fields = [
        *(f"q{j + 1}" for j in range(n_joints)),
        *(f"dq{j + 1}" for j in range(n_joints)),
        *(f"q_d{j + 1}" for j in range(n_joints)),
        *(f"dq_d{j + 1}" for j in range(n_joints)),
        *(f"dq_r{j + 1}" for j in range(n_joints)),
        *(f"ddq_r{j + 1}" for j in range(n_joints)),
        *(f"tau{j + 1}" for j in range(n_joints)),
        *(f"tau_raw{j + 1}" for j in range(n_joints)),
        *(f"inertia_hat{j + 1}" for j in range(n_joints)),
        *(f"damping_hat{j + 1}" for j in range(n_joints)),
        *(f"bias_hat{j + 1}" for j in range(n_joints)),
        *(f"mass_hat{j + 1}" for j in range(n_joints)),
        *(f"disturbance{j + 1}" for j in range(n_joints)),
    ]
    fieldnames = [
        "controller",
        "time_s",
        "target_x_px",
        "target_y_px",
        "end_effector_x_px",
        "end_effector_y_px",
        "target_error_px",
        "q_error_norm_rad",
        "sliding_norm",
        "tracking_lyapunov",
        "augmented_lyapunov",
        "clearance_px",
        "reference_clearance_px",
        "collision",
        "saturated",
        *joint_fields,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label, rollout in rollouts.items():
            for index, time in enumerate(rollout.time):
                row = {
                    "controller": label,
                    "time_s": float(time),
                    "target_x_px": float(rollout.target[index, 0]),
                    "target_y_px": float(rollout.target[index, 1]),
                    "end_effector_x_px": float(rollout.end_effector[index, 0]),
                    "end_effector_y_px": float(rollout.end_effector[index, 1]),
                    "target_error_px": float(rollout.target_error[index]),
                    "q_error_norm_rad": float(rollout.q_error_norm[index]),
                    "sliding_norm": float(rollout.sliding_norm[index]),
                    "tracking_lyapunov": float(rollout.tracking_lyapunov[index]),
                    "augmented_lyapunov": _csv_number(rollout.augmented_lyapunov[index]),
                    "clearance_px": float(rollout.clearance[index]),
                    "reference_clearance_px": float(rollout.reference_clearance[index]),
                    "collision": int(rollout.collision[index]),
                    "saturated": int(rollout.saturated[index]),
                }
                row.update(_joint_row("q", rollout.q[index]))
                row.update(_joint_row("dq", rollout.dq[index]))
                row.update(_joint_row("q_d", rollout.q_des[index]))
                row.update(_joint_row("dq_d", rollout.dq_des[index]))
                row.update(_joint_row("dq_r", rollout.dq_r[index]))
                row.update(_joint_row("ddq_r", rollout.ddq_r[index]))
                row.update(_joint_row("tau", rollout.tau[index]))
                row.update(_joint_row("tau_raw", rollout.tau_raw[index]))
                row.update(_joint_row("inertia_hat", rollout.inertia_hat[index]))
                row.update(_joint_row("damping_hat", rollout.damping_hat[index]))
                row.update(_joint_row("bias_hat", rollout.bias_hat[index]))
                row.update(_joint_row("mass_hat", rollout.mass_hat[index]))
                row.update(_joint_row("disturbance", rollout.disturbance_torque[index]))
                writer.writerow(row)
    return output_path


def _augmented_lyapunov(config: ProjectConfig, real_dynamics: DynamicsConfig, name: str, info) -> float:
    if name not in {"adaptive", "adaptive_simp"}:
        return float("nan")
    inertia_error = info.inertia_hat - real_dynamics.true_inertia
    damping_error = info.damping_hat - real_dynamics.true_damping
    bias_error = info.bias_hat - real_dynamics.disturbance_constant
    mass_error = info.mass_hat - real_dynamics.link_masses
    mass_term = 0.0
    if np.all(np.isfinite(mass_error)):
        mass_term = 0.5 * float(np.sum(mass_error * mass_error / config.adaptive_controller.gamma_mass))
    return (
        0.5 * float(np.sum(real_dynamics.true_inertia * info.sliding_error * info.sliding_error))
        + 0.5 * float(np.sum(inertia_error * inertia_error / config.adaptive_controller.gamma_inertia))
        + 0.5 * float(np.sum(damping_error * damping_error / config.adaptive_controller.gamma_damping))
        + 0.5 * float(np.sum(bias_error * bias_error / config.adaptive_controller.gamma_bias))
        + mass_term
    )


def _finite_list_or_none(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return [float(value) for value in values]


def _parameter_error_norm(values: np.ndarray, true_values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return float(np.linalg.norm(values - true_values))


def _csv_number(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def _joint_row(prefix: str, values: np.ndarray) -> dict[str, float | str]:
    return {f"{prefix}{index + 1}": _csv_number(value) for index, value in enumerate(values)}
