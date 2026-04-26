"""Closed-loop simulation for adaptive manipulator control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import ProjectConfig
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
    q_ref: np.ndarray
    dq_ref: np.ndarray
    tau: np.ndarray
    tau_raw: np.ndarray
    inertia_hat: np.ndarray
    damping_hat: np.ndarray
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


def run_rollout(config: ProjectConfig, controller: Controller) -> Rollout:
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
    plant = JointSpacePlant(config.robot.initial_angles, config.dynamics)
    controller.reset()

    dt = config.simulation.dt
    n_steps = int(round(config.simulation.duration / dt)) + 1
    time = np.linspace(0.0, config.simulation.duration, n_steps)
    n_joints = arm.n_joints
    n_obstacles = int(config.obstacles.base_centers.shape[0])

    q = np.zeros((n_steps, n_joints))
    dq = np.zeros((n_steps, n_joints))
    q_ref = np.zeros((n_steps, n_joints))
    dq_ref = np.zeros((n_steps, n_joints))
    tau = np.zeros((n_steps, n_joints))
    tau_raw = np.zeros((n_steps, n_joints))
    inertia_hat = np.zeros((n_steps, n_joints))
    damping_hat = np.zeros((n_steps, n_joints))
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
        info = controller.compute(plant.q, plant.dq, ref, dt)
        q_now, dq_now, tau_now = plant.step(info.tau, dt)

        centers = ref.obstacle_centers
        clearance_result = arm.clearance(q_now, centers, config.obstacles.radius)
        ee = arm.end_effector(q_now)

        q[index] = q_now
        dq[index] = dq_now
        q_ref[index] = ref.q
        dq_ref[index] = ref.dq
        tau[index] = tau_now
        tau_raw[index] = info.tau_raw
        inertia_hat[index] = info.inertia_hat
        damping_hat[index] = info.damping_hat
        end_effector[index] = ee
        target[index] = ref.target
        obstacle_centers[index] = centers
        target_error[index] = float(np.linalg.norm(ee - ref.target))
        q_error_norm[index] = float(np.linalg.norm(info.q_error))
        sliding_norm[index] = float(np.linalg.norm(info.sliding_error))
        tracking_lyapunov[index] = 0.5 * float(
            np.dot(info.q_error, info.q_error) + np.dot(info.dq_error, info.dq_error)
        )
        augmented_lyapunov[index] = _augmented_lyapunov(config, controller.name, info)
        clearance[index] = clearance_result.min_clearance
        reference_clearance[index] = ref.reference_clearance
        collision[index] = clearance_result.collision
        saturated[index] = info.saturated

    return Rollout(
        label=controller.name,
        time=time,
        q=q,
        dq=dq,
        q_ref=q_ref,
        dq_ref=dq_ref,
        tau=tau,
        tau_raw=tau_raw,
        inertia_hat=inertia_hat,
        damping_hat=damping_hat,
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


def summarize_rollout(config: ProjectConfig, rollout: Rollout) -> dict[str, float]:
    tail = max(1, int(round(config.simulation.tail_window_seconds / config.simulation.dt)))
    tail_error = rollout.target_error[-tail:]
    return {
        "final_target_error_px": float(rollout.target_error[-1]),
        "mean_target_error_px": float(np.mean(rollout.target_error)),
        "tail_mean_target_error_px": float(np.mean(tail_error)),
        "tail_max_target_error_px": float(np.max(tail_error)),
        "max_target_error_px": float(np.max(rollout.target_error)),
        "tail_success_fraction": float(np.mean(tail_error <= config.target.threshold)),
        "min_clearance_px": float(np.min(rollout.clearance)),
        "collision_count": float(np.sum(rollout.collision)),
        "saturation_fraction": float(np.mean(rollout.saturated)),
        "final_q_error_norm_rad": float(rollout.q_error_norm[-1]),
        "tail_mean_q_error_norm_rad": float(np.mean(rollout.q_error_norm[-tail:])),
        "rms_torque": float(np.sqrt(np.mean(np.sum(rollout.tau * rollout.tau, axis=1)))),
    }


def _augmented_lyapunov(config: ProjectConfig, name: str, info) -> float:
    if name != "adaptive":
        return float("nan")
    inertia_error = info.inertia_hat - config.dynamics.true_inertia
    damping_error = info.damping_hat - config.dynamics.true_damping
    return (
        0.5 * float(np.sum(config.dynamics.true_inertia * info.sliding_error * info.sliding_error))
        + 0.5 * float(np.sum(inertia_error * inertia_error / config.adaptive_controller.gamma_inertia))
        + 0.5 * float(np.sum(damping_error * damping_error / config.adaptive_controller.gamma_damping))
    )

