"""Reference generation and Lyapunov/adaptive controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    AdaptiveControllerConfig,
    FixedLyapunovControllerConfig,
    PDControllerConfig,
    PlannerConfig,
    TargetConfig,
)
from .system import MovingObstacles, PlanarArm, angle_error, wrap_angles


@dataclass(frozen=True)
class ReferenceState:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    target: np.ndarray
    target_velocity: np.ndarray
    obstacle_centers: np.ndarray
    reference_clearance: float


@dataclass(frozen=True)
class ControlInfo:
    tau_raw: np.ndarray
    tau: np.ndarray
    q_error: np.ndarray
    dq_error: np.ndarray
    sliding_error: np.ndarray
    qddot_r: np.ndarray
    inertia_hat: np.ndarray
    damping_hat: np.ndarray
    saturated: bool


class ObstacleAwareReferenceGenerator:
    """Kinematic APF reference generator for the manipulator.

    This is a planner, not the adaptive-control proof itself. It turns the
    target and obstacle geometry into a bounded joint reference. The adaptive
    controller then tracks that reference.
    """

    def __init__(
        self,
        *,
        arm: PlanarArm,
        obstacles: MovingObstacles,
        target_cfg: TargetConfig,
        planner_cfg: PlannerConfig,
        q0: np.ndarray,
        obstacle_radius: float,
    ) -> None:
        self.arm = arm
        self.obstacles = obstacles
        self.target_cfg = target_cfg
        self.cfg = planner_cfg
        self.obstacle_radius = float(obstacle_radius)
        self.q = wrap_angles(np.asarray(q0, dtype=float))
        self.dq = np.zeros_like(self.q)

    def reset(self) -> None:
        self.q = self.arm.initial_angles.copy()
        self.dq = np.zeros_like(self.q)

    def step(self, time: float, dt: float) -> ReferenceState:
        target = self.target_position(time)
        target_velocity = self.target_velocity(time)
        obstacle_centers = self.obstacles.centers(time)
        q_next, dq_next = self._next_reference(self.q, target, target_velocity, obstacle_centers, dt)
        ddq = (dq_next - self.dq) / dt
        self.q = q_next
        self.dq = dq_next
        clearance = self.arm.clearance(self.q, obstacle_centers, self.obstacle_radius).min_clearance
        return ReferenceState(
            q=self.q.copy(),
            dq=self.dq.copy(),
            ddq=ddq,
            target=target,
            target_velocity=target_velocity,
            obstacle_centers=obstacle_centers,
            reference_clearance=clearance,
        )

    def target_position(self, time: float) -> np.ndarray:
        phase = self.target_cfg.omega * float(time)
        return self.target_cfg.center_xy + np.array(
            [
                self.target_cfg.amplitude_xy[0] * np.cos(phase),
                self.target_cfg.amplitude_xy[1] * np.sin(phase),
            ],
            dtype=float,
        )

    def target_velocity(self, time: float) -> np.ndarray:
        phase = self.target_cfg.omega * float(time)
        omega = self.target_cfg.omega
        return np.array(
            [
                -self.target_cfg.amplitude_xy[0] * omega * np.sin(phase),
                self.target_cfg.amplitude_xy[1] * omega * np.cos(phase),
            ],
            dtype=float,
        )

    def _next_reference(
        self,
        q: np.ndarray,
        target: np.ndarray,
        target_velocity: np.ndarray,
        obstacle_centers: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = self.arm.forward_kinematics(q)
        end_effector = points[-1]
        desired_task_velocity = self.cfg.goal_gain * (target - end_effector) + target_velocity
        desired_task_velocity = _limit_norm(desired_task_velocity, self.cfg.max_task_speed)

        jacobian = self.arm.point_jacobian(q, self.arm.n_joints)
        damping_matrix = (self.cfg.damping**2) * np.eye(2)
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_matrix, desired_task_velocity)

        for center in obstacle_centers:
            for link_index in range(1, self.arm.n_joints + 1):
                point = points[link_index]
                delta = point - center
                distance = float(np.linalg.norm(delta))
                clearance = distance - self.obstacle_radius - self.cfg.safe_margin
                if clearance >= self.cfg.repulsion_influence:
                    continue

                direction = delta / (distance + 1e-9)
                effective_clearance = max(clearance, 4.0)
                magnitude = (
                    self.cfg.repulsion_gain
                    * (1.0 / effective_clearance - 1.0 / self.cfg.repulsion_influence)
                    / (effective_clearance**2)
                    * self.cfg.repulsion_scale
                )
                if clearance < 0.0:
                    magnitude += self.cfg.repulsion_gain * (-clearance + 1.0) * 0.5
                point_velocity = magnitude * direction
                point_jacobian = self.arm.point_jacobian(q, link_index)
                dq += (
                    self.cfg.repulsion_joint_gain
                    * (point_jacobian.T @ point_velocity)
                    / (np.linalg.norm(point_jacobian, ord="fro") + 1e-6)
                )

        dq = np.clip(dq, -self.cfg.max_joint_speed, self.cfg.max_joint_speed)
        return wrap_angles(q + dt * dq), dq


class AdaptiveLyapunovController:
    """Slotine-Li-style adaptive controller for diagonal joint dynamics."""

    name = "adaptive"

    def __init__(self, cfg: AdaptiveControllerConfig, torque_limits: np.ndarray) -> None:
        self.cfg = cfg
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        self.inertia_hat = cfg.initial_inertia_hat.copy()
        self.damping_hat = cfg.initial_damping_hat.copy()

    def reset(self) -> None:
        self.inertia_hat = self.cfg.initial_inertia_hat.copy()
        self.damping_hat = self.cfg.initial_damping_hat.copy()

    def compute(self, q: np.ndarray, dq: np.ndarray, ref: ReferenceState, dt: float) -> ControlInfo:
        q_error, dq_error, sliding, qddot_r = _filtered_errors(
            q,
            dq,
            ref,
            self.cfg.lambda_gain,
        )
        tau_raw = self.inertia_hat * qddot_r + self.damping_hat * dq - self.cfg.sliding_gain * sliding
        tau = np.clip(tau_raw, -self.torque_limits, self.torque_limits)

        self.inertia_hat = np.clip(
            self.inertia_hat + dt * (-self.cfg.gamma_inertia * sliding * qddot_r),
            self.cfg.inertia_bounds[0],
            self.cfg.inertia_bounds[1],
        )
        self.damping_hat = np.clip(
            self.damping_hat + dt * (-self.cfg.gamma_damping * sliding * dq),
            self.cfg.damping_bounds[0],
            self.cfg.damping_bounds[1],
        )

        return ControlInfo(
            tau_raw=tau_raw,
            tau=tau,
            q_error=q_error,
            dq_error=dq_error,
            sliding_error=sliding,
            qddot_r=qddot_r,
            inertia_hat=self.inertia_hat.copy(),
            damping_hat=self.damping_hat.copy(),
            saturated=bool(np.any(np.abs(tau_raw - tau) > 1e-9)),
        )


class FixedLyapunovController:
    """Same filtered-error law as adaptive control, but with fixed wrong parameters."""

    name = "fixed_lyapunov"

    def __init__(self, cfg: FixedLyapunovControllerConfig, torque_limits: np.ndarray) -> None:
        self.cfg = cfg
        self.torque_limits = np.asarray(torque_limits, dtype=float)

    def reset(self) -> None:
        return None

    def compute(self, q: np.ndarray, dq: np.ndarray, ref: ReferenceState, dt: float) -> ControlInfo:
        del dt
        q_error, dq_error, sliding, qddot_r = _filtered_errors(
            q,
            dq,
            ref,
            self.cfg.lambda_gain,
        )
        tau_raw = self.cfg.nominal_inertia * qddot_r + self.cfg.nominal_damping * dq
        tau_raw = tau_raw - self.cfg.sliding_gain * sliding
        tau = np.clip(tau_raw, -self.torque_limits, self.torque_limits)
        return ControlInfo(
            tau_raw=tau_raw,
            tau=tau,
            q_error=q_error,
            dq_error=dq_error,
            sliding_error=sliding,
            qddot_r=qddot_r,
            inertia_hat=self.cfg.nominal_inertia.copy(),
            damping_hat=self.cfg.nominal_damping.copy(),
            saturated=bool(np.any(np.abs(tau_raw - tau) > 1e-9)),
        )


class PlainPDController:
    """Classical joint-space Lyapunov PD baseline."""

    name = "plain_pd"

    def __init__(self, cfg: PDControllerConfig, torque_limits: np.ndarray) -> None:
        self.cfg = cfg
        self.torque_limits = np.asarray(torque_limits, dtype=float)

    def reset(self) -> None:
        return None

    def compute(self, q: np.ndarray, dq: np.ndarray, ref: ReferenceState, dt: float) -> ControlInfo:
        del dt
        q_error = angle_error(q, ref.q)
        dq_error = dq - ref.dq
        tau_raw = -self.cfg.kp * q_error - self.cfg.kd * dq_error
        tau = np.clip(tau_raw, -self.torque_limits, self.torque_limits)
        return ControlInfo(
            tau_raw=tau_raw,
            tau=tau,
            q_error=q_error,
            dq_error=dq_error,
            sliding_error=dq_error,
            qddot_r=np.zeros_like(q),
            inertia_hat=np.full_like(q, np.nan, dtype=float),
            damping_hat=np.full_like(q, np.nan, dtype=float),
            saturated=bool(np.any(np.abs(tau_raw - tau) > 1e-9)),
        )


def _filtered_errors(
    q: np.ndarray,
    dq: np.ndarray,
    ref: ReferenceState,
    lambda_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_error = angle_error(q, ref.q)
    dq_error = dq - ref.dq
    sliding = dq_error + lambda_gain * q_error
    qddot_r = ref.ddq - lambda_gain * dq_error
    return q_error, dq_error, sliding, qddot_r


def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)

