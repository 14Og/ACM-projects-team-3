"""Planar manipulator kinematics, obstacles, and joint-space dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynamicsConfig, ObstacleConfig, RobotConfig


@dataclass(frozen=True)
class ClearanceResult:
    min_clearance: float
    collision: bool


class PlanarArm:
    """Kinematic model for a fixed-base serial planar arm."""

    def __init__(self, cfg: RobotConfig) -> None:
        self.base = np.asarray(cfg.base_xy, dtype=float)
        self.links = np.asarray(cfg.link_lengths, dtype=float)
        self.initial_angles = np.asarray(cfg.initial_angles, dtype=float)
        self.n_joints = int(self.links.size)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        points = [self.base.copy()]
        angle = 0.0
        point = self.base.copy()
        for joint_angle, link_length in zip(q, self.links, strict=True):
            angle += float(joint_angle)
            point = point + link_length * np.array([np.cos(angle), np.sin(angle)])
            points.append(point.copy())
        return np.asarray(points, dtype=float)

    def end_effector(self, q: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(q)[-1]

    def point_jacobian(self, q: np.ndarray, link_index: int) -> np.ndarray:
        """Return the 2 x n Jacobian for the point at the end of link_index."""
        q = np.asarray(q, dtype=float)
        link_index = int(link_index)
        if link_index < 1 or link_index > self.n_joints:
            raise ValueError("link_index must be in [1, n_joints]")

        cumulative = np.cumsum(q)
        jacobian = np.zeros((2, self.n_joints), dtype=float)
        for joint in range(link_index):
            contribution = np.zeros(2, dtype=float)
            for link in range(joint, link_index):
                contribution += self.links[link] * np.array(
                    [-np.sin(cumulative[link]), np.cos(cumulative[link])],
                    dtype=float,
                )
            jacobian[:, joint] = contribution
        return jacobian

    def clearance(self, q: np.ndarray, obstacle_centers: np.ndarray, radius: float) -> ClearanceResult:
        points = self.forward_kinematics(q)
        min_clearance = float("inf")
        for center in obstacle_centers:
            center = np.asarray(center, dtype=float)
            for start, end in zip(points[:-1], points[1:], strict=True):
                closest = closest_point_on_segment(center, start, end)
                clearance = float(np.linalg.norm(center - closest) - radius)
                min_clearance = min(min_clearance, clearance)
        return ClearanceResult(min_clearance=min_clearance, collision=min_clearance < 0.0)


class MovingObstacles:
    """Circular obstacles moving on small ellipses around nominal centers."""

    def __init__(self, cfg: ObstacleConfig) -> None:
        self.cfg = cfg

    def centers(self, time: float) -> np.ndarray:
        phase = self.cfg.omegas * float(time) + self.cfg.phases
        offsets = np.column_stack(
            [
                self.cfg.amplitudes[:, 0] * np.cos(phase),
                self.cfg.amplitudes[:, 1] * np.sin(phase),
            ]
        )
        return self.cfg.base_centers + offsets


class JointSpacePlant:
    """Simplified torque-level dynamics with unknown diagonal inertia and damping."""

    def __init__(self, q0: np.ndarray, cfg: DynamicsConfig) -> None:
        self.q = wrap_angles(np.asarray(q0, dtype=float))
        self.dq = np.zeros_like(self.q)
        self.inertia = np.asarray(cfg.true_inertia, dtype=float)
        self.damping = np.asarray(cfg.true_damping, dtype=float)
        self.torque_limits = np.asarray(cfg.torque_limits, dtype=float)

    def acceleration(self, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return (tau - self.damping * dq) / self.inertia

    def step(self, tau: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate with constant torque using RK4."""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limits, self.torque_limits)
        q0 = self.q.copy()
        dq0 = self.dq.copy()

        def rhs(q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            del q
            return dq, self.acceleration(dq, tau)

        k1_q, k1_dq = rhs(q0, dq0)
        k2_q, k2_dq = rhs(q0 + 0.5 * dt * k1_q, dq0 + 0.5 * dt * k1_dq)
        k3_q, k3_dq = rhs(q0 + 0.5 * dt * k2_q, dq0 + 0.5 * dt * k2_dq)
        k4_q, k4_dq = rhs(q0 + dt * k3_q, dq0 + dt * k3_dq)

        self.q = wrap_angles(q0 + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q))
        self.dq = dq0 + (dt / 6.0) * (k1_dq + 2.0 * k2_dq + 2.0 * k3_dq + k4_dq)
        return self.q.copy(), self.dq.copy(), tau


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return start.copy()
    fraction = float(np.dot(point - start, segment) / denom)
    fraction = min(max(fraction, 0.0), 1.0)
    return start + fraction * segment


def wrap_angles(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def angle_error(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    return wrap_angles(np.asarray(q, dtype=float) - np.asarray(q_ref, dtype=float))

