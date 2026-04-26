"""Planar manipulator kinematics and joint-space dynamics."""

from __future__ import annotations

import numpy as np

from .config import DynamicsConfig, RobotConfig
from .robot_dynamics import RobotDynamics3DOF


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

class JointSpacePlant:
    """Torque-level plant backed by the full 3DOF rigid-body manipulator model."""

    def __init__(self, q0: np.ndarray, robot_cfg: RobotConfig, cfg: DynamicsConfig) -> None:
        self.q = wrap_angles(np.asarray(q0, dtype=float))
        self.dq = np.zeros_like(self.q)
        self.torque_limits = np.asarray(cfg.torque_limits, dtype=float)
        self.disturbance_constant = np.asarray(cfg.disturbance_constant, dtype=float)
        self.disturbance_amplitude = np.asarray(cfg.disturbance_amplitude, dtype=float)
        self.disturbance_frequency = np.asarray(cfg.disturbance_frequency, dtype=float)
        self.robot = RobotDynamics3DOF(
            masses=np.asarray(cfg.link_masses, dtype=float),
            lengths=np.asarray(robot_cfg.link_lengths, dtype=float) / 20000.0,
            damping=np.asarray(cfg.joint_damping, dtype=float),
        )

    def disturbance(self, time: float) -> np.ndarray:
        return self.disturbance_constant + self.disturbance_amplitude * np.sin(
            self.disturbance_frequency * float(time)
        )

    def acceleration(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, time: float) -> np.ndarray:
        applied_tau = np.asarray(tau, dtype=float) + self.disturbance(time)
        return self.robot.acceleration(q, dq, applied_tau)

    def step(self, tau: np.ndarray, dt: float, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrate with constant torque using RK4."""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limits, self.torque_limits)
        applied_tau = tau + self.disturbance(time)
        self.q, self.dq = self.robot.step(self.q, self.dq, applied_tau, dt)
        self.q = wrap_angles(self.q)
        return self.q.copy(), self.dq.copy(), tau, self.disturbance(time)
def wrap_angles(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def angle_error(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    return wrap_angles(np.asarray(q, dtype=float) - np.asarray(q_des, dtype=float))
