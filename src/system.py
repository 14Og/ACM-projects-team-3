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
        # Handle empty obstacle_centers (no obstacles)
        if obstacle_centers.size == 0:
            return ClearanceResult(min_clearance=1e6, collision=False)
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
    """Full planar manipulator dynamics with unknown diagonal inertia and damping."""

    GRAVITY = 9.81

    def __init__(self, q0: np.ndarray, cfg: DynamicsConfig, robot_cfg: RobotConfig) -> None:
        self.q = wrap_angles(np.asarray(q0, dtype=float))
        self.dq = np.zeros_like(self.q)
        self.link_lengths = np.asarray(robot_cfg.link_lengths, dtype=float)
        self.link_lengths_m = self.link_lengths / 200.0
        self.masses = np.asarray(cfg.link_masses, dtype=float)
        self.lc = self.link_lengths_m / 2.0
        self.I = (1.0 / 12.0) * self.masses * (self.link_lengths_m ** 2)
        self.n_joints = self.q.size
        self.torque_limits = np.asarray(cfg.torque_limits, dtype=float)
        self.damping = np.asarray(cfg.true_damping, dtype=float)
        self.disturbance_constant = np.asarray(cfg.disturbance_constant, dtype=float)
        self.disturbance_amplitude = np.asarray(cfg.disturbance_amplitude, dtype=float)
        self.disturbance_frequency = np.asarray(cfg.disturbance_frequency, dtype=float)

    def disturbance(self, time: float) -> np.ndarray:
        return self.disturbance_constant + self.disturbance_amplitude * np.sin(
            self.disturbance_frequency * float(time)
        )

    def acceleration(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, time: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        tau = np.asarray(tau, dtype=float)
        M, G = _dynamics_matrices(q, self.masses, self.link_lengths_m, self.lc, self.I, self.GRAVITY)
        C = _coriolis_vector(q, dq, self.masses, self.link_lengths_m, self.lc)
        damping = self.damping * dq
        rhs = tau + self.disturbance(time) - G - C - damping
        return np.linalg.solve(M, rhs)

    def step(self, tau: np.ndarray, dt: float, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrate with constant torque using RK4 using smaller physics substeps."""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limits, self.torque_limits)
        sub_dt = min(0.0005, dt)
        steps = max(1, int(round(dt / sub_dt)))
        sub_dt = dt / steps

        for step in range(steps):
            q0 = self.q.copy()
            dq0 = self.dq.copy()

            def rhs(q: np.ndarray, dq: np.ndarray, stage_time: float) -> tuple[np.ndarray, np.ndarray]:
                return dq, self.acceleration(q, dq, tau, time + step * sub_dt)

            k1_q, k1_dq = rhs(q0, dq0, time + step * sub_dt)
            k2_q, k2_dq = rhs(
                q0 + 0.5 * sub_dt * k1_q,
                dq0 + 0.5 * sub_dt * k1_dq,
                time + step * sub_dt + 0.5 * sub_dt,
            )
            k3_q, k3_dq = rhs(
                q0 + 0.5 * sub_dt * k2_q,
                dq0 + 0.5 * sub_dt * k2_dq,
                time + step * sub_dt + 0.5 * sub_dt,
            )
            k4_q, k4_dq = rhs(
                q0 + sub_dt * k3_q,
                dq0 + sub_dt * k3_dq,
                time + step * sub_dt + sub_dt,
            )

            self.q = wrap_angles(
                q0 + (sub_dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            )
            self.dq = np.clip(
                dq0 + (sub_dt / 6.0) * (k1_dq + 2.0 * k2_dq + 2.0 * k3_dq + k4_dq),
                -15.0,
                15.0,
            )

        return self.q.copy(), self.dq.copy(), tau, self.disturbance(time)


def _dynamics_matrices(q: np.ndarray, masses: np.ndarray, lengths: np.ndarray, lc: np.ndarray, I: np.ndarray, g: float) -> tuple[np.ndarray, np.ndarray]:
    m1, m2, m3 = masses
    l1, l2, l3 = lengths
    lc1, lc2, lc3 = lc
    I1, I2, I3 = I
    th1, th2, th3 = q

    c2 = np.cos(th2)
    c3 = np.cos(th3)
    c23 = np.cos(th2 + th3)

    M = np.zeros((3, 3), dtype=float)
    M[2, 2] = I3 + m3 * lc3**2
    M[1, 2] = M[2, 2] + m3 * l2 * lc3 * c3
    M[0, 2] = M[2, 2] + m3 * l2 * lc3 * c3 + m3 * l1 * lc3 * c23
    M[2, 1] = M[1, 2]
    M[1, 1] = I2 + m2 * lc2**2 + I3 + m3 * (l2**2 + lc3**2 + 2.0 * l2 * lc3 * c3)
    M[0, 1] = M[1, 1] + (m2 * l1 * lc2 + m3 * l1 * l2) * c2 + m3 * l1 * lc3 * c23
    M[2, 0] = M[0, 2]
    M[1, 0] = M[0, 1]
    M[0, 0] = (
        I1
        + m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2)
        + m3 * (l1**2 + l2**2 + lc3**2 + 2.0 * l1 * l2 * c2 + 2.0 * l2 * lc3 * c3 + 2.0 * l1 * lc3 * c23)
    )

    G = np.zeros(3, dtype=float)
    G[2] = m3 * g * lc3 * np.cos(th1 + th2 + th3)
    G[1] = (m2 * lc2 + m3 * l2) * g * np.cos(th1 + th2) + G[2]
    G[0] = (m1 * lc1 + (m2 + m3) * l1) * g * np.cos(th1) + G[1]
    return M, G


def _coriolis_vector(q: np.ndarray, dq: np.ndarray, masses: np.ndarray, lengths: np.ndarray, lc: np.ndarray) -> np.ndarray:
    _, m2, m3 = masses
    l1, l2, _ = lengths
    _, lc2, lc3 = lc
    th2, th3 = q[1], q[2]
    dq0, dq1, dq2 = dq[0], dq[1], dq[2]

    f1 = (m2 * l1 * lc2 + m3 * l1 * l2) * np.sin(th2)
    f2 = m3 * l1 * lc3 * np.sin(th2 + th3)
    f3 = m3 * l2 * lc3 * np.sin(th3)

    Cdq = np.zeros(3, dtype=float)
    Cdq[0] = (-2.0 * (f1 + f2) * dq0 * dq1 - 2.0 * (f3 + f2) * dq0 * dq2
              - (f1 + f2) * dq1 * dq1 - 2.0 * (f3 + f2) * dq1 * dq2
              - (f3 + f2) * dq2 * dq2)
    Cdq[1] = ((f1 + f2) * dq0 * dq0 - 2.0 * f3 * dq0 * dq2 - 2.0 * f3 * dq1 * dq2 - f3 * dq2 * dq2)
    Cdq[2] = ((f3 + f2) * dq0 * dq0 + 2.0 * f3 * dq0 * dq1 + f3 * dq1 * dq1)
    return Cdq


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


def angle_error(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    return wrap_angles(np.asarray(q, dtype=float) - np.asarray(q_des, dtype=float))
