"""Three-link planar rigid-body dynamics for the simulated robot plant."""

from __future__ import annotations

import numpy as np
try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class RobotDynamics3DOF:
    """Rigid-body 3DOF manipulator dynamics with gravity, Coriolis, and damping."""

    def __init__(self, masses: np.ndarray, lengths: np.ndarray, damping: np.ndarray) -> None:
        self.masses = np.asarray(masses, dtype=float)
        self.lengths = np.asarray(lengths, dtype=float) / 100
        self.damping = np.asarray(damping, dtype=float)
        self.centers = self.lengths / 2.0 
        self.inertias = (1.0 / 12.0) * self.masses * self.lengths**2
        self.gravity = 9.81

    def get_matrices(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return get_matrices_fast(
            np.asarray(q, dtype=float),
            self.masses,
            self.lengths,
            self.centers,
            self.inertias,
            self.gravity,
        )

    def acceleration(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        _, q_dd = dynamics_fast(
            np.asarray(q, dtype=float),
            np.asarray(dq, dtype=float),
            np.asarray(tau, dtype=float),
            self.masses,
            self.lengths,
            self.centers,
            self.inertias,
            self.gravity,
            self.damping,
        )
        return q_dd

    def step(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return update_rk4_fast(
            np.asarray(q, dtype=float),
            np.asarray(dq, dtype=float),
            np.asarray(tau, dtype=float),
            float(dt),
            self.masses,
            self.lengths,
            self.centers,
            self.inertias,
            self.gravity,
            self.damping,
        )


@njit(cache=True)
def get_matrices_fast(q, masses, lengths, centers, inertias, gravity):
    m1, m2, m3 = masses
    l1, l2, _ = lengths
    lc1, lc2, lc3 = centers
    I1, I2, I3 = inertias
    th1, th2, th3 = q

    c2 = np.cos(th2)
    c3 = np.cos(th3)
    c23 = np.cos(th2 + th3)

    M = np.zeros((3, 3))

    m3_l1_lc3 = m3 * l1 * lc3
    m3_l2_lc3 = m3 * l2 * lc3
    m2_l1_lc2 = m2 * l1 * lc2
    m3_l1_l2 = m3 * l1 * l2

    M[2, 2] = I3 + m3 * lc3**2
    M[1, 2] = M[2, 2] + m3_l2_lc3 * c3
    M[0, 2] = M[2, 2] + m3_l2_lc3 * c3 + m3_l1_lc3 * c23

    M[2, 1] = M[1, 2]
    M[1, 1] = I2 + m2 * lc2**2 + I3 + m3 * (l2**2 + lc3**2 + 2.0 * l2 * lc3 * c3)
    M[0, 1] = M[1, 1] + (m2_l1_lc2 + m3_l1_l2) * c2 + m3_l1_lc3 * c23

    M[2, 0] = M[0, 2]
    M[1, 0] = M[0, 1]
    M[0, 0] = (
        I1
        + m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2)
        + m3 * (l1**2 + l2**2 + lc3**2 + 2.0 * l1 * l2 * c2 + 2.0 * l2 * lc3 * c3 + 2.0 * l1 * lc3 * c23)
    )

    G = np.zeros(3)
    G[2] = m3 * gravity * lc3 * np.cos(th1 + th2 + th3)
    G[1] = (m2 * lc2 + m3 * l2) * gravity * np.cos(th1 + th2) + G[2]
    G[0] = (m1 * lc1 + (m2 + m3) * l1) * gravity * np.cos(th1) + G[1]
    return M, G


@njit(cache=True)
def coriolis_vector(q, dq, masses, lengths, centers):
    _, m2, m3 = masses
    l1, l2, _ = lengths
    _, lc2, lc3 = centers
    th2, th3 = q[1], q[2]
    dq0, dq1, dq2 = dq[0], dq[1], dq[2]

    f1 = (m2 * l1 * lc2 + m3 * l1 * l2) * np.sin(th2)
    f2 = m3 * l1 * lc3 * np.sin(th2 + th3)
    f3 = m3 * l2 * lc3 * np.sin(th3)

    cdq = np.zeros(3)
    cdq[0] = (
        -2.0 * (f1 + f2) * dq0 * dq1
        - 2.0 * (f3 + f2) * dq0 * dq2
        - (f1 + f2) * dq1 * dq1
        - 2.0 * (f3 + f2) * dq1 * dq2
        - (f3 + f2) * dq2 * dq2
    )
    cdq[1] = (
        (f1 + f2) * dq0 * dq0
        - 2.0 * f3 * dq0 * dq2
        - 2.0 * f3 * dq1 * dq2
        - f3 * dq2 * dq2
    )
    cdq[2] = (
        (f3 + f2) * dq0 * dq0
        + 2.0 * f3 * dq0 * dq1
        + f3 * dq1 * dq1
    )
    return cdq


@njit(cache=True)
def dynamics_fast(q, dq, tau, masses, lengths, centers, inertias, gravity, damping):
    M, G = get_matrices_fast(q, masses, lengths, centers, inertias, gravity)
    M += np.eye(3) * 1e-4
    cdq = coriolis_vector(q, dq, masses, lengths, centers)
    rhs = tau - G - cdq - damping * dq
    q_dd = np.linalg.solve(M, rhs)
    return dq, q_dd


@njit(cache=True)
def update_rk4_fast(q, dq, tau, dt, masses, lengths, centers, inertias, gravity, damping):
    v1, a1 = dynamics_fast(q, dq, tau, masses, lengths, centers, inertias, gravity, damping)
    v2, a2 = dynamics_fast(
        q + v1 * dt / 2.0,
        dq + a1 * dt / 2.0,
        tau,
        masses,
        lengths,
        centers,
        inertias,
        gravity,
        damping,
    )
    v3, a3 = dynamics_fast(
        q + v2 * dt / 2.0,
        dq + a2 * dt / 2.0,
        tau,
        masses,
        lengths,
        centers,
        inertias,
        gravity,
        damping,
    )
    v4, a4 = dynamics_fast(
        q + v3 * dt,
        dq + a3 * dt,
        tau,
        masses,
        lengths,
        centers,
        inertias,
        gravity,
        damping,
    )

    q_new = q + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
    dq_new = dq + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    dq_new = np.clip(dq_new, -15.0, 15.0)
    return q_new, dq_new
