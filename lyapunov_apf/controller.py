"""CLF + HOCBF tracking controller solved as a QP."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


from .config import APFConfig


class CLFCBFController:
    """Reference-tracking controller with HOCBF safety filter.

    Nominal (CLF-based) control:
        u_des = a_ref - k_p * e_p - k_v * e_v        (e_p = p - p_ref, e_v = v - v_ref)

    HOCBF safety constraint per obstacle i with safety radius
    r_safe_i = r_obs_i + r_plant + safe_margin:
        h_i(p)   = ||p - c_i||^2 - r_safe_i^2
        h_dot_i  = 2 (p - c_i)^T v
        h_ddot_i = 2 ||v||^2 + 2 (p - c_i)^T u
        h_ddot_i + alpha_1 * h_dot_i + alpha_2 * h_i  >=  0

    The constraint is linear in u:
        2 (p - c_i)^T u  >=  -2 ||v||^2 - 2 alpha_1 (p - c_i)^T v - alpha_2 h_i

    Final control solves a QP:
        min  (1/2) ||u - u_des||^2 + rho * s
        s.t. A u + s * 1 >= b              (CBF, with single shared slack)
             -u_max <= u_j <= u_max         (box, when constrain_control)
             s >= 0
    The slack `s` keeps the QP feasible if no safe control exists; the
    penalty `rho` is large enough that s == 0 whenever a strictly safe
    control is available.
    """

    SLACK_PENALTY: float = 1.0e4

    def __init__(self, cfg: APFConfig, plant_radius: float) -> None:
        self.cfg = cfg
        self.plant_radius = plant_radius

    # ------------------------------------------------------------------
    # Lyapunov function and barriers
    # ------------------------------------------------------------------

    def lyapunov_value(
        self,
        p: np.ndarray,
        v: np.ndarray,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
    ) -> float:
        """V = (k_p / 2) ||e_p||^2 + (1/2) ||e_v||^2."""
        e_p = p - p_ref
        e_v = v - v_ref
        return 0.5 * self.cfg.k_att * float(np.dot(e_p, e_p)) + 0.5 * float(np.dot(e_v, e_v))

    def barrier_value(self, p: np.ndarray, obstacle: Tuple[float, float, float]) -> float:
        """h_i(p) = ||p - c_i||^2 - r_safe_i^2 with per-obstacle r_safe."""
        ox, oy, r = obstacle
        delta = p - np.array([ox, oy], dtype=float)
        r_safe = r + self.plant_radius + self.cfg.safe_margin
        return float(np.dot(delta, delta)) - r_safe ** 2

    # ------------------------------------------------------------------
    # Control force (QP with slack-relaxed CBF + box bounds)
    # ------------------------------------------------------------------

    def _cbf_rows(
        self,
        p: np.ndarray,
        v: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (A, b) such that the HOCBF constraints read A u >= b."""
        cfg = self.cfg
        v_norm_sq = float(np.dot(v, v))
        rows: List[np.ndarray] = []
        rhs: List[float] = []
        for ox, oy, r in obstacles:
            c = np.array([ox, oy], dtype=float)
            delta = p - c
            r_safe = r + self.plant_radius + cfg.safe_margin
            h = float(np.dot(delta, delta)) - r_safe ** 2
            h_dot = 2.0 * float(np.dot(delta, v))
            rows.append(2.0 * delta)
            rhs.append(-2.0 * v_norm_sq - cfg.alpha_1 * h_dot - cfg.alpha_2 * h)
        if not rows:
            return np.zeros((0, 2)), np.zeros(0)
        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)

    def compute_force(
        self,
        p: np.ndarray,
        v: np.ndarray,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        cfg = self.cfg
        e_p = p - p_ref
        e_v = v - v_ref

        u_des = -cfg.k_att * e_p - cfg.k_v * e_v
        if cfg.feedforward:
            u_des = u_des + a_ref

        A, b = self._cbf_rows(p, v, obstacles)

        # No obstacles: nominal control with optional box clipping.
        if A.shape[0] == 0:
            if cfg.constrain_control:
                return np.clip(u_des, -cfg.u_max, cfg.u_max)
            return u_des

        # QP variables z = [u_x, u_y, s]; minimise (1/2)||u-u_des||^2 + rho s.
        rho = self.SLACK_PENALTY
        n = A.shape[0]
        A_aug = np.hstack([A, np.ones((n, 1))])
        cbf = LinearConstraint(A_aug, lb=b, ub=np.inf)

        if cfg.constrain_control:
            lb = np.array([-cfg.u_max, -cfg.u_max, 0.0])
            ub = np.array([cfg.u_max, cfg.u_max, np.inf])
        else:
            lb = np.array([-np.inf, -np.inf, 0.0])
            ub = np.array([np.inf, np.inf, np.inf])
        bounds = Bounds(lb, ub)

        def objective(z: np.ndarray) -> float:
            diff = z[:2] - u_des
            return 0.5 * float(np.dot(diff, diff)) + rho * z[2]

        def gradient(z: np.ndarray) -> np.ndarray:
            return np.array([z[0] - u_des[0], z[1] - u_des[1], rho])

        z0 = np.array([u_des[0], u_des[1], 0.0])
        # Warm-start slack so the initial point is feasible whenever possible.
        residual = b - A @ u_des
        if np.any(residual > 0.0):
            z0[2] = float(np.max(residual))

        result = minimize(
            objective,
            z0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=[cbf],
            options={"ftol": 1e-8, "maxiter": 80},
        )
        u = result.x[:2]
        if cfg.constrain_control:
            # Box bounds are already in `bounds`; clip defensively against
            # numerical SLSQP overshoot.
            u = np.clip(u, -cfg.u_max, cfg.u_max)
        return u
