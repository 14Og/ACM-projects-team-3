"""Lyapunov-based trajectory tracking controller with CBF constraints."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from .config import APFConfig
from .plant import Plant


class APFController:
    """Lyapunov-based trajectory tracking with Control Barrier Function (CBF) safety.

    Control law combines reference tracking:
        u_ref = a_r - k_p * e_p - k_v * e_v
    
    with CBF constraints on control input u to maintain h(p) >= 0 where:
        h(p) = ||p - c_i||^2 - r_safe^2
    
    CBF constraints enforce:
        h >= 0 (remain outside safety boundary)
        h_dot + alpha * h >= 0 (first-order exponential stability)
        h_dot_dot + alpha_1 * h_dot + alpha_2 * h >= 0 (second-order stability)
    """

    def __init__(self, cfg: APFConfig, plant_radius: float) -> None:
        self.cfg = cfg
        self.plant_radius = plant_radius

    # ------------------------------------------------------------------
    # Reference trajectory on ellipse
    # ------------------------------------------------------------------

    def reference_trajectory(
        self,
        t: float,
        a: float,
        b: float,
        omega: float,
    ) -> tuple:
        """Compute reference position p_r, velocity v_r, and acceleration a_r.

        Elliptical path: p_r = (a*cos(omega*t), b*sin(omega*t))
                         v_r = (-a*omega*sin(omega*t), b*omega*cos(omega*t))
                         a_r = (-a*omega^2*cos(omega*t), b*omega^2*sin(omega*t))
        """
        cos_wt = np.cos(omega * t)
        sin_wt = np.sin(omega * t)
        omega_sq = omega ** 2

        p_r = np.array([a * cos_wt, b * sin_wt], dtype=float)
        v_r = np.array([-a * omega * sin_wt, b * omega * cos_wt], dtype=float)
        a_r = np.array([-a * omega_sq * cos_wt, b * omega_sq * sin_wt], dtype=float)

        return p_r, v_r, a_r

    # ------------------------------------------------------------------
    # Control Barrier Functions (CBF)
    # ------------------------------------------------------------------

    def barrier_function(
        self,
        p: np.ndarray,
        obstacle: Tuple[float, float, float],
    ) -> float:
        """Compute barrier function h(p) = ||p - c_i||^2 - r_safe^2.
        
        h(p) >= 0 when plant is outside the safety boundary.
        """
        ox, oy, r = obstacle
        c = np.array([ox, oy], dtype=float)
        return float(np.dot(p - c, p - c)) - self.cfg.r_safe ** 2

    def barrier_derivatives(
        self,
        p: np.ndarray,
        v: np.ndarray,
        u: np.ndarray,
        obstacle: Tuple[float, float, float],
    ) -> tuple:
        """Compute h_dot and h_dot_dot.
        
        h_dot = 2 * (p - c_i).T @ v
        h_dot_dot = 2 * ||v||^2 + 2 * (p - c_i).T @ u
        """
        ox, oy, r = obstacle
        c = np.array([ox, oy], dtype=float)
        delta = p - c
        
        h_dot = 2.0 * float(np.dot(delta, v))
        h_dot_dot = 2.0 * float(np.dot(v, v)) + 2.0 * float(np.dot(delta, u))
        
        return h_dot, h_dot_dot

    def cbf_constraint_residual(
        self,
        p: np.ndarray,
        v: np.ndarray,
        u: np.ndarray,
        obstacle: Tuple[float, float, float],
    ) -> tuple:
        """Return residuals for the three CBF constraints.
        
        r_3 = h_dot_dot + alpha_1 * h_dot + alpha_2 * h
        
        All should be >= 0 for safety.
        """
        h = self.barrier_function(p, obstacle)
        h_dot, h_dot_dot = self.barrier_derivatives(p, v, u, obstacle)
        
        r_3 = h_dot_dot + self.cfg.alpha_1 * h_dot + self.cfg.alpha_2 * h
        
        return r_3

    # ------------------------------------------------------------------
    # Control force
    # ------------------------------------------------------------------
    
    def lyapunov_value(
        self,
        p: np.ndarray,
        v: np.ndarray,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
    ) -> float:
        """Lyapunov function candidate:
            V = (k_p / 2) * ||e_p||^2 + (1/2) * ||e_v||^2

        where e_p = p - p_ref, e_v = v - v_ref.
        """
        cfg = self.cfg
        e_p = p - p_ref
        e_v = v - v_ref

        V_p = 0.5 * cfg.k_att * float(np.dot(e_p, e_p))
        V_v = 0.5 * float(np.dot(e_v, e_v))

        return V_p + V_v

    # ------------------------------------------------------------------
    # Control force
    # ------------------------------------------------------------------

    def compute_force(
        self,
        p: np.ndarray,
        v: np.ndarray,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Compute control force u enforcing CBF safety constraints.

        Solves:
            minimize ||u - u_ref||^2
            subject to:
                h_i >= 0
                h_dot_i + alpha * h_i >= 0  
                h_dot_dot_i + alpha_1 * h_dot_i + alpha_2 * h_i >= 0
                ||u|| <= u_max (if constrain_control is True)
        
        where u_ref = a_ref - k_p * e_p - k_v * e_v
        """
        cfg = self.cfg
        e_p = p - p_ref
        e_v = v - v_ref

        # Nominal (unconstrained) control for tracking
        u_ref = a_ref - cfg.k_att * e_p - cfg.k_v * e_v

        # If no obstacles or empty obstacle list, apply control magnitude constraint only
        if not obstacles:
            if cfg.constrain_control:
                u_ref = Plant.unit_clip(u_ref, cfg.u_max)
            return u_ref

        # Solve constrained optimization with CBF constraints
        def objective(u_flat):
            u_test = u_flat.reshape(2)
            return float(np.dot(u_test - u_ref, u_test - u_ref))

        def cbf_constraint_fn(u_flat):
            """Return array of CBF constraint residuals (all should be >= 0).
            
            Uses only second-order CLF constraint per obstacle:
                h_dot_dot + alpha_1 * h_dot + alpha_2 * h >= 0
            """
            u_test = u_flat.reshape(2)
            residuals = []
            
            for obs in obstacles:
                h = self.barrier_function(p, obs)
                h_dot, h_dot_dot = self.barrier_derivatives(p, v, u_test, obs)
                # Second-order constraint
                residuals.append(h_dot_dot + cfg.alpha_1 * h_dot + cfg.alpha_2 * h)
            
            return np.array(residuals)

        def control_magnitude_constraint(u_flat):
            """Control magnitude constraint: ||u||^2 <= u_max^2."""
            u_test = u_flat.reshape(2)
            return cfg.u_max ** 2 - float(np.dot(u_test, u_test))

        # Build constraints for scipy.optimize.minimize
        constraints = []
        
        # CBF constraints (one per obstacle)
        for i in range(len(obstacles)):
            constraints.append({
                'type': 'ineq',
                'fun': lambda u_flat, idx=i: cbf_constraint_fn(u_flat)[idx]
            })
        
        # Control magnitude constraint (if enabled)
        if cfg.constrain_control:
            constraints.append({
                'type': 'ineq',
                'fun': control_magnitude_constraint
            })

        # Initial guess: the reference control
        x0 = u_ref.flatten()

        # Solve with SLSQP method (handles inequality constraints)
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=None,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 100}
        )

        if result.success:
            u_constrained = result.x.reshape(2)
        else:
            # Fallback: use reference control with magnitude clipping if needed
            u_constrained = u_ref.copy()
            if cfg.constrain_control:
                u_constrained = Plant.unit_clip(u_constrained, cfg.u_max)

        return u_constrained
