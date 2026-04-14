"""Lyapunov-based APF controller for trajectory tracking."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import APFConfig
from .plant import Plant


class APFController:
    """Energy-based / passivity-based controller with APF obstacle repulsion.

    Control law (pure PBC: potential shaping + damping injection):
        u = -k_att * e_p  -  k_v * e_v  -  grad V_rep

    where e_p = p - p_ref, e_v = v - v_ref, and V_rep is the sum of APF
    repulsive potentials. Optionally clips ||u|| <= u_max when
    cfg.constrain_control is True.
    """

    def __init__(self, cfg: APFConfig, plant_radius: float) -> None:
        self.cfg = cfg
        self.plant_radius = plant_radius

    # ------------------------------------------------------------------
    # Potential field components
    # ------------------------------------------------------------------

    def repulsive_gradient(
        self,
        p: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Gradient of the sum of repulsive potentials:
            V_rep_i = (1/2) * k_rep * (1/d_eff - 1/influence_eff)^2
        where d_eff is signed clearance from the safety boundary.
        """
        cfg = self.cfg
        u_rep = np.zeros(2, dtype=float)
        for ox, oy, r in obstacles:
            c = np.array([ox, oy], dtype=float)
            delta = p - c
            center_dist = np.linalg.norm(delta)
            if center_dist < 1e-12:
                continue
            d = center_dist - (r + self.plant_radius)
            if d >= cfg.influence_radius:
                continue
            d_eff = max(d - cfg.safe_margin, cfg.eps_dist)
            influence_eff = max(cfg.influence_radius - cfg.safe_margin, cfg.eps_dist)
            grad_d = delta / center_dist
            mag = cfg.k_rep * (1.0 / d_eff - 1.0 / influence_eff) / (d_eff ** 2)
            u_rep += mag * grad_d
        return u_rep

    def lyapunov_value(
        self,
        p: np.ndarray,
        v: np.ndarray,
        p_ref: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> float:
        """Total Lyapunov candidate:
            V_total = V_att + V_rep + (1/2)||v||^2

            V_att   = (1/2) * k_att * ||p - p_ref||^2
            V_rep   = sum_i (1/2) * k_rep * (1/d_eff_i - 1/influence_eff)^2
        """
        cfg = self.cfg

        e_p = p - p_ref
        V_att = 0.5 * cfg.k_att * float(np.dot(e_p, e_p))

        V_rep = 0.0
        for ox, oy, r in obstacles:
            c = np.array([ox, oy], dtype=float)
            center_dist = float(np.linalg.norm(p - c))
            d = center_dist - (r + self.plant_radius)
            if d >= cfg.influence_radius:
                continue
            d_eff = max(d - cfg.safe_margin, cfg.eps_dist)
            influence_eff = max(cfg.influence_radius - cfg.safe_margin, cfg.eps_dist)
            V_rep += 0.5 * cfg.k_rep * (1.0 / d_eff - 1.0 / influence_eff) ** 2

        V_kin = 0.5 * float(np.dot(v, v))
        return V_att + V_rep + V_kin

    # ------------------------------------------------------------------
    # Control force
    # ------------------------------------------------------------------

    def compute_force(
        self,
        p: np.ndarray,
        v: np.ndarray,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Compute control force u at current plant state."""
        cfg = self.cfg
        e_p = p - p_ref
        e_v = v - v_ref

        # Potential shaping (attractive) + damping injection + repulsive gradient.
        u = -cfg.k_att * e_p - cfg.k_v * e_v + self.repulsive_gradient(p, obstacles)

        if cfg.constrain_control:
            u = Plant.unit_clip(u, cfg.u_max)
        return u
