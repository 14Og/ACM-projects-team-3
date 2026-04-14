"""Controlled point-mass plant model and related vector helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Plant:
    """2D double integrator: p_dot = v, v_dot = u (unit mass, no friction)."""

    p: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @staticmethod
    def unit_clip(vec: np.ndarray, max_norm: float) -> np.ndarray:
        """Return vec scaled so its norm does not exceed max_norm."""
        norm = np.linalg.norm(vec)
        if norm <= max_norm or norm < 1e-12:
            return vec
        return vec * (max_norm / norm)

    @staticmethod
    def safe_normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        """Return a unit vector, falling back gracefully when near-zero."""
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            return vec / norm
        if fallback is not None:
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm > 1e-9:
                return fallback / fallback_norm
        return np.array([1.0, 0.0], dtype=float)

    def step(self, u: np.ndarray, dt: float) -> None:
        """Advance the plant state by one RK4 integration step.
        
        System dynamics: p' = v, v' = u
        Uses constant control input u throughout the integration step.
        """
        # RK4 stages for p' = v, v' = u
        k1_p = self.v.copy()
        k1_v = u.copy()
        
        k2_p = self.v + 0.5 * dt * k1_v
        k2_v = u.copy()
        
        k3_p = self.v + 0.5 * dt * k2_v
        k3_v = u.copy()
        
        k4_p = self.v + dt * k3_v
        k4_v = u.copy()
        
        # Integrate using RK4 formula
        self.p = self.p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        self.v = self.v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

