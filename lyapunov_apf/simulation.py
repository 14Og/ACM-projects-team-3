"""Scenario randomisation, reference trajectory, and simulation loop."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import EnvConfig, EpisodeState, SimConfig
from .controller import APFController
from .plant import Plant


class SimulationEngine:
    """Simulation service for scenario generation and closed-loop rollout."""

    def __init__(self, env: EnvConfig, sim: SimConfig) -> None:
        self.env = env
        self.sim = sim

    def ellipse_point(self, theta: float, scale: float = 1.0) -> np.ndarray:
        return np.array(
            [self.env.cx + scale * self.env.a * np.cos(theta),
             self.env.cy + scale * self.env.b * np.sin(theta)],
            dtype=float,
        )

    def ellipse_level(self, p: np.ndarray) -> float:
        """Return <= 1 when p is inside/on the ellipse, > 1 outside."""
        x = (p[0] - self.env.cx) / self.env.a
        y = (p[1] - self.env.cy) / self.env.b
        return x * x + y * y

    def reference_state(self, t: float, episode: EpisodeState) -> tuple:
        """Reference position, velocity, and acceleration on the ellipse at time t."""
        theta = episode.theta0 + episode.omega * t
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        omega = episode.omega
        omega_sq = omega ** 2

        p_ref = np.array(
            [self.env.cx + self.env.a * cos_theta,
             self.env.cy + self.env.b * sin_theta],
            dtype=float,
        )
        v_ref = np.array(
            [-self.env.a * omega * sin_theta,
              self.env.b * omega * cos_theta],
            dtype=float,
        )
        a_ref = np.array(
            [-self.env.a * omega_sq * cos_theta,
              self.env.b * omega_sq * sin_theta],
            dtype=float,
        )
        return p_ref, v_ref, a_ref

    def _build_obstacles(self, rng: np.random.Generator) -> List[Tuple[float, float, float]]:
        """Build obstacle list from config bases with optional drift.
        
        Returns list of (x, y, radius) tuples.
        If obstacle_bases is empty, returns empty list.
        """
        if not self.env.obstacle_bases:
            return []
        
        obstacles: List[Tuple[float, float, float]] = []
        for ox_base, oy_base in self.env.obstacle_bases:
            # Apply drift randomization
            dx = float(rng.uniform(-self.env.obstacle_drift, self.env.obstacle_drift))
            dy = float(rng.uniform(-self.env.obstacle_drift, self.env.obstacle_drift))
            ox = ox_base + dx
            oy = oy_base + dy
            r = self.env.obstacle_radius
            obstacles.append((ox, oy, r))
        
        return obstacles

    def randomize_episode(self, rng: np.random.Generator) -> EpisodeState:
        """Sample a fully randomised episode with obstacles."""
        theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
        omega = float(rng.uniform(*self.env.omega_range))
        t_final = self.sim.num_cycles * (2.0 * np.pi / abs(omega))

        # Build obstacles with randomized positions
        obstacles = self._build_obstacles(rng)

        # Sample plant start position (avoiding obstacles)
        p_target0 = self.ellipse_point(theta0)
        for _ in range(1200):
            th = float(rng.uniform(0.0, 2.0 * np.pi))
            scale = float(rng.uniform(*self.env.plant_spawn_scale_range))
            p = self.ellipse_point(th, scale=scale)
            if self.ellipse_level(p) <= 1.01:
                continue
            if np.linalg.norm(p - p_target0) < self.env.min_target_plant_start_distance:
                continue
            # Check clearance from obstacles
            safe = True
            for ox, oy, r in obstacles:
                if np.linalg.norm(p - np.array([ox, oy], dtype=float)) <= r + self.env.plant_radius + self.env.plant_spawn_clearance:
                    safe = False
                    break
            if not safe:
                continue
            p0 = (float(p[0]), float(p[1]))
            break
        else:
            raise RuntimeError("Failed to sample a plant start position.")

        return EpisodeState(
            theta0=theta0,
            omega=omega,
            t_final=t_final,
            p0=p0,
            v0=(0.0, 0.0),
            obstacles=obstacles,
        )

    def next_episode(self, prev: EpisodeState, rng: np.random.Generator) -> EpisodeState:
        """Start a new episode after the previous one ends with new obstacles."""
        theta0 = prev.theta0
        omega = prev.omega
        t_final = prev.t_final

        # Build obstacles with new randomized positions
        obstacles = self._build_obstacles(rng)

        # Sample plant start position (avoiding obstacles)
        p_target0 = self.ellipse_point(theta0)
        for _ in range(1200):
            th = float(rng.uniform(0.0, 2.0 * np.pi))
            scale = float(rng.uniform(*self.env.plant_spawn_scale_range))
            p = self.ellipse_point(th, scale=scale)
            if self.ellipse_level(p) <= 1.01:
                continue
            if np.linalg.norm(p - p_target0) < self.env.min_target_plant_start_distance:
                continue
            # Check clearance from obstacles
            safe = True
            for ox, oy, r in obstacles:
                if np.linalg.norm(p - np.array([ox, oy], dtype=float)) <= r + self.env.plant_radius + self.env.plant_spawn_clearance:
                    safe = False
                    break
            if not safe:
                continue
            p0 = (float(p[0]), float(p[1]))
            break
        else:
            raise RuntimeError("Failed to sample a plant start position.")

        return EpisodeState(
            theta0=theta0,
            omega=omega,
            t_final=t_final,
            p0=p0,
            v0=(0.0, 0.0),
            obstacles=obstacles,
        )

    def run_simulation(
        self,
        episode: EpisodeState,
        controller: APFController,
    ) -> dict[str, np.ndarray]:
        """Integrate the closed-loop system and return trajectory data."""
        n = self.sim.steps_per_episode
        t = np.linspace(0.0, episode.t_final, n)
        dt = float(t[1] - t[0])

        p_arr = np.zeros((n, 2))
        v_arr = np.zeros((n, 2))
        u_arr = np.zeros((n, 2))
        p_ref_arr = np.zeros((n, 2))
        v_ref_arr = np.zeros((n, 2))
        a_ref_arr = np.zeros((n, 2))
        V_arr = np.zeros(n)

        plant = Plant(
            p=np.array(episode.p0, dtype=float),
            v=np.array(episode.v0, dtype=float),
        )
        p_arr[0] = plant.p
        v_arr[0] = plant.v
        p_ref_arr[0], v_ref_arr[0], a_ref_arr[0] = self.reference_state(t[0], episode)
        V_arr[0] = controller.lyapunov_value(plant.p, plant.v, p_ref_arr[0], v_ref_arr[0])

        for k in range(n - 1):
            p_ref_arr[k], v_ref_arr[k], a_ref_arr[k] = self.reference_state(t[k], episode)
            u = controller.compute_force(
                plant.p, plant.v, p_ref_arr[k], v_ref_arr[k], a_ref_arr[k], episode.obstacles
            )
            u_arr[k] = u

            plant.step(u, dt)

            p_arr[k + 1] = plant.p
            v_arr[k + 1] = plant.v
            p_ref_arr[k + 1], v_ref_arr[k + 1], a_ref_arr[k + 1] = self.reference_state(t[k + 1], episode)
            V_arr[k + 1] = controller.lyapunov_value(
                plant.p, plant.v, p_ref_arr[k + 1], v_ref_arr[k + 1]
            )

        u_arr[-1] = controller.compute_force(
            plant.p, plant.v, p_ref_arr[-1], v_ref_arr[-1], a_ref_arr[-1], episode.obstacles
        )

        return {
            "t": t,
            "p": p_arr,
            "v": v_arr,
            "u": u_arr,
            "p_ref": p_ref_arr,
            "v_ref": v_ref_arr,
            "a_ref": a_ref_arr,
            "V": V_arr,
            "err": np.linalg.norm(p_arr - p_ref_arr, axis=1),
            "err_v": np.linalg.norm(v_arr - v_ref_arr, axis=1),
            "speed": np.linalg.norm(v_arr, axis=1),
            "accel": np.linalg.norm(u_arr, axis=1),
        }

