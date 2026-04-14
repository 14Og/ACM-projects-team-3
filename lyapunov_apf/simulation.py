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

    def reference_state(self, t: float, episode: EpisodeState) -> Tuple[np.ndarray, np.ndarray]:
        """Reference position and velocity on the ellipse at time t."""
        theta = episode.theta0 + episode.omega * t
        p_ref = np.array(
            [self.env.cx + self.env.a * np.cos(theta),
             self.env.cy + self.env.b * np.sin(theta)],
            dtype=float,
        )
        v_ref = np.array(
            [-self.env.a * episode.omega * np.sin(theta),
              self.env.b * episode.omega * np.cos(theta)],
            dtype=float,
        )
        return p_ref, v_ref

    def reference_acceleration(self, t: float, episode: EpisodeState) -> np.ndarray:
        """Reference acceleration a_ref = p̈_ref on the ellipse at time t."""
        theta = episode.theta0 + episode.omega * t
        return np.array(
            [-self.env.a * episode.omega ** 2 * np.cos(theta),
             -self.env.b * episode.omega ** 2 * np.sin(theta)],
            dtype=float,
        )

    def _safe_from_obstacles(
        self,
        p: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        entity_radius: float,
        extra_clearance: float,
    ) -> bool:
        for ox, oy, r in obstacles:
            if np.linalg.norm(p - np.array([ox, oy], dtype=float)) <= r + entity_radius + extra_clearance:
                return False
        return True

    def _drift_obstacles(self, rng: np.random.Generator) -> List[Tuple[float, float, float]]:
        """Sample drifted obstacle positions from base positions."""
        obstacles: List[Tuple[float, float, float]] = []
        for ox, oy in self.env.obstacle_bases:
            dx = float(rng.uniform(-self.env.obstacle_drift, self.env.obstacle_drift))
            dy = float(rng.uniform(-self.env.obstacle_drift, self.env.obstacle_drift))
            obstacles.append((ox + dx, oy + dy, self.env.obstacle_radius))
        return obstacles

    def _sample_plant_start(
        self,
        obstacles: List[Tuple[float, float, float]],
        theta0: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        """Sample a valid plant start position."""
        p_target0 = self.ellipse_point(theta0)
        for _ in range(1200):
            th = float(rng.uniform(0.0, 2.0 * np.pi))
            scale = float(rng.uniform(*self.env.plant_spawn_scale_range))
            p = self.ellipse_point(th, scale=scale)
            if self.ellipse_level(p) <= 1.01:
                continue
            if np.linalg.norm(p - p_target0) < self.env.min_target_plant_start_distance:
                continue
            if not self._safe_from_obstacles(p, obstacles, self.env.plant_radius, self.env.plant_spawn_clearance):
                continue
            return (float(p[0]), float(p[1]))
        raise RuntimeError("Failed to sample a plant start position.")

    def randomize_episode(self, rng: np.random.Generator) -> EpisodeState:
        """Sample a fully randomised episode."""
        obstacles = self._drift_obstacles(rng)

        for _ in range(800):
            theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
            p_t = self.ellipse_point(theta0)
            if self._safe_from_obstacles(p_t, obstacles, self.env.target_radius, self.env.target_obstacle_clearance):
                break
        else:
            raise RuntimeError("Failed to sample a target start angle away from obstacles.")

        omega = float(rng.uniform(*self.env.omega_range))
        t_final = self.sim.num_cycles * (2.0 * np.pi / abs(omega))
        p0 = self._sample_plant_start(obstacles, theta0, rng)

        return EpisodeState(
            theta0=theta0,
            omega=omega,
            t_final=t_final,
            p0=p0,
            v0=(0.0, 0.0),
            obstacles=obstacles,
        )

    def next_episode(self, prev: EpisodeState, rng: np.random.Generator) -> EpisodeState:
        """Start a new episode after the previous one ends."""
        obstacles = self._drift_obstacles(rng)
        p0 = self._sample_plant_start(obstacles, prev.theta0, rng)

        return EpisodeState(
            theta0=prev.theta0,
            omega=prev.omega,
            t_final=prev.t_final,
            p0=p0,
            v0=(0.0, 0.0),
            obstacles=obstacles,
        )

    def enforce_obstacle_clearance(
        self,
        p: np.ndarray,
        v: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        plant_radius: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hard-project the plant outside all obstacle safety boundaries."""
        p_new = p.copy()
        v_new = v.copy()
        tol = 1e-4

        for _ in range(3):
            changed = False
            for ox, oy, r in obstacles:
                c = np.array([ox, oy], dtype=float)
                delta = p_new - c
                dist = float(np.linalg.norm(delta))
                min_dist = r + plant_radius
                if dist < max(min_dist, 1e-12):
                    normal = delta / dist if dist > 1e-9 else np.array([1.0, 0.0], dtype=float)
                    p_new = c + (min_dist + tol) * normal
                    vn = float(np.dot(v_new, normal))
                    if vn < 0.0:
                        v_new = v_new - vn * normal
                    changed = True
            if not changed:
                break

        return p_new, v_new

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
        V_arr = np.zeros(n)

        plant = Plant(
            p=np.array(episode.p0, dtype=float),
            v=np.array(episode.v0, dtype=float),
        )
        p_arr[0] = plant.p
        v_arr[0] = plant.v
        p_ref_arr[0], v_ref_arr[0] = self.reference_state(t[0], episode)
        V_arr[0] = controller.lyapunov_value(plant.p, plant.v, p_ref_arr[0], episode.obstacles, v_ref_arr[0])

        for k in range(n - 1):
            p_ref_arr[k], v_ref_arr[k] = self.reference_state(t[k], episode)
            a_ref = self.reference_acceleration(t[k], episode)
            u = controller.compute_force(
                plant.p, plant.v, p_ref_arr[k], v_ref_arr[k], episode.obstacles, a_ref
            )
            u_arr[k] = u

            plant.step(u, dt)
            plant.p, plant.v = self.enforce_obstacle_clearance(
                plant.p, plant.v, episode.obstacles, self.env.plant_radius
            )

            p_arr[k + 1] = plant.p
            v_arr[k + 1] = plant.v
            p_ref_arr[k + 1], v_ref_arr[k + 1] = self.reference_state(t[k + 1], episode)
            V_arr[k + 1] = controller.lyapunov_value(
                plant.p, plant.v, p_ref_arr[k + 1], episode.obstacles, v_ref_arr[k + 1]
            )

        u_arr[-1] = controller.compute_force(
            plant.p, plant.v, p_ref_arr[-1], v_ref_arr[-1], episode.obstacles,
            self.reference_acceleration(t[-1], episode),
        )

        return {
            "t": t,
            "p": p_arr,
            "v": v_arr,
            "u": u_arr,
            "p_ref": p_ref_arr,
            "v_ref": v_ref_arr,
            "V": V_arr,
            "err": np.linalg.norm(p_arr - p_ref_arr, axis=1),
            "err_v": np.linalg.norm(v_arr - v_ref_arr, axis=1),
            "speed": np.linalg.norm(v_arr, axis=1),
            "accel": np.linalg.norm(u_arr, axis=1),
        }
