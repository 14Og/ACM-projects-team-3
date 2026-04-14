"""Combined trajectory + metrics animation in a single window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from .config import EnvConfig, EpisodeState


class Visualizer:
    """Visualization service for trajectory plots and animations."""

    @staticmethod
    def _finite_values(values: np.ndarray) -> np.ndarray:
        finite = np.asarray(values, dtype=float)
        return finite[np.isfinite(finite)]

    @staticmethod
    def _display_cap(values: np.ndarray, percentile: float = 95.0, floor: float = 1e-3) -> float:
        finite = Visualizer._finite_values(values)
        if finite.size == 0:
            return floor
        cap = float(np.percentile(finite, percentile))
        if not np.isfinite(cap) or cap <= 0.0:
            cap = float(np.max(finite))
        return max(floor, cap * 1.25)

    @staticmethod
    def _display_series(values: np.ndarray, cap: float) -> np.ndarray:
        series = np.asarray(values, dtype=float)
        return np.clip(np.nan_to_num(series, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap)

    @staticmethod
    def _scene_bounds(env: EnvConfig, episode: EpisodeState) -> tuple[float, float, float, float]:
        theta_grid = np.linspace(0.0, 2.0 * np.pi, 500)
        ellipse_x = env.cx + env.a * np.cos(theta_grid)
        ellipse_y = env.cy + env.b * np.sin(theta_grid)

        obstacle_x = np.array([ox for ox, _, _ in episode.obstacles], dtype=float)
        obstacle_y = np.array([oy for _, oy, _ in episode.obstacles], dtype=float)
        obstacle_r = np.array([r for _, _, r in episode.obstacles], dtype=float)

        x_min = float(np.min(ellipse_x))
        x_max = float(np.max(ellipse_x))
        y_min = float(np.min(ellipse_y))
        y_max = float(np.max(ellipse_y))

        if obstacle_x.size:
            x_min = min(x_min, float(np.min(obstacle_x - obstacle_r)))
            x_max = max(x_max, float(np.max(obstacle_x + obstacle_r)))
            y_min = min(y_min, float(np.min(obstacle_y - obstacle_r)))
            y_max = max(y_max, float(np.max(obstacle_y + obstacle_r)))

        margin = max(env.plant_radius, env.target_radius, env.obstacle_radius) + 6.0
        return x_min - margin, x_max + margin, y_min - margin, y_max + margin

    @staticmethod
    def _clip_points(points: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        clipped = np.asarray(points, dtype=float).copy()
        clipped[:, 0] = np.clip(np.nan_to_num(clipped[:, 0], nan=x_min, posinf=x_max, neginf=x_min), x_min, x_max)
        clipped[:, 1] = np.clip(np.nan_to_num(clipped[:, 1], nan=y_min, posinf=y_max, neginf=y_min), y_min, y_max)
        return clipped

    def make_animation(
        self,
        data: dict[str, Any],
        env: EnvConfig,
        episode: EpisodeState,
        ep_num: int = 1,
        frame_step: int = 3,
        fig: plt.Figure | None = None,
    ) -> tuple[plt.Figure, animation.FuncAnimation]:
        """Single window: trajectory (left) + 4 metrics panels (right)."""

        p = data["p"]
        v = data["v"]
        p_ref = data["p_ref"]
        speed = data["speed"]
        n_steps = len(p)
        steps = np.arange(n_steps)

        raw_series = [data["speed"], data["accel"], data["err"], data["err_v"], data["V"]]
        y_labels = ["Speed |v|", "Accel |u|", "Error ||e_p||", "Error ||e_v||", r"$V_\mathrm{total}$"]
        m_colors = ["#1f88d8", "#d64f4f", "#2f9d67", "#e39b2f", "#9b59b6"]
        y_caps = [self._display_cap(series) for series in raw_series]
        y_series = [self._display_series(series, cap) for series, cap in zip(raw_series, y_caps)]
        x_min, x_max, y_min, y_max = self._scene_bounds(env, episode)
        p_display = self._clip_points(p, x_min, x_max, y_min, y_max)
        p_ref_display = self._clip_points(p_ref, x_min, x_max, y_min, y_max)

        if fig is None:
            fig = plt.figure(figsize=(17, 9))
        else:
            fig.clear()
        fig.canvas.manager.set_window_title(f"Episode {ep_num}")

        gs = gridspec.GridSpec(
            5, 2,
            width_ratios=[1.5, 1],
            hspace=0.5,
            wspace=0.3,
            left=0.05, right=0.97,
            top=0.93, bottom=0.07,
        )
        ax_traj = fig.add_subplot(gs[:, 0])
        m_axes = [fig.add_subplot(gs[i, 1]) for i in range(5)]

        fig.suptitle(f"Episode {ep_num} — Lyapunov APF Tracking", fontsize=17)

        theta_grid = np.linspace(0.0, 2.0 * np.pi, 500)
        ex = env.cx + env.a * np.cos(theta_grid)
        ey = env.cy + env.b * np.sin(theta_grid)
        ax_traj.plot(ex, ey, linestyle=(0, (1.5, 3.0)), linewidth=1.6,
                     color="#d9b454", alpha=0.55, label="Reference ellipse")

        for ox, oy, r in episode.obstacles:
            ax_traj.add_patch(plt.Circle((ox, oy), r,
                                         facecolor="#f05b6a", edgecolor="#7a2630", alpha=0.7))
            ax_traj.plot(ox, oy, marker="P", markersize=10, color="#4b1ea6")

        (line_plant,) = ax_traj.plot([], [], color="#2ca9ff", linewidth=2, label="Plant")

        target_patch = Circle(
            (p_ref[0, 0], p_ref[0, 1]), radius=env.target_radius,
            facecolor="#f4c757", edgecolor="#a88523", linewidth=1.2, zorder=5,
        )
        plant_patch = Circle(
            (p_display[0, 0], p_display[0, 1]), radius=env.plant_radius,
            facecolor="#1f88d8", edgecolor="#0f4f80", linewidth=1.2, zorder=6, alpha=0.5,
        )
        ax_traj.add_patch(target_patch)
        ax_traj.add_patch(plant_patch)

        speed_arrow = FancyArrowPatch(
            (0.0, 0.0), (0.0, 0.0),
            arrowstyle="-|>", mutation_scale=15, linewidth=2.2, color="black",
        )
        ax_traj.add_patch(speed_arrow)
        step_text = ax_traj.text(0.02, 0.97, "", transform=ax_traj.transAxes,
                                 fontsize=14, va="top")

        ax_traj.set_xlim(x_min, x_max)
        ax_traj.set_ylim(y_min, y_max)
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.grid(alpha=0.18, linewidth=0.7)
        ax_traj.set_xlabel("X", fontsize=13)
        ax_traj.set_ylabel("Y", fontsize=13)
        ax_traj.tick_params(labelsize=11)
        ax_traj.legend(loc="upper right", frameon=True, fontsize=11)

        m_lines, m_dots = [], []
        for ax, y, c, label in zip(m_axes, y_series, m_colors, y_labels):
            (line,) = ax.plot([], [], color=c, linewidth=2.0)
            (dot,) = ax.plot([], [], marker="o", markersize=5, color=c)
            ax.set_ylabel(label, fontsize=11)
            ax.set_ylim(0.0, max(1e-3, float(np.max(y))) * 1.12)
            ax.set_xlim(0, n_steps - 1)
            ax.grid(alpha=0.24, linewidth=0.7)
            ax.tick_params(labelsize=10)
            m_lines.append(line)
            m_dots.append(dot)

        m_axes[-1].set_xlabel("Step", fontsize=11)

        frames = np.arange(0, n_steps, frame_step, dtype=int)
        if frames[-1] != n_steps - 1:
            frames = np.append(frames, n_steps - 1)

        last_dir = {"vec": np.array([1.0, 0.0], dtype=float)}

        def update(idx: int):
            k = int(frames[idx])

            line_plant.set_data(p_display[: k + 1, 0], p_display[: k + 1, 1])
            target_patch.center = (p_ref_display[k, 0], p_ref_display[k, 1])
            plant_patch.center = (p_display[k, 0], p_display[k, 1])

            v_vec = v[k]
            v_norm = float(np.linalg.norm(v_vec))
            if v_norm > 1e-6:
                last_dir["vec"] = v_vec / v_norm
            v_disp = min(v_norm, y_caps[0])
            arrow_len = float(np.clip(1.1 + 0.95 * np.log1p(v_disp), 1.1, 16.0))
            p0 = p_display[k]
            p1 = p0 + arrow_len * last_dir["vec"]
            speed_arrow.set_positions((p0[0], p0[1]), (p1[0], p1[1]))
            speed_arrow.set_mutation_scale(13.0 + 2.0 * np.log1p(v_disp))
            step_text.set_text(f"step {k}/{n_steps}   |v| = {speed[k]:.2f}")

            sk = steps[: k + 1]
            for line, dot, y in zip(m_lines, m_dots, y_series):
                line.set_data(sk, y[: k + 1])
                dot.set_data([k], [y[k]])

            return (line_plant, target_patch, plant_patch, speed_arrow, step_text, *m_lines, *m_dots)

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=30, blit=True, repeat=True,
        )
        return fig, ani

    def save_animation(self, ani: animation.FuncAnimation, path: Path, fps: int = 30) -> None:
        ext = path.suffix.lower()
        if ext not in {".gif", ".mp4"}:
            raise ValueError("Animation path must end with .gif or .mp4")
        writer = animation.PillowWriter(fps=fps) if ext == ".gif" else animation.FFMpegWriter(fps=fps)
        ani.save(path, writer=writer)
