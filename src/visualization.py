"""Plotting and animation for the adaptive manipulator project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .config import ProjectConfig
from .simulation import Rollout
from .system import PlanarArm


COLORS = {
    "adaptive": "#087F5B",
    "fixed_lyapunov": "#C92A2A",
    "plain_pd": "#5F3DC4",
    "reference": "#1C7ED6",
    "target": "#F08C00",
}

LABELS = {
    "adaptive": "adaptive",
    "fixed_lyapunov": "fixed Lyapunov",
    "plain_pd": "plain PD",
}


def save_all_plots(
    *,
    config: ProjectConfig,
    arm: PlanarArm,
    rollouts: dict[str, Rollout],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        _save_workspace(config, arm, rollouts, figures_dir),
        _save_errors(config, rollouts, figures_dir),
        _save_lyapunov_values(rollouts, figures_dir),
        _save_parameters(config, rollouts["adaptive"], figures_dir),
        _save_control_clearance(config, rollouts, figures_dir),
    ]


def save_animation(
    *,
    config: ProjectConfig,
    arm: PlanarArm,
    rollouts: dict[str, Rollout],
    animations_dir: Path,
    frame_stride: int = 15,
    fps: int = 30,
) -> Path:
    animations_dir.mkdir(parents=True, exist_ok=True)
    path = animations_dir / "adaptive_manipulator.gif"
    adaptive = rollouts["adaptive"]
    fixed = rollouts["fixed_lyapunov"]
    pd = rollouts["plain_pd"]
    frames = np.arange(0, adaptive.time.size, frame_stride, dtype=int)
    if frames[-1] != adaptive.time.size - 1:
        frames = np.append(frames, adaptive.time.size - 1)

    fig, (ax_scene, ax_lyapunov) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.suptitle("3-DOF Manipulator: Adaptive Control vs Lyapunov Baselines", fontsize=14)

    _setup_scene_axes(config, ax_scene)
    (line_adaptive,) = ax_scene.plot([], [], "-o", lw=3, color=COLORS["adaptive"], label="adaptive")
    (line_fixed,) = ax_scene.plot([], [], "-o", lw=2, color=COLORS["fixed_lyapunov"], label="fixed Lyapunov")
    (line_pd,) = ax_scene.plot([], [], "-o", lw=2, color=COLORS["plain_pd"], label="plain PD")
    (trace_adaptive,) = ax_scene.plot([], [], lw=1.4, color=COLORS["adaptive"], alpha=0.7)
    (trace_fixed,) = ax_scene.plot([], [], lw=1.1, color=COLORS["fixed_lyapunov"], alpha=0.55)
    (trace_pd,) = ax_scene.plot([], [], lw=1.1, color=COLORS["plain_pd"], alpha=0.55)
    (target_dot,) = ax_scene.plot([], [], "o", ms=8, color=COLORS["target"], label="moving target")
    obstacle_patches = [
        Circle((0.0, 0.0), config.obstacles.radius, fc="#868E96", ec="#343A40", alpha=0.4)
        for _ in range(config.obstacles.base_centers.shape[0])
    ]
    for patch in obstacle_patches:
        ax_scene.add_patch(patch)
    text = ax_scene.text(0.02, 0.98, "", transform=ax_scene.transAxes, va="top")
    ax_scene.legend(loc="lower left", fontsize=9)

    lyapunov_series = {
        "adaptive": adaptive.tracking_lyapunov,
        "fixed_lyapunov": fixed.tracking_lyapunov,
        "plain_pd": pd.tracking_lyapunov,
    }
    normalized_lyapunov = {
        key: _normalize_positive(series) for key, series in lyapunov_series.items()
    }
    for key, series in normalized_lyapunov.items():
        ax_lyapunov.plot(
            rollouts[key].time,
            series,
            color=COLORS[key],
            lw=1.4,
            label=f"{LABELS[key]} V_e / V_e(0)",
        )
    time_marker = ax_lyapunov.axvline(0.0, color="black", lw=1.0, alpha=0.7)
    ax_lyapunov.set_title("Comparable Tracking Lyapunov Values")
    ax_lyapunov.set_xlabel("time [s]")
    ax_lyapunov.set_ylabel("normalized value")
    ax_lyapunov.set_ylim(0.0, 1.08 * max(float(np.max(v)) for v in normalized_lyapunov.values()))
    ax_lyapunov.grid(alpha=0.28)
    ax_lyapunov.legend(fontsize=8)

    def update(frame_number: int):
        index = int(frames[frame_number])
        for line, rollout in [
            (line_adaptive, adaptive),
            (line_fixed, fixed),
            (line_pd, pd),
        ]:
            points = arm.forward_kinematics(rollout.q[index])
            line.set_data(points[:, 0], points[:, 1])
        trace_adaptive.set_data(adaptive.end_effector[: index + 1, 0], adaptive.end_effector[: index + 1, 1])
        trace_fixed.set_data(fixed.end_effector[: index + 1, 0], fixed.end_effector[: index + 1, 1])
        trace_pd.set_data(pd.end_effector[: index + 1, 0], pd.end_effector[: index + 1, 1])
        target_dot.set_data([adaptive.target[index, 0]], [adaptive.target[index, 1]])
        for patch, center in zip(obstacle_patches, adaptive.obstacle_centers[index], strict=True):
            patch.center = (center[0], center[1])
        time_marker.set_xdata([adaptive.time[index], adaptive.time[index]])
        text.set_text(
            f"t = {adaptive.time[index]:.1f} s\n"
            f"adaptive V_e = {adaptive.tracking_lyapunov[index]:.3f}\n"
            f"fixed V_e = {fixed.tracking_lyapunov[index]:.3f}\n"
            f"PD V_e = {pd.tracking_lyapunov[index]:.3f}\n"
            f"clearance = {adaptive.clearance[index]:.1f} px"
        )
        return (
            line_adaptive,
            line_fixed,
            line_pd,
            trace_adaptive,
            trace_fixed,
            trace_pd,
            target_dot,
            time_marker,
            text,
            *obstacle_patches,
        )

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def _save_workspace(
    config: ProjectConfig,
    arm: PlanarArm,
    rollouts: dict[str, Rollout],
    figures_dir: Path,
) -> Path:
    path = figures_dir / "workspace_trajectories.png"
    fig, ax = plt.subplots(figsize=(9, 8))
    _setup_scene_axes(config, ax)
    reference = rollouts["adaptive"]
    ax.plot(reference.target[:, 0], reference.target[:, 1], color=COLORS["target"], lw=2, label="target path")
    ax.plot(
        reference.end_effector[:, 0],
        reference.end_effector[:, 1],
        color=COLORS["adaptive"],
        lw=2,
        label="adaptive ee path",
    )
    for key in ["fixed_lyapunov", "plain_pd"]:
        ax.plot(
            rollouts[key].end_effector[:, 0],
            rollouts[key].end_effector[:, 1],
            color=COLORS[key],
            lw=1.5,
            label=f"{LABELS[key]} ee path",
        )
    for center in reference.obstacle_centers[0]:
        ax.add_patch(Circle(center, config.obstacles.radius, fc="#ADB5BD", ec="#495057", alpha=0.35))
    for rollout in rollouts.values():
        points = arm.forward_kinematics(rollout.q[-1])
        ax.plot(points[:, 0], points[:, 1], "-o", color=COLORS[rollout.label], lw=1.5, alpha=0.9)
    ax.set_title("Workspace Trajectories and Final Arm Poses")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    initial = finite[0]
    if abs(initial) <= 1e-12:
        initial = max(float(np.max(np.abs(finite))), 1.0)
    return np.nan_to_num(values / initial, nan=0.0, posinf=0.0, neginf=0.0)


def _save_errors(config: ProjectConfig, rollouts: dict[str, Rollout], figures_dir: Path) -> Path:
    path = figures_dir / "tracking_errors.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, rollout in rollouts.items():
        axes[0].plot(rollout.time, rollout.target_error, color=COLORS[key], label=LABELS[key])
        axes[1].plot(rollout.time, rollout.q_error_norm, color=COLORS[key], label=LABELS[key])
    axes[0].axhline(
        config.target.threshold,
        color="black",
        lw=1.0,
        ls="--",
        label="target threshold",
    )
    axes[0].set_title("End-Effector Target Error")
    axes[0].set_ylabel("error [px]")
    axes[1].set_title("Joint Desired-Trajectory Tracking Error")
    axes[1].set_ylabel("||q - q_d|| [rad]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.28)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_lyapunov_values(rollouts: dict[str, Rollout], figures_dir: Path) -> Path:
    path = figures_dir / "lyapunov_values.png"
    adaptive = rollouts["adaptive"]
    augmented = adaptive.augmented_lyapunov
    positive_augmented = np.maximum(augmented, 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].semilogy(
        adaptive.time,
        positive_augmented,
        color=COLORS["adaptive"],
        lw=2.0,
        label="adaptive augmented V",
    )
    axes[0].scatter(
        [adaptive.time[0], adaptive.time[-1]],
        [positive_augmented[0], positive_augmented[-1]],
        color=COLORS["adaptive"],
        s=35,
        zorder=4,
    )
    axes[0].set_title(
        f"Adaptive Lyapunov Decrease: V(0)={augmented[0]:.2f}, V(T)={augmented[-1]:.2f}"
    )
    axes[0].set_ylabel("augmented V (log scale)")
    axes[0].grid(alpha=0.28, which="both")
    axes[0].legend()

    for key, rollout in rollouts.items():
        axes[1].plot(
            rollout.time,
            rollout.tracking_lyapunov,
            color=COLORS[key],
            lw=1.5,
            label=LABELS[key],
        )
    axes[1].set_title("Tracking Lyapunov Candidate Across Controllers")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("V_e = 0.5(||e||^2 + ||de||^2)")
    axes[1].grid(alpha=0.28)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_parameters(config: ProjectConfig, adaptive: Rollout, figures_dir: Path) -> Path:
    path = figures_dir / "adaptive_parameter_estimates.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for joint in range(config.robot.link_lengths.size):
        axes[0].plot(adaptive.time, adaptive.inertia_hat[:, joint], label=f"I_hat_{joint + 1}")
        axes[0].axhline(config.dynamics.true_inertia[joint], lw=1.0, ls="--", alpha=0.7)
        axes[1].plot(adaptive.time, adaptive.damping_hat[:, joint], label=f"D_hat_{joint + 1}")
        axes[1].axhline(config.dynamics.true_damping[joint], lw=1.0, ls="--", alpha=0.7)
        axes[2].plot(adaptive.time, adaptive.bias_hat[:, joint], label=f"b_hat_{joint + 1}")
        axes[2].axhline(config.dynamics.disturbance_constant[joint], lw=1.0, ls="--", alpha=0.7)
    axes[0].set_title("Adaptive Inertia Estimates")
    axes[0].set_ylabel("inertia")
    axes[1].set_title("Adaptive Damping Estimates")
    axes[1].set_ylabel("damping")
    axes[2].set_title("Adaptive Constant Bias Estimates")
    axes[2].set_ylabel("bias torque")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.28)
        ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_control_clearance(
    config: ProjectConfig,
    rollouts: dict[str, Rollout],
    figures_dir: Path,
) -> Path:
    path = figures_dir / "control_and_clearance.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    adaptive = rollouts["adaptive"]
    for joint in range(config.robot.link_lengths.size):
        axes[0].plot(adaptive.time, adaptive.tau[:, joint], label=f"tau_{joint + 1}")
        axes[0].axhline(config.dynamics.torque_limits[joint], color="black", ls="--", lw=0.7, alpha=0.45)
        axes[0].axhline(-config.dynamics.torque_limits[joint], color="black", ls="--", lw=0.7, alpha=0.45)
    for key, rollout in rollouts.items():
        axes[1].plot(rollout.time, rollout.clearance, color=COLORS[key], label=LABELS[key])
    axes[1].axhline(0.0, color="black", ls="--", lw=1.0, label="collision boundary")
    axes[0].set_title("Adaptive Control Torques")
    axes[0].set_ylabel("torque")
    axes[1].set_title("Minimum Link-Obstacle Clearance")
    axes[1].set_ylabel("clearance [px]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.28)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _setup_scene_axes(config: ProjectConfig, ax: plt.Axes) -> None:
    base = config.robot.base_xy
    reach = float(np.sum(config.robot.link_lengths))
    ax.set_xlim(base[0] - reach - 35.0, base[0] + reach + 35.0)
    ax.set_ylim(base[1] - reach - 60.0, base[1] + reach + 180.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.grid(alpha=0.24)
