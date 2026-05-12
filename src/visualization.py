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
    "robust": "#E67700",
}

LABELS = {
    "adaptive": "adaptive",
    "fixed_lyapunov": "fixed Lyapunov",
    "plain_pd": "plain PD",
    "robust": "robust",
}


def _color(label: str) -> str:
    return COLORS.get(label, "#212529")


def _label(label: str) -> str:
    return LABELS.get(label, label.replace("_", " "))


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
    primary = rollouts["adaptive"]
    frames = np.arange(0, primary.time.size, frame_stride, dtype=int)
    if frames[-1] != primary.time.size - 1:
        frames = np.append(frames, primary.time.size - 1)

    fig, (ax_scene, ax_lyapunov) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    primary_label = _label(primary.label)
    fig.suptitle(f"3-DOF Manipulator: {primary_label} Control", fontsize=14)

    _setup_scene_axes(config, ax_scene)
    (line_primary,) = ax_scene.plot([], [], "-o", lw=3, color=_color(primary.label), label=primary_label)
    (trace_primary,) = ax_scene.plot([], [], lw=1.4, color=_color(primary.label), alpha=0.7)
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
        "adaptive": primary.tracking_lyapunov,
    }
    normalized_lyapunov = {
        key: _normalize_positive(series) for key, series in lyapunov_series.items()
    }
    for key, series in normalized_lyapunov.items():
        ax_lyapunov.plot(
            rollouts[key].time,
            series,
            color=_color(rollouts[key].label),
            lw=1.4,
            label=f"{_label(rollouts[key].label)} V_e / V_e(0)",
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
        points = arm.forward_kinematics(primary.q[index])
        line_primary.set_data(points[:, 0], points[:, 1])
        trace_primary.set_data(primary.end_effector[: index + 1, 0], primary.end_effector[: index + 1, 1])
        target_dot.set_data([primary.target[index, 0]], [primary.target[index, 1]])
        for patch, center in zip(obstacle_patches, primary.obstacle_centers[index], strict=True):
            patch.center = (center[0], center[1])
        time_marker.set_xdata([primary.time[index], primary.time[index]])
        text.set_text(
            f"t = {primary.time[index]:.1f} s\n"
            f"{primary_label} V_e = {primary.tracking_lyapunov[index]:.3f}\n"
            f"clearance = {primary.clearance[index]:.1f} px"
        )
        return (
            line_primary,
            trace_primary,
            target_dot,
            time_marker,
            text,
            *obstacle_patches,
        )

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def show_live_animation(
    *,
    config: ProjectConfig,
    arm: PlanarArm,
    rollouts: dict[str, Rollout],
    fps: int = 30,
) -> None:
    """Display real-time animation of the manipulator during simulation."""
    primary = rollouts["adaptive"]
    fig, (ax_scene, ax_lyapunov) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    primary_label = _label(primary.label)
    fig.suptitle(f"3-DOF Manipulator: {primary_label} Control (Live)", fontsize=14)

    _setup_scene_axes(config, ax_scene)
    (line_primary,) = ax_scene.plot([], [], "-o", lw=3, color=_color(primary.label), label=primary_label)
    (trace_primary,) = ax_scene.plot([], [], lw=1.4, color=_color(primary.label), alpha=0.7)
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
        "adaptive": primary.tracking_lyapunov,
    }
    normalized_lyapunov = {
        key: _normalize_positive(series) for key, series in lyapunov_series.items()
    }
    for key, series in normalized_lyapunov.items():
        ax_lyapunov.plot(
            rollouts[key].time,
            series,
            color=_color(rollouts[key].label),
            lw=1.4,
            label=f"{_label(rollouts[key].label)} V_e / V_e(0)",
        )
    time_marker = ax_lyapunov.axvline(0.0, color="black", lw=1.0, alpha=0.7)
    ax_lyapunov.set_title("Comparable Tracking Lyapunov Values")
    ax_lyapunov.set_xlabel("time [s]")
    ax_lyapunov.set_ylabel("normalized value")
    ax_lyapunov.set_ylim(0.0, 1.08 * max(float(np.max(v)) for v in normalized_lyapunov.values()))
    ax_lyapunov.grid(alpha=0.28)
    ax_lyapunov.legend(fontsize=8)

    def update(frame_number: int):
        index = frame_number
        points = arm.forward_kinematics(primary.q[index])
        line_primary.set_data(points[:, 0], points[:, 1])
        trace_primary.set_data(primary.end_effector[: index + 1, 0], primary.end_effector[: index + 1, 1])
        target_dot.set_data([primary.target[index, 0]], [primary.target[index, 1]])
        for patch, center in zip(obstacle_patches, primary.obstacle_centers[index], strict=True):
            patch.center = (center[0], center[1])
        time_marker.set_xdata([primary.time[index], primary.time[index]])
        text.set_text(
            f"t = {primary.time[index]:.1f} s\n"
            f"{primary_label} V_e = {primary.tracking_lyapunov[index]:.3f}\n"
            f"clearance = {primary.clearance[index]:.1f} px"
        )
        return (
            line_primary,
            trace_primary,
            target_dot,
            time_marker,
            text,
            *obstacle_patches,
        )

    ani = animation.FuncAnimation(fig, update, frames=primary.time.size, interval=1000 / fps, blit=False)
    plt.show()
    plt.close(fig)


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
        color=_color(reference.label),
        lw=2,
        label=f"{_label(reference.label)} ee path",
    )
    for center in reference.obstacle_centers[0]:
        ax.add_patch(Circle(center, config.obstacles.radius, fc="#ADB5BD", ec="#495057", alpha=0.35))
    for rollout in rollouts.values():
        points = arm.forward_kinematics(rollout.q[-1])
        ax.plot(points[:, 0], points[:, 1], "-o", color=_color(rollout.label), lw=1.5, alpha=0.9)
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
    for rollout in rollouts.values():
        axes[0].plot(rollout.time, rollout.target_error, color=_color(rollout.label), label=_label(rollout.label))
        axes[1].plot(rollout.time, rollout.q_error_norm, color=_color(rollout.label), label=_label(rollout.label))
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
    primary = rollouts["adaptive"]
    primary_label = _label(primary.label)
    augmented = primary.augmented_lyapunov
    positive_augmented = np.maximum(augmented, 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].semilogy(
        primary.time,
        positive_augmented,
        color=_color(primary.label),
        lw=2.0,
        label=f"{primary_label} augmented V",
    )
    axes[0].scatter(
        [primary.time[0], primary.time[-1]],
        [positive_augmented[0], positive_augmented[-1]],
        color=_color(primary.label),
        s=35,
        zorder=4,
    )
    axes[0].set_title(
        f"{primary_label.title()} Lyapunov Candidate: V(0)={augmented[0]:.2f}, V(T)={augmented[-1]:.2f}"
    )
    axes[0].set_ylabel("augmented V (log scale)")
    axes[0].grid(alpha=0.28, which="both")
    axes[0].legend()

    for rollout in rollouts.values():
        axes[1].plot(
            rollout.time,
            rollout.tracking_lyapunov,
            color=_color(rollout.label),
            lw=1.5,
            label=_label(rollout.label),
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
    controller_label = _label(adaptive.label).title()
    axes[0].set_title(f"{controller_label} Inertia Estimates")
    axes[0].set_ylabel("inertia")
    axes[1].set_title(f"{controller_label} Damping Estimates")
    axes[1].set_ylabel("damping")
    axes[2].set_title(f"{controller_label} Constant Bias Estimates")
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
    primary = rollouts["adaptive"]
    for joint in range(config.robot.link_lengths.size):
        axes[0].plot(primary.time, primary.tau[:, joint], label=f"tau_{joint + 1}")
        axes[0].axhline(config.dynamics.torque_limits[joint], color="black", ls="--", lw=0.7, alpha=0.45)
        axes[0].axhline(-config.dynamics.torque_limits[joint], color="black", ls="--", lw=0.7, alpha=0.45)
    for rollout in rollouts.values():
        axes[1].plot(rollout.time, rollout.clearance, color=_color(rollout.label), label=_label(rollout.label))
    axes[1].axhline(0.0, color="black", ls="--", lw=1.0, label="collision boundary")
    axes[0].set_title(f"{_label(primary.label).title()} Control Torques")
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
