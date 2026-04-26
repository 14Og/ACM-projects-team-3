"""Plotting and animation for the robust manipulator project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .config import ProjectConfig
from .simulation import Rollout
from .system import PlanarArm


COLORS = {
    "robust": "#E67700",
    "reference": "#1C7ED6",
    "planner": "#364FC7",
    "target": "#F08C00",
}

LABELS = {
    "robust": "robust",
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
        save_reference_plot(config=config, rollout=rollouts["robust"], figures_dir=figures_dir),
        _save_parameters(config, rollouts["robust"], figures_dir),
        _save_control(config, rollouts, figures_dir),
    ]


def save_reference_plot(
    *,
    config: ProjectConfig,
    rollout: Rollout,
    figures_dir: Path,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return _save_reference_model(config, rollout, figures_dir)


def save_animation(
    *,
    config: ProjectConfig,
    arm: PlanarArm,
    rollouts: dict[str, Rollout],
    animations_dir: Path,
    frame_stride: int = 15,
    fps: int = 30,
    reference_only: bool = False,
) -> Path:
    animations_dir.mkdir(parents=True, exist_ok=True)
    path = animations_dir / "robust_manipulator.gif"
    rollout = rollouts["robust"]
    frames = np.arange(0, rollout.time.size, frame_stride, dtype=int)
    if frames[-1] != rollout.time.size - 1:
        frames = np.append(frames, rollout.time.size - 1)

    fig, (ax_scene, ax_lyapunov) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    title = "3-DOF Manipulator Reference Model" if reference_only else "3-DOF Manipulator with Robust Control and Reference Model"
    fig.suptitle(title, fontsize=14)

    _setup_scene_axes(config, ax_scene)
    scene_color = COLORS["reference"] if reference_only else COLORS["robust"]
    scene_label = "reference model" if reference_only else "robust"
    (line_robot,) = ax_scene.plot([], [], "-o", lw=3, color=scene_color, label=scene_label)
    (trace_robot,) = ax_scene.plot([], [], lw=1.4, color=scene_color, alpha=0.7)
    line_reference = None
    trace_reference = None
    if not reference_only:
        (line_reference,) = ax_scene.plot([], [], "-o", lw=2.2, color=COLORS["reference"], label="reference model")
        (trace_reference,) = ax_scene.plot([], [], lw=1.2, color=COLORS["reference"], alpha=0.65)
    (target_dot,) = ax_scene.plot([], [], "o", ms=8, color=COLORS["target"], label="moving target")
    text = ax_scene.text(0.02, 0.98, "", transform=ax_scene.transAxes, va="top")
    ax_scene.legend(loc="lower left", fontsize=9)

    for joint in range(rollout.q.shape[1]):
        ax_lyapunov.plot(
            rollout.time,
            rollout.q_des[:, joint],
            color=COLORS["planner"],
            lw=1.0,
            ls="--",
            alpha=0.55,
        )
        ax_lyapunov.plot(
            rollout.time,
            rollout.q_model[:, joint],
            color=COLORS["reference"],
            lw=1.4,
            label=f"q_m{joint + 1}" if joint == 0 else None,
        )
        ax_lyapunov.plot(
            rollout.time,
            rollout.q[:, joint],
            color=COLORS["robust"],
            lw=1.2,
            alpha=0.75,
            label=f"q{joint + 1}" if joint == 0 else None,
        )
    time_marker = ax_lyapunov.axvline(0.0, color="black", lw=1.0, alpha=0.7)
    ax_lyapunov.plot([], [], color=COLORS["planner"], lw=1.0, ls="--", label="planner q_d")
    ax_lyapunov.set_title("Joint Angles and Reference Model")
    ax_lyapunov.set_xlabel("time [s]")
    ax_lyapunov.set_ylabel("angle [rad]")
    ax_lyapunov.grid(alpha=0.28)
    ax_lyapunov.legend(fontsize=8)

    def update(frame_number: int):
        index = int(frames[frame_number])
        q_scene = rollout.q_model[index] if reference_only else rollout.q[index]
        ee_trace = np.array(
            [arm.end_effector(q_state) for q_state in (rollout.q_model[: index + 1] if reference_only else rollout.q[: index + 1])]
        )
        points = arm.forward_kinematics(q_scene)
        line_robot.set_data(points[:, 0], points[:, 1])
        trace_robot.set_data(ee_trace[:, 0], ee_trace[:, 1])
        if not reference_only and line_reference is not None and trace_reference is not None:
            reference_points = arm.forward_kinematics(rollout.q_model[index])
            reference_trace = np.array([arm.end_effector(q_state) for q_state in rollout.q_model[: index + 1]])
            line_reference.set_data(reference_points[:, 0], reference_points[:, 1])
            trace_reference.set_data(reference_trace[:, 0], reference_trace[:, 1])
        target_dot.set_data([rollout.target[index, 0]], [rollout.target[index, 1]])
        time_marker.set_xdata([rollout.time[index], rollout.time[index]])
        text.set_text(
            f"t = {rollout.time[index]:.1f} s\n"
            f"||q-q_m|| = {np.linalg.norm(rollout.q[index] - rollout.q_model[index]):.3f}\n"
            f"||q_m-q_d|| = {np.linalg.norm(rollout.q_model[index] - rollout.q_des[index]):.3f}"
        )
        artists = (
            line_robot,
            trace_robot,
            target_dot,
            time_marker,
            text,
        )
        if not reference_only and line_reference is not None and trace_reference is not None:
            artists += (
                line_reference,
                trace_reference,
            )
        return artists
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
    frame_stride: int = 1,
    reference_only: bool = False,
) -> None:
    """Display real-time animation of the manipulator during simulation."""
    import matplotlib
    matplotlib.use("TkAgg")  # Use interactive backend

    rollout = rollouts["robust"]
    frames = np.arange(0, rollout.time.size, max(1, int(frame_stride)), dtype=int)
    if frames[-1] != rollout.time.size - 1:
        frames = np.append(frames, rollout.time.size - 1)

    fig, (ax_scene, ax_lyapunov) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    title = (
        "3-DOF Manipulator Reference Model (Live)"
        if reference_only
        else "3-DOF Manipulator with Robust Control and Reference Model (Live)"
    )
    fig.suptitle(title, fontsize=14)

    _setup_scene_axes(config, ax_scene)
    scene_color = COLORS["reference"] if reference_only else COLORS["robust"]
    scene_label = "reference model" if reference_only else "robust"
    (line_robot,) = ax_scene.plot([], [], "-o", lw=3, color=scene_color, label=scene_label)
    (trace_robot,) = ax_scene.plot([], [], lw=1.4, color=scene_color, alpha=0.7)
    line_reference = None
    trace_reference = None
    if not reference_only:
        (line_reference,) = ax_scene.plot([], [], "-o", lw=2.2, color=COLORS["reference"], label="reference model")
        (trace_reference,) = ax_scene.plot([], [], lw=1.2, color=COLORS["reference"], alpha=0.65)
    (target_dot,) = ax_scene.plot([], [], "o", ms=8, color=COLORS["target"], label="moving target")
    text = ax_scene.text(0.02, 0.98, "", transform=ax_scene.transAxes, va="top")
    ax_scene.legend(loc="lower left", fontsize=9)

    for joint in range(rollout.q.shape[1]):
        ax_lyapunov.plot(
            rollout.time,
            rollout.q_des[:, joint],
            color=COLORS["planner"],
            lw=1.0,
            ls="--",
            alpha=0.55,
        )
        ax_lyapunov.plot(
            rollout.time,
            rollout.q_model[:, joint],
            color=COLORS["reference"],
            lw=1.4,
            label=f"q_m{joint + 1}" if joint == 0 else None,
        )
        ax_lyapunov.plot(
            rollout.time,
            rollout.q[:, joint],
            color=COLORS["robust"],
            lw=1.2,
            alpha=0.75,
            label=f"q{joint + 1}" if joint == 0 else None,
        )
    time_marker = ax_lyapunov.axvline(0.0, color="black", lw=1.0, alpha=0.7)
    ax_lyapunov.plot([], [], color=COLORS["planner"], lw=1.0, ls="--", label="planner q_d")
    ax_lyapunov.set_title("Joint Angles and Reference Model")
    ax_lyapunov.set_xlabel("time [s]")
    ax_lyapunov.set_ylabel("angle [rad]")
    ax_lyapunov.grid(alpha=0.28)
    ax_lyapunov.legend(fontsize=8)

    def update(frame_number: int):
        index = int(frames[frame_number])
        q_scene = rollout.q_model[index] if reference_only else rollout.q[index]
        ee_trace = np.array(
            [arm.end_effector(q_state) for q_state in (rollout.q_model[: index + 1] if reference_only else rollout.q[: index + 1])]
        )
        points = arm.forward_kinematics(q_scene)
        line_robot.set_data(points[:, 0], points[:, 1])
        trace_robot.set_data(ee_trace[:, 0], ee_trace[:, 1])
        if not reference_only and line_reference is not None and trace_reference is not None:
            reference_points = arm.forward_kinematics(rollout.q_model[index])
            reference_trace = np.array([arm.end_effector(q_state) for q_state in rollout.q_model[: index + 1]])
            line_reference.set_data(reference_points[:, 0], reference_points[:, 1])
            trace_reference.set_data(reference_trace[:, 0], reference_trace[:, 1])
        target_dot.set_data([rollout.target[index, 0]], [rollout.target[index, 1]])
        time_marker.set_xdata([rollout.time[index], rollout.time[index]])
        text.set_text(
            f"t = {rollout.time[index]:.1f} s\n"
            f"||q-q_m|| = {np.linalg.norm(rollout.q[index] - rollout.q_model[index]):.3f}\n"
            f"||q_m-q_d|| = {np.linalg.norm(rollout.q_model[index] - rollout.q_des[index]):.3f}"
        )
        return (
            line_robot,
            trace_robot,
            target_dot,
            time_marker,
            text,
        )

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
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
    reference = rollouts["robust"]
    ax.plot(reference.target[:, 0], reference.target[:, 1], color=COLORS["target"], lw=2, label="target path")
    ax.plot(
        reference.end_effector[:, 0],
        reference.end_effector[:, 1],
        color=COLORS["robust"],
        lw=2,
        label="robust ee path",
    )
    points = arm.forward_kinematics(reference.q[-1])
    ax.plot(points[:, 0], points[:, 1], "-o", color=COLORS[reference.label], lw=1.5, alpha=0.9)
    ax.set_title("Workspace Trajectories and Final Arm Poses")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_errors(config: ProjectConfig, rollouts: dict[str, Rollout], figures_dir: Path) -> Path:
    path = figures_dir / "tracking_errors.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    rollout = rollouts["robust"]
    axes[0].plot(rollout.time, rollout.target_error, color=COLORS["robust"], label="end-effector error")
    axes[1].plot(rollout.time, rollout.q_error_norm, color=COLORS["robust"], label="||q - q_m||")
    axes[0].axhline(
        config.target.threshold,
        color="black",
        lw=1.0,
        ls="--",
        label="target threshold",
    )
    axes[0].set_title("End-Effector Target Error")
    axes[0].set_ylabel("error [px]")
    axes[1].set_title("Joint Reference-Model Tracking Error")
    axes[1].set_ylabel("||q - q_m|| [rad]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.28)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_reference_model(config: ProjectConfig, rollout: Rollout, figures_dir: Path) -> Path:
    path = figures_dir / "reference_model_tracking.png"
    fig, axes = plt.subplots(config.robot.link_lengths.size, 1, figsize=(10, 9), sharex=True)
    axes = np.atleast_1d(axes)
    for joint, ax in enumerate(axes):
        ax.plot(rollout.time, rollout.q[:, joint], color=COLORS["robust"], lw=1.5, label=f"q{joint + 1}")
        ax.plot(rollout.time, rollout.q_model[:, joint], color=COLORS["reference"], lw=1.5, label=f"q_m{joint + 1}")
        ax.plot(
            rollout.time,
            rollout.q_des[:, joint],
            color=COLORS["planner"],
            lw=1.0,
            ls="--",
            label=f"q_d{joint + 1}",
        )
        ax.set_ylabel("rad")
        ax.set_title(f"Joint {joint + 1}: actual, reference-model, planner")
        ax.grid(alpha=0.28)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_parameters(config: ProjectConfig, rollout: Rollout, figures_dir: Path) -> Path:
    path = figures_dir / "robust_parameter_estimates.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for joint in range(config.robot.link_lengths.size):
        axes[0].plot(rollout.time, rollout.inertia_hat[:, joint], label=f"I_hat_{joint + 1}")
        axes[1].plot(rollout.time, rollout.damping_hat[:, joint], label=f"D_hat_{joint + 1}")
        axes[2].plot(rollout.time, rollout.bias_hat[:, joint], label=f"b_hat_{joint + 1}")
        axes[2].axhline(config.dynamics.disturbance_constant[joint], lw=1.0, ls="--", alpha=0.7)
    axes[0].set_title("Robust Controller Inertia Estimates")
    axes[0].set_ylabel("inertia")
    axes[1].set_title("Robust Controller Damping Estimates")
    axes[1].set_ylabel("damping")
    axes[2].set_title("Robust Controller Constant Bias Estimates")
    axes[2].set_ylabel("bias torque")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.28)
        ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_control(
    config: ProjectConfig,
    rollouts: dict[str, Rollout],
    figures_dir: Path,
) -> Path:
    path = figures_dir / "control_torques.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    rollout = rollouts["robust"]
    for joint in range(config.robot.link_lengths.size):
        ax.plot(rollout.time, rollout.tau[:, joint], label=f"tau_{joint + 1}")
        ax.axhline(config.dynamics.torque_limits[joint], color="black", ls="--", lw=0.7, alpha=0.45)
        ax.axhline(-config.dynamics.torque_limits[joint], color="black", ls="--", lw=0.7, alpha=0.45)
    ax.set_title("Robust Control Torques")
    ax.set_ylabel("torque")
    ax.set_xlabel("time [s]")
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
