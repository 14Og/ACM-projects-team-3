import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import BacksteppingConfig, RobotConfig
from .physics_robot import Robot_Dynamic_3DOF


def wrap_angles(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q) + math.pi) % (2.0 * math.pi) - math.pi


def forward_kinematics(
    q: np.ndarray,
    link_lengths: np.ndarray,
    base_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return planar joint positions [base, joint1, joint2, end-effector]."""
    base = np.zeros(2, dtype=float) if base_xy is None else np.asarray(base_xy, dtype=float)
    points = [base.copy()]
    p = base.copy()
    angle = 0.0

    for theta, length in zip(q, link_lengths):
        angle += float(theta)
        p = p + np.array([length * math.cos(angle), length * math.sin(angle)], dtype=float)
        points.append(p.copy())

    return np.asarray(points, dtype=float)


def ellipse_reference(t: float, cfg: BacksteppingConfig) -> np.ndarray:
    center = np.asarray(cfg.ellipse_center_xy, dtype=float)
    radii = np.asarray(cfg.ellipse_radii_xy, dtype=float)
    w = float(cfg.omega)
    return center + np.array(
        [radii[0] * math.cos(w * t), radii[1] * math.sin(w * t)],
        dtype=float,
    )


def scaled_ellipse_reference(t: float, cfg: BacksteppingConfig, scale: float) -> np.ndarray:
    center = np.asarray(cfg.ellipse_center_xy, dtype=float)
    radii = np.asarray(cfg.ellipse_radii_xy, dtype=float) * float(scale)
    w = float(cfg.omega)
    return center + np.array(
        [radii[0] * math.cos(w * t), radii[1] * math.sin(w * t)],
        dtype=float,
    )


def _is_inside_fixed_orientation_workspace(
    ee_xy: np.ndarray,
    link_lengths: np.ndarray,
    orientation: float,
    margin: float,
) -> bool:
    l1, l2, l3 = np.asarray(link_lengths, dtype=float)
    tool = np.array([l3 * math.cos(orientation), l3 * math.sin(orientation)], dtype=float)
    wrist_radius = float(np.linalg.norm(np.asarray(ee_xy, dtype=float) - tool))
    min_radius = abs(l1 - l2) + float(margin)
    max_radius = l1 + l2 - float(margin)
    return min_radius <= wrist_radius <= max_radius


def ellipse_workspace_scale(
    cfg: BacksteppingConfig,
    link_lengths: np.ndarray,
    samples: int = 720,
) -> float:
    """Largest sampled scale in [0, 1] that keeps the whole ellipse reachable."""
    if not cfg.fit_ellipse_to_workspace:
        return 1.0

    orientation = float(cfg.ellipse_orientation)
    margin = float(cfg.workspace_margin)
    period = 2.0 * math.pi / float(cfg.omega)

    for scale in np.linspace(1.0, 0.0, 1001):
        ok = True
        for t in np.linspace(0.0, period, samples, endpoint=False):
            ee = scaled_ellipse_reference(float(t), cfg, float(scale))
            if not _is_inside_fixed_orientation_workspace(ee, link_lengths, orientation, margin):
                ok = False
                break
        if ok:
            return float(scale)

    return 1.0


def project_ee_to_ik_workspace(
    ee_xy: np.ndarray,
    link_lengths: np.ndarray,
    orientation: float,
    margin: float,
) -> np.ndarray:
    """Project an EE point to the fixed-orientation 3DOF IK workspace."""
    l1, l2, l3 = np.asarray(link_lengths, dtype=float)
    tool = np.array([l3 * math.cos(orientation), l3 * math.sin(orientation)], dtype=float)
    wrist = np.asarray(ee_xy, dtype=float) - tool
    wrist_radius = float(np.linalg.norm(wrist))

    min_radius = abs(l1 - l2) + float(margin)
    max_radius = l1 + l2 - float(margin)
    if wrist_radius < 1e-9:
        wrist = np.array([min_radius, 0.0], dtype=float)
    else:
        target_radius = float(np.clip(wrist_radius, min_radius, max_radius))
        wrist = wrist * (target_radius / wrist_radius)

    return wrist + tool


def inverse_kinematics_3dof(
    ee_xy: np.ndarray,
    link_lengths: np.ndarray,
    orientation: float = 0.0,
    elbow: int = 1,
) -> np.ndarray:
    """Geometric IK for a 3-link planar arm with fixed end-effector orientation."""
    l1, l2, l3 = np.asarray(link_lengths, dtype=float)
    x, y = np.asarray(ee_xy, dtype=float)
    wrist = np.array(
        [x - l3 * math.cos(orientation), y - l3 * math.sin(orientation)],
        dtype=float,
    )

    r = float(np.linalg.norm(wrist))
    min_radius = abs(l1 - l2) + 1e-9
    max_radius = l1 + l2 - 1e-9
    if r > max_radius or r < min_radius:
        raise ValueError(
            "IK target is outside the fixed-orientation workspace. "
            "Enable project_ellipse_to_workspace or reduce the ellipse."
        )

    r2 = float(wrist @ wrist)
    cos_q2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = float(np.sign(elbow) or 1.0) * math.acos(cos_q2)
    q1 = math.atan2(wrist[1], wrist[0]) - math.atan2(
        l2 * math.sin(q2),
        l1 + l2 * math.cos(q2),
    )
    q3 = float(orientation) - q1 - q2
    return wrap_angles(np.array([q1, q2, q3], dtype=float))


@dataclass
class BacksteppingController:
    physics: Robot_Dynamic_3DOF
    cfg: BacksteppingConfig
    link_lengths: np.ndarray

    def __post_init__(self) -> None:
        self.amplitudes = np.asarray(self.cfg.amplitudes, dtype=float)
        self.k1 = np.diag(np.asarray(self.cfg.k1, dtype=float))
        self.k2 = np.diag(np.asarray(self.cfg.k2, dtype=float))
        self.damping = float(self.physics.cfg.damping)
        self.ellipse_scale = ellipse_workspace_scale(self.cfg, self.link_lengths)

    def desired_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return desired joint position, velocity, and acceleration."""
        if self.cfg.trajectory_mode == "ellipse":
            return self._ellipse_joint_trajectory(t)
        if self.cfg.trajectory_mode == "joint_sine":
            return self._joint_sine_trajectory(t)
        raise ValueError(f"Unknown trajectory_mode: {self.cfg.trajectory_mode!r}")

    def _joint_sine_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1, a2, a3 = self.amplitudes
        w = float(self.cfg.omega)

        q_d = np.array(
            [
                a1 * math.sin(w * t),
                a2 * math.cos(w * t),
                a3 * math.sin(2.0 * w * t),
            ],
            dtype=float,
        )
        q_d_dot = np.array(
            [
                a1 * w * math.cos(w * t),
                -a2 * w * math.sin(w * t),
                2.0 * a3 * w * math.cos(2.0 * w * t),
            ],
            dtype=float,
        )
        q_d_ddot = np.array(
            [
                -a1 * w * w * math.sin(w * t),
                -a2 * w * w * math.cos(w * t),
                -4.0 * a3 * w * w * math.sin(2.0 * w * t),
            ],
            dtype=float,
        )
        return q_d, q_d_dot, q_d_ddot

    def _ellipse_joint_position(self, t: float) -> np.ndarray:
        ee_d = self.desired_end_effector(t)
        return inverse_kinematics_3dof(
            ee_d,
            self.link_lengths,
            orientation=float(self.cfg.ellipse_orientation),
            elbow=int(self.cfg.ellipse_elbow),
        )

    def desired_end_effector(self, t: float) -> np.ndarray:
        ee_d = scaled_ellipse_reference(t, self.cfg, self.ellipse_scale)
        if not self.cfg.project_ellipse_to_workspace:
            return ee_d
        return project_ee_to_ik_workspace(
            ee_d,
            self.link_lengths,
            orientation=float(self.cfg.ellipse_orientation),
            margin=float(self.cfg.workspace_margin),
        )

    def _ellipse_joint_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The ellipse is analytical in XY. For backstepping we need q_d_dot and
        # q_d_ddot, so we differentiate the IK-generated q_d(t) by central
        # differences. np.unwrap keeps angles continuous around +/-pi.
        h = float(self.cfg.derivative_dt)
        q_minus = self._ellipse_joint_position(t - h)
        q_d = self._ellipse_joint_position(t)
        q_plus = self._ellipse_joint_position(t + h)
        q_stack = np.unwrap(np.vstack([q_minus, q_d, q_plus]), axis=0)

        q_d = q_stack[1]
        q_d_dot = (q_stack[2] - q_stack[0]) / (2.0 * h)
        q_d_ddot = (q_stack[2] - 2.0 * q_stack[1] + q_stack[0]) / (h * h)
        return q_d, q_d_dot, q_d_ddot

    def errors(self, t: float, q: np.ndarray, q_dot: np.ndarray) -> Dict[str, np.ndarray]:
        q_d, q_d_dot, q_d_ddot = self.desired_trajectory(t)
        z1 = wrap_angles(q - q_d)
        alpha = q_d_dot - self.k1 @ z1
        z2 = q_dot - alpha
        alpha_dot = q_d_ddot - self.k1 @ (q_dot - q_d_dot)
        return {
            "q_d": q_d,
            "q_d_dot": q_d_dot,
            "q_d_ddot": q_d_ddot,
            "z1": z1,
            "z2": z2,
            "alpha": alpha,
            "alpha_dot": alpha_dot,
        }

    def control(self, t: float, q: np.ndarray, q_dot: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        data = self.errors(t, q, q_dot)
        z1 = data["z1"]
        z2 = data["z2"]
        alpha_dot = data["alpha_dot"]

        M, G = self.physics.get_matrices(q)
        c_qdot = self.physics.coriolis_times_velocity(q, q_dot)

        # Backstepping design:
        # Given z1 = q - q_d, alpha = q_d_dot - K1 z1, z2 = q_dot - alpha,
        # then z1_dot = z2 - K1 z1.
        #
        # The nominal Lyapunov candidate is
        # V = 0.5 z1.T z1 + 0.5 z2.T z2.
        #
        # Choosing
        # tau = M(q)(alpha_dot - z1 - K2 z2) + C(q, q_dot)q_dot + G(q)
        # gives
        # V_dot = -z1.T K1 z1 - z2.T K2 z2 <= 0
        # for positive definite K1 and K2, so the tracking error is
        # asymptotically stable in the nominal manipulator model.
        tau = M @ (alpha_dot - z1 - self.k2 @ z2) + c_qdot + G

        # The project simulator also subtracts viscous damping in forward dynamics.
        # Compensating it here keeps the implemented closed loop close to the
        # nominal Lyapunov derivation above.
        if self.cfg.compensate_damping:
            tau = tau + self.damping * q_dot

        if self.cfg.torque_clip is not None:
            limits = np.asarray(self.cfg.torque_clip, dtype=float)
            tau = np.clip(tau, -limits, limits)

        data["tau"] = tau
        data["V"] = np.array(0.5 * z1.T @ z1 + 0.5 * z2.T @ z2, dtype=float)
        data["V_dot_nominal"] = np.array(
            -z1.T @ self.k1 @ z1 - z2.T @ self.k2 @ z2,
            dtype=float,
        )
        return tau, data


def _rk4_step(controller: BacksteppingController, state: np.ndarray, t: float, dt: float) -> np.ndarray:
    def rhs(t_local: float, x: np.ndarray) -> np.ndarray:
        q = x[:3]
        q_dot = x[3:]
        tau, _ = controller.control(t_local, q, q_dot)
        _, q_ddot = controller.physics.dynamics(q, q_dot, tau)
        return np.concatenate([q_dot, q_ddot])

    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(t + dt, state + dt * k3)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[:3] = wrap_angles(next_state[:3])
    return next_state


def simulate_backstepping_tracking(
    robot_cfg: Optional[RobotConfig] = None,
    tracking_cfg: Optional[BacksteppingConfig] = None,
) -> Dict[str, np.ndarray]:
    robot_cfg = robot_cfg or RobotConfig()
    tracking_cfg = tracking_cfg or BacksteppingConfig()

    link_lengths_px = np.asarray(robot_cfg.link_lengths, dtype=float)
    physics = Robot_Dynamic_3DOF(
        masses=np.asarray(robot_cfg.masses, dtype=float),
        lengthes=link_lengths_px / 200.0,
        cfg=robot_cfg,
    )
    controller = BacksteppingController(
        physics=physics,
        cfg=tracking_cfg,
        link_lengths=link_lengths_px,
    )

    dt = float(tracking_cfg.dt)
    times = np.arange(0.0, float(tracking_cfg.duration) + 0.5 * dt, dt)
    q_d0, q_d_dot0, _ = controller.desired_trajectory(0.0)
    initial_q = (
        q_d0 + np.asarray(tracking_cfg.initial_q_error, dtype=float)
        if tracking_cfg.initial_q is None
        else np.asarray(tracking_cfg.initial_q, dtype=float)
    )
    initial_q_dot = (
        q_d_dot0 + np.asarray(tracking_cfg.initial_q_dot_error, dtype=float)
        if tracking_cfg.initial_q_dot is None
        else np.asarray(tracking_cfg.initial_q_dot, dtype=float)
    )
    state = np.concatenate([initial_q, initial_q_dot])

    log = {
        "time": [],
        "q": [],
        "q_dot": [],
        "q_d": [],
        "q_d_dot": [],
        "tau": [],
        "z1": [],
        "z2": [],
        "V": [],
        "V_dot_nominal": [],
        "ee": [],
        "ee_d": [],
        "ee_ref_raw": [],
        "ellipse_scale": [],
        "joints": [],
        "joints_d": [],
    }

    for t in times:
        q = state[:3].copy()
        q_dot = state[3:].copy()
        tau, data = controller.control(float(t), q, q_dot)
        joints = forward_kinematics(q, link_lengths_px)
        joints_d = forward_kinematics(data["q_d"], link_lengths_px)
        ee_ref_raw = ellipse_reference(float(t), tracking_cfg)

        log["time"].append(float(t))
        log["q"].append(q)
        log["q_dot"].append(q_dot)
        log["q_d"].append(data["q_d"])
        log["q_d_dot"].append(data["q_d_dot"])
        log["tau"].append(tau)
        log["z1"].append(data["z1"])
        log["z2"].append(data["z2"])
        log["V"].append(float(data["V"]))
        log["V_dot_nominal"].append(float(data["V_dot_nominal"]))
        log["ee"].append(joints[-1])
        log["ee_d"].append(joints_d[-1])
        log["ee_ref_raw"].append(ee_ref_raw)
        log["ellipse_scale"].append(controller.ellipse_scale)
        log["joints"].append(joints)
        log["joints_d"].append(joints_d)

        state = _rk4_step(controller, state, float(t), dt)

    return {key: np.asarray(value) for key, value in log.items()}


def plot_backstepping_results(log: Dict[str, np.ndarray], output_dir: str) -> None:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = log["time"]
    joint_labels = ["q1", "q2", "q3"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, log["q"][:, i], label=f"{joint_labels[i]} actual")
        ax.plot(t, log["q_d"][:, i], "--", label=f"{joint_labels[i]} desired")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time, s")
    fig.suptitle("Joint tracking")
    fig.tight_layout()
    fig.savefig(out / "joint_tracking.png", dpi=160)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(t, log["z1"][:, i], label=f"z1_{i + 1}")
    ax.set_title("Tracking error z1 = q - q_d")
    ax.set_xlabel("time, s")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "tracking_error.png", dpi=160)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, np.linalg.norm(log["z1"], axis=1), color="#C62828", label="||z1||")
    ax.set_title("Tracking error norm")
    ax.set_xlabel("time, s")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "tracking_error_norm.png", dpi=160)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(t, log["tau"][:, i], label=f"tau_{i + 1}")
    ax.set_title("Control torques")
    ax.set_xlabel("time, s")
    ax.set_ylabel("Nm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "torques.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 7))
    if "ee_ref_raw" in log:
        ax.plot(
            log["ee_ref_raw"][:, 0],
            log["ee_ref_raw"][:, 1],
            ":",
            color="#9E9E9E",
            label="requested ellipse",
        )
    ax.plot(log["ee"][:, 0], log["ee"][:, 1], label="actual EE")
    ax.plot(log["ee_d"][:, 0], log["ee_d"][:, 1], "--", label="reachable desired EE")
    ax.set_title("End-effector 2D trajectory")
    ax.set_xlabel("x, px")
    ax.set_ylabel("y, px")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "end_effector_trajectory.png", dpi=160)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, log["V"], color="#2E7D32", label="V")
    ax.set_title("Lyapunov function")
    ax.set_xlabel("time, s")
    ax.set_ylabel("V")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "lyapunov.png", dpi=160)

    if "V_dot_nominal" in log:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, log["V_dot_nominal"], color="#1565C0", label="nominal V_dot")
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title("Nominal Lyapunov derivative")
        ax.set_xlabel("time, s")
        ax.set_ylabel("V_dot")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out / "lyapunov_derivative.png", dpi=160)

    plt.close("all")


def animate_backstepping(log: Dict[str, np.ndarray], output_dir: str, save_gif: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    t = log["time"]
    joints = log["joints"]
    joints_d = log["joints_d"]
    stride = max(1, len(t) // 600)
    frames = np.arange(0, len(t), stride)

    reach = float(np.max(np.linalg.norm(joints[:, -1, :], axis=1))) + 80.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Backstepping tracking: planar 3DOF manipulator")

    desired_line, = ax.plot([], [], "--", color="#757575", lw=2, label="desired")
    actual_line, = ax.plot([], [], "-o", color="#1565C0", lw=3, markersize=5, label="actual")
    trace_line, = ax.plot([], [], color="#C62828", lw=1.5, alpha=0.75, label="EE trace")
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.legend(loc="best")

    def update(frame_idx: int):
        frame = int(frame_idx)
        desired_line.set_data(joints_d[frame, :, 0], joints_d[frame, :, 1])
        actual_line.set_data(joints[frame, :, 0], joints[frame, :, 1])
        trace_line.set_data(log["ee"][: frame + 1, 0], log["ee"][: frame + 1, 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        return desired_line, actual_line, trace_line, time_text

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

    if save_gif:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            anim.save(out / "manipulator_tracking.gif", writer="pillow", fps=30)
        except Exception as exc:
            print(f"[backstepping] animation was not saved: {exc}")

    plt.show()


def run_backstepping_tracking(
    robot_cfg: Optional[RobotConfig] = None,
    tracking_cfg: Optional[BacksteppingConfig] = None,
    show_animation: bool = True,
    save_animation: bool = False,
) -> Dict[str, np.ndarray]:
    robot_cfg = robot_cfg or RobotConfig()
    tracking_cfg = tracking_cfg or BacksteppingConfig()
    log = simulate_backstepping_tracking(robot_cfg=robot_cfg, tracking_cfg=tracking_cfg)

    out = Path(tracking_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "simulation_log.npz", **log)

    plots_saved = True
    try:
        plot_backstepping_results(log, tracking_cfg.output_dir)
    except ModuleNotFoundError as exc:
        plots_saved = False
        print(f"[backstepping] plots were not generated because a dependency is missing: {exc}")
        print("[backstepping] install the project requirements to enable matplotlib plots.")

    if show_animation:
        try:
            animate_backstepping(log, tracking_cfg.output_dir, save_gif=save_animation)
        except ModuleNotFoundError as exc:
            print(f"[backstepping] animation was not shown because a dependency is missing: {exc}")

    if plots_saved:
        print(f"[backstepping] saved plots and log to {out}")
    else:
        print(f"[backstepping] saved simulation log to {out}")
    if "ellipse_scale" in log:
        print(f"[backstepping] ellipse workspace scale = {float(log['ellipse_scale'][0]):.3f}")
    print(f"[backstepping] final ||z1|| = {np.linalg.norm(log['z1'][-1]):.6f}")
    print(f"[backstepping] final V = {log['V'][-1]:.6f}")
    return log
