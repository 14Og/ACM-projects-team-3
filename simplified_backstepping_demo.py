"""
Simplified decentralized backstepping control for a 3DOF planar manipulator.

This script intentionally does NOT use the full nonlinear robot dynamics.
It uses:
    M_diag q_ddot + G(q) = tau
    C(q, q_dot) = 0

Run:
    python simplified_backstepping_demo.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# Tunable Parameters
# =============================================================================

LENGTHS = np.array([90.0, 70.0, 40.0], dtype=float)    # same units as the full demo
MASSES = np.array([1.0, 0.7, 0.6], dtype=float)
M_DIAG = np.diag([1.2, 0.8, 0.45])                    # constant diagonal inertia
M_INV = np.linalg.inv(M_DIAG)
G_CONST = 9.81

K1 = np.diag([10.0, 10.0, 10.0])
K2 = np.diag([10.0, 10.0, 10.0])

T_START = 0.0
T_END = 12.0
N_SAMPLES = 2000
OUTPUT_DIR = Path("outputs/simplified_backstepping")
SAVE_GIF = True
GIF_FPS = 30

ELLIPSE_CENTER_XY = np.array([60.0, 0.0], dtype=float)
ELLIPSE_RADII_XY = np.array([165.0, 45.0], dtype=float)
ELLIPSE_OMEGA = 0.7
ELLIPSE_ORIENTATION = 0.0
ELLIPSE_ELBOW = 1
FIT_ELLIPSE_TO_WORKSPACE = True
PROJECT_ELLIPSE_TO_WORKSPACE = True
WORKSPACE_MARGIN = 2.0
DERIVATIVE_DT = 1e-4
INITIAL_Q_ERROR = np.array([0.15, -0.10, 0.08], dtype=float)
SETTLING_TIME = 1.0


# =============================================================================
# Robot Model Helpers
# =============================================================================

def gravity_vector(q: np.ndarray) -> np.ndarray:
    """
    Gravity vector for a 3DOF planar manipulator.

    Angles are relative joint angles. Each link center of mass is at half length.
    This is the standard gravity term from potential energy differentiation.
    """
    q1, q2, q3 = q
    l1, l2, l3 = LENGTHS
    m1, m2, m3 = MASSES

    a1 = q1
    a2 = q1 + q2
    a3 = q1 + q2 + q3

    g1 = (
        (m1 * l1 / 2.0 + m2 * l1 + m3 * l1) * G_CONST * math.cos(a1)
        + (m2 * l2 / 2.0 + m3 * l2) * G_CONST * math.cos(a2)
        + m3 * l3 / 2.0 * G_CONST * math.cos(a3)
    )
    g2 = (
        (m2 * l2 / 2.0 + m3 * l2) * G_CONST * math.cos(a2)
        + m3 * l3 / 2.0 * G_CONST * math.cos(a3)
    )
    g3 = m3 * l3 / 2.0 * G_CONST * math.cos(a3)
    return np.array([g1, g2, g3], dtype=float)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Return XY positions of base, joints, and end-effector."""
    points = [np.array([0.0, 0.0], dtype=float)]
    position = points[0].copy()
    angle = 0.0

    for theta, length in zip(q, LENGTHS):
        angle += float(theta)
        position = position + np.array(
            [length * math.cos(angle), length * math.sin(angle)],
            dtype=float,
        )
        points.append(position.copy())

    return np.array(points)


# =============================================================================
# Desired End-Effector Ellipse and IK
# =============================================================================

def wrap_angles(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q) + math.pi) % (2.0 * math.pi) - math.pi


def ellipse_reference(t: float, scale: float = 1.0) -> np.ndarray:
    radii = ELLIPSE_RADII_XY * float(scale)
    return ELLIPSE_CENTER_XY + np.array(
        [
            radii[0] * math.cos(ELLIPSE_OMEGA * t),
            radii[1] * math.sin(ELLIPSE_OMEGA * t),
        ],
        dtype=float,
    )


def is_inside_fixed_orientation_workspace(ee_xy: np.ndarray) -> bool:
    l1, l2, l3 = LENGTHS
    tool = np.array(
        [l3 * math.cos(ELLIPSE_ORIENTATION), l3 * math.sin(ELLIPSE_ORIENTATION)],
        dtype=float,
    )
    wrist_radius = float(np.linalg.norm(np.asarray(ee_xy, dtype=float) - tool))
    min_radius = abs(l1 - l2) + WORKSPACE_MARGIN
    max_radius = l1 + l2 - WORKSPACE_MARGIN
    return min_radius <= wrist_radius <= max_radius


def ellipse_workspace_scale(samples: int = 720) -> float:
    if not FIT_ELLIPSE_TO_WORKSPACE:
        return 1.0

    period = 2.0 * math.pi / ELLIPSE_OMEGA
    for scale in np.linspace(1.0, 0.0, 1001):
        is_valid = True
        for t in np.linspace(0.0, period, samples, endpoint=False):
            if not is_inside_fixed_orientation_workspace(ellipse_reference(float(t), float(scale))):
                is_valid = False
                break
        if is_valid:
            return float(scale)
    return 1.0


ELLIPSE_SCALE = ellipse_workspace_scale()


def project_ee_to_ik_workspace(ee_xy: np.ndarray) -> np.ndarray:
    l1, l2, l3 = LENGTHS
    tool = np.array(
        [l3 * math.cos(ELLIPSE_ORIENTATION), l3 * math.sin(ELLIPSE_ORIENTATION)],
        dtype=float,
    )
    wrist = np.asarray(ee_xy, dtype=float) - tool
    wrist_radius = float(np.linalg.norm(wrist))
    min_radius = abs(l1 - l2) + WORKSPACE_MARGIN
    max_radius = l1 + l2 - WORKSPACE_MARGIN

    if wrist_radius < 1e-9:
        wrist = np.array([min_radius, 0.0], dtype=float)
    else:
        target_radius = float(np.clip(wrist_radius, min_radius, max_radius))
        wrist = wrist * (target_radius / wrist_radius)

    return wrist + tool


def desired_end_effector(t: float) -> np.ndarray:
    ee_d = ellipse_reference(t, ELLIPSE_SCALE)
    if PROJECT_ELLIPSE_TO_WORKSPACE:
        return project_ee_to_ik_workspace(ee_d)
    return ee_d


def inverse_kinematics_3dof(ee_xy: np.ndarray) -> np.ndarray:
    l1, l2, l3 = LENGTHS
    x, y = np.asarray(ee_xy, dtype=float)
    wrist = np.array(
        [
            x - l3 * math.cos(ELLIPSE_ORIENTATION),
            y - l3 * math.sin(ELLIPSE_ORIENTATION),
        ],
        dtype=float,
    )

    wrist_radius = float(np.linalg.norm(wrist))
    min_radius = abs(l1 - l2) + 1e-9
    max_radius = l1 + l2 - 1e-9
    if wrist_radius < min_radius or wrist_radius > max_radius:
        raise ValueError("Ellipse point is outside the fixed-orientation IK workspace.")

    cos_q2 = (wrist @ wrist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = float(np.sign(ELLIPSE_ELBOW) or 1.0) * math.acos(cos_q2)
    q1 = math.atan2(wrist[1], wrist[0]) - math.atan2(
        l2 * math.sin(q2),
        l1 + l2 * math.cos(q2),
    )
    q3 = ELLIPSE_ORIENTATION - q1 - q2
    return wrap_angles(np.array([q1, q2, q3], dtype=float))


def desired_joint_position(t: float) -> np.ndarray:
    return inverse_kinematics_3dof(desired_end_effector(t))


def desired_trajectory(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same desired trajectory as the full method: XY ellipse -> IK -> q_d(t)."""
    h = DERIVATIVE_DT
    q_minus = desired_joint_position(t - h)
    q_d = desired_joint_position(t)
    q_plus = desired_joint_position(t + h)
    q_stack = np.unwrap(np.vstack([q_minus, q_d, q_plus]), axis=0)

    q_d = q_stack[1]
    q_d_dot = (q_stack[2] - q_stack[0]) / (2.0 * h)
    q_d_ddot = (q_stack[2] - 2.0 * q_stack[1] + q_stack[0]) / (h * h)
    return q_d, q_d_dot, q_d_ddot


# =============================================================================
# Backstepping Controller and Simplified Dynamics
# =============================================================================

def backstepping_control(
    t: float,
    q: np.ndarray,
    q_dot: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    q_d, q_d_dot, q_d_ddot = desired_trajectory(t)

    z1 = wrap_angles(q - q_d)
    alpha = q_d_dot - K1 @ z1
    z2 = q_dot - alpha
    alpha_dot = q_d_ddot - K1 @ (q_dot - q_d_dot)

    tau = M_DIAG @ (alpha_dot - z1 - K2 @ z2) + gravity_vector(q)
    V = 0.5 * z1.T @ z1 + 0.5 * z2.T @ z2
    V_dot_nominal = -z1.T @ K1 @ z1 - z2.T @ K2 @ z2

    return tau, {
        "q_d": q_d,
        "q_d_dot": q_d_dot,
        "q_d_ddot": q_d_ddot,
        "z1": z1,
        "z2": z2,
        "V": float(V),
        "V_dot_nominal": float(V_dot_nominal),
    }


def robot_dynamics(t: float, state: np.ndarray) -> np.ndarray:
    q = state[:3]
    q_dot = state[3:]

    tau, _ = backstepping_control(t, q, q_dot)

    # Simplified decentralized dynamics:
    # M_diag q_ddot + G(q) = tau, with C(q, q_dot) = 0.
    q_ddot = M_INV @ (tau - gravity_vector(q))
    return np.concatenate([q_dot, q_ddot])


# =============================================================================
# Simulation and Plotting
# =============================================================================

def simulate() -> dict[str, np.ndarray]:
    t_eval = np.linspace(T_START, T_END, N_SAMPLES)

    q_d0, q_d_dot0, _ = desired_trajectory(T_START)
    q0 = q_d0 + INITIAL_Q_ERROR
    q_dot0 = q_d_dot0 + np.array([0.0, 0.0, 0.0], dtype=float)
    state0 = np.concatenate([q0, q_dot0])

    solution = solve_ivp(
        robot_dynamics,
        (T_START, T_END),
        state0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    q = solution.y[:3].T
    q_dot = solution.y[3:].T

    q_d = []
    z1 = []
    z2 = []
    tau = []
    V = []
    V_dot_nominal = []
    ee = []
    ee_d = []

    for t, q_now, q_dot_now in zip(solution.t, q, q_dot):
        tau_now, data = backstepping_control(float(t), q_now, q_dot_now)
        q_d_now = data["q_d"]

        q_d.append(q_d_now)
        z1.append(data["z1"])
        z2.append(data["z2"])
        tau.append(tau_now)
        V.append(data["V"])
        V_dot_nominal.append(data["V_dot_nominal"])
        ee.append(forward_kinematics(q_now)[-1])
        ee_d.append(forward_kinematics(q_d_now)[-1])

    return {
        "time": solution.t,
        "q": q,
        "q_dot": q_dot,
        "q_d": np.asarray(q_d),
        "z1": np.asarray(z1),
        "z2": np.asarray(z2),
        "tau": np.asarray(tau),
        "V": np.asarray(V),
        "V_dot_nominal": np.asarray(V_dot_nominal),
        "ee": np.asarray(ee),
        "ee_d": np.asarray(ee_d),
        "ellipse_scale": np.array([ELLIPSE_SCALE], dtype=float),
    }


def plot_results(log: dict[str, np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t = log["time"]
    z1 = log["z1"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, z1[:, i], label=f"z1_{i + 1}")
        ax.axhline(0.0, color="k", lw=0.7, ls="--")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time, s")
    fig.suptitle("Joint tracking errors")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tracking_errors.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 7))
    steady_mask = log["time"] >= SETTLING_TIME
    ax.plot(log["ee_d"][:, 0], log["ee_d"][:, 1], "--", label="reachable desired EE")
    ax.plot(
        log["ee"][~steady_mask, 0],
        log["ee"][~steady_mask, 1],
        color="tab:green",
        alpha=0.35,
        label="actual transient",
    )
    ax.plot(
        log["ee"][steady_mask, 0],
        log["ee"][steady_mask, 1],
        color="tab:green",
        label=f"actual EE after {SETTLING_TIME:g}s",
    )
    ax.set_title("End-effector trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "end_effector_trajectory.png", dpi=160)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, log["V"], label="V")
    ax.set_title("Lyapunov function")
    ax.set_xlabel("time, s")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lyapunov.png", dpi=160)

    plt.show()


def animate_results(log: dict[str, np.ndarray]) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not SAVE_GIF:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t = log["time"]
    q = log["q"]
    q_d = log["q_d"]
    ee = log["ee"]
    stride = max(1, len(t) // 450)
    frames = np.arange(0, len(t), stride)

    all_xy = np.vstack([log["ee_d"], log["ee"]])
    pad = 25.0
    x_min, y_min = np.min(all_xy, axis=0) - pad
    x_max, y_max = np.max(all_xy, axis=0) + pad

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Simplified model: backstepping ellipse tracking")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.plot(log["ee_d"][:, 0], log["ee_d"][:, 1], "--", color="tab:orange", label="reachable desired EE")
    trace_line, = ax.plot([], [], color="tab:green", lw=1.8, label="actual EE trace")
    desired_arm, = ax.plot([], [], "o--", color="tab:orange", lw=1.5, markersize=4, alpha=0.8)
    actual_arm, = ax.plot([], [], "o-", color="tab:green", lw=3.0, markersize=5)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right")

    def update(frame: int):
        joints = forward_kinematics(q[frame])
        joints_d = forward_kinematics(q_d[frame])

        trace_line.set_data(ee[: frame + 1, 0], ee[: frame + 1, 1])
        desired_arm.set_data(joints_d[:, 0], joints_d[:, 1])
        actual_arm.set_data(joints[:, 0], joints[:, 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        return trace_line, desired_arm, actual_arm, time_text

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / GIF_FPS, blit=True)
    gif_path = OUTPUT_DIR / "simplified_tracking.gif"
    anim.save(gif_path, writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)


def main() -> None:
    log = simulate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "simulation_log.npz", **log)
    try:
        animate_results(log)
    except Exception as exc:
        print(f"GIF was not saved: {exc}")
    plot_results(log)

    print(f"Saved results to {OUTPUT_DIR}")
    if SAVE_GIF:
        print(f"GIF path: {OUTPUT_DIR / 'simplified_tracking.gif'}")
    print(f"Ellipse workspace scale = {ELLIPSE_SCALE:.3f}")
    print(f"Final ||z1|| = {np.linalg.norm(log['z1'][-1]):.6f}")
    print(f"Max nominal V_dot = {np.max(log['V_dot_nominal']):.6e}")


if __name__ == "__main__":
    main()
