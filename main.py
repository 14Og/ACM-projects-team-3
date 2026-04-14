"""Lyapunov-based obstacle avoidance and trajectory tracking demo.

Upgraded behavior:
1) Random target initial position on the ellipse.
2) Random target speed (via random angular velocity).
3) Random robot initial position outside the ellipse and away from obstacles.
4) Random obstacle placement near the ellipse, with two guaranteed obstacle pairs:
   - Pair A: narrow gap that blocks passing between obstacles.
   - Pair B: wider gap that allows passing but causes deviation.
5) Visualization over two full target cycles.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


@dataclass
class SimConfig:
    # Simulation timing
    dt: float = 0.03
    num_cycles: float = 2.0
    t_final: float = 36.0  # Overwritten after random omega is sampled.

    # Reference ellipse: (cx + a*cos(theta), cy + b*sin(theta))
    cx: float = 22.0
    cy: float = 20.0
    a: float = 14.0
    b: float = 10.0
    theta0: float = np.pi / 2.0  # Overwritten randomly.
    omega: float = 0.22  # Overwritten randomly.
    omega_range: tuple[float, float] = (0.16, 0.32)

    # Initial robot state (overwritten randomly)
    p0: tuple[float, float] = (0.0, 0.0)
    v0: tuple[float, float] = (0.0, 0.0)
    robot_spawn_scale_range: tuple[float, float] = (1.20, 1.85)
    robot_spawn_clearance: float = 1.0
    target_obstacle_clearance: float = 0.1
    min_target_robot_start_distance: float = 7.0

    # Object sizes in data units
    target_radius_vis: float = 1.5
    robot_radius_calc: float = 2.6
    robot_radius_vis: float = 2.0  # Must stay < robot_radius_calc

    # Lyapunov tracking gains
    k_p: float = 3.5
    k_v: float = 3.6
    k_p_err_boost: float = 1.1
    err_boost_dist: float = 6.0

    # Speed shaping / catch-up:
    # minimum speed anchor follows target speed; when distance error grows,
    # desired speed increases to catch up.
    k_speed: float = 2.8
    speed_err_gain: float = 0.22
    min_speed_ratio_to_target: float = 1.0
    min_speed_abs: float = 0.15
    k_catch: float = 2.2

    # Anti-stall near obstacles
    stall_speed_threshold: float = 0.35
    stall_err_threshold: float = 2.0
    stall_obstacle_clearance: float = 1.3
    k_escape: float = 4.8
    escape_vref_weight: float = 0.9

    # Obstacle repulsion settings
    influence_radius: float = 1.0
    safe_margin: float = 0.1
    k_obs: float = 1.05
    eps_dist: float = 1e-3

    # Saturation limits
    a_max: float = 13.0
    v_max: float = 13.5

    # Obstacles: random each run, near the ellipse trajectory
    obstacle_radius: float = 1.2
    # Surface gap between obstacles in each pair:
    # gap_surface = dist_between_centers - 2*obstacle_radius
    # Blocking pair must satisfy: gap_surface < robot_radius_vis
    # Passable pair must satisfy: gap_surface > robot_radius_calc
    block_gap_range: tuple[float, float] = (0.20, 1.70)
    pass_gap_range: tuple[float, float] = (5.60, 7.30)
    pair_theta_jitter: float = 0.5
    pair_theta_separation_min: float = 1.75
    block_offset_range: tuple[float, float] = (1.45, 2.20)
    pass_offset_range: tuple[float, float] = (4.00, 4.90)
    path_clearance_min: float = 0.08
    obstacles: tuple[tuple[float, float, float], ...] = (
        (20.0, 38.0, 2.0),
        (20.0, 27.0, 2.0),
        (38.0, 20.0, 2.0),
        (45.0, 20.0, 2.0),
    )

    # Runtime diagnostics (set when randomizing)
    blocking_gap: float = 0.0  # surface gap
    passable_gap: float = 0.0  # surface gap


def unit_clip(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip vector norm to max_norm."""
    n = np.linalg.norm(vec)
    if n <= max_norm or n < 1e-12:
        return vec
    return vec * (max_norm / n)


def safe_normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n > 1e-9:
        return vec / n
    if fallback is None:
        return np.array([1.0, 0.0], dtype=float)
    fb_norm = float(np.linalg.norm(fallback))
    if fb_norm > 1e-9:
        return fallback / fb_norm
    return np.array([1.0, 0.0], dtype=float)


def ellipse_point(theta: float, cfg: SimConfig, scale: float = 1.0) -> np.ndarray:
    return np.array(
        [
            cfg.cx + scale * cfg.a * np.cos(theta),
            cfg.cy + scale * cfg.b * np.sin(theta),
        ],
        dtype=float,
    )


def ellipse_level(p: np.ndarray, cfg: SimConfig) -> float:
    """Implicit ellipse level: <=1 inside/on ellipse, >1 outside."""
    x = (p[0] - cfg.cx) / cfg.a
    y = (p[1] - cfg.cy) / cfg.b
    return x * x + y * y


def safe_from_obstacles(
    p: np.ndarray,
    obstacles: tuple[tuple[float, float, float], ...],
    entity_radius: float,
    extra_clearance: float,
) -> bool:
    for ox, oy, r in obstacles:
        min_center_dist = r + entity_radius + extra_clearance
        if np.linalg.norm(p - np.array([ox, oy], dtype=float)) <= min_center_dist:
            return False
    return True


def make_straddling_pair(
    theta: float,
    cfg: SimConfig,
    rng: np.random.Generator,
    gap_surface_range: tuple[float, float],
    offset_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create two obstacles around the ellipse path so the path passes between them.

    The pair is arranged across the local normal/radial direction:
    one obstacle inward, one outward from the ellipse.
    """
    r = cfg.obstacle_radius
    c = np.array([cfg.cx, cfg.cy], dtype=float)
    p_on = ellipse_point(theta, cfg)
    radial = p_on - c
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-9:
        radial = np.array([1.0, 0.0], dtype=float)
    else:
        radial = radial / radial_norm

    for _ in range(400):
        d_in = float(rng.uniform(*offset_range))
        d_out = float(rng.uniform(*offset_range))
        p_in = p_on - d_in * radial
        p_out = p_on + d_out * radial

        center_dist = np.linalg.norm(p_out - p_in)
        gap_surface = center_dist - 2.0 * r
        if gap_surface_range[0] <= gap_surface <= gap_surface_range[1]:
            # Keep the reference trajectory point between pair obstacles.
            if d_in > (r + cfg.path_clearance_min) and d_out > (r + cfg.path_clearance_min):
                return p_in, p_out, float(gap_surface)

    raise RuntimeError("Failed to create obstacle pair with requested surface-gap range.")


def generate_obstacles(
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[float, float, float], ...], float, float]:
    """Generate 4 near-ellipse obstacles in two straddling pairs:
    - Pair A: blocking (robot cannot pass between them).
    - Pair B: passable (robot can pass but must deviate).
    """
    r = cfg.obstacle_radius

    # Surface gap rules from the request.
    block_gap_range = cfg.block_gap_range
    pass_gap_range = cfg.pass_gap_range
    if not (block_gap_range[1] < cfg.robot_radius_vis):
        raise RuntimeError("Blocking pair rule must satisfy gap_surface < robot_radius_vis.")
    if not (pass_gap_range[0] > cfg.robot_radius_calc):
        raise RuntimeError("Passable pair rule must satisfy gap_surface > robot_radius_calc.")

    for _ in range(1200):
        theta_block = float(rng.uniform(0.0, 2.0 * np.pi))
        theta_pass = float(
            (theta_block + np.pi + rng.uniform(-cfg.pair_theta_jitter, cfg.pair_theta_jitter))
            % (2.0 * np.pi)
        )

        # Ensure distinct regions along trajectory.
        dtheta = abs(((theta_pass - theta_block + np.pi) % (2.0 * np.pi)) - np.pi)
        if dtheta < cfg.pair_theta_separation_min:
            continue

        try:
            b1, b2, block_gap = make_straddling_pair(
                theta_block,
                cfg,
                rng,
                gap_surface_range=block_gap_range,
                offset_range=cfg.block_offset_range,
            )
            p1, p2, pass_gap = make_straddling_pair(
                theta_pass,
                cfg,
                rng,
                gap_surface_range=pass_gap_range,
                offset_range=cfg.pass_offset_range,
            )
        except RuntimeError:
            continue

        pts = np.vstack([b1, b2, p1, p2])
        dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

        # Avoid collisions/crowding among all obstacles.
        ok = True
        for i in range(4):
            for j in range(i + 1, 4):
                # Slightly looser for pair-internal spacing because free-gap ranges control them.
                min_extra = 0.05 if (i, j) in {(0, 1), (2, 3)} else 0.9
                if dmat[i, j] < (2.0 * r + min_extra):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Keep pair rules guaranteed after sampling.
        if not (block_gap < cfg.robot_radius_vis):
            continue
        if not (pass_gap > cfg.robot_radius_calc):
            continue

        obstacles = tuple((float(p[0]), float(p[1]), r) for p in pts)
        return obstacles, float(block_gap), float(pass_gap)

    raise RuntimeError("Failed to generate valid near-ellipse obstacle layout.")


def generate_target_theta(
    cfg: SimConfig,
    rng: np.random.Generator,
    obstacles: tuple[tuple[float, float, float], ...],
) -> float:
    """Random target start angle on ellipse, with clearance from obstacles."""
    for _ in range(800):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        p_t = ellipse_point(theta, cfg)
        if safe_from_obstacles(
            p_t,
            obstacles,
            entity_radius=cfg.target_radius_vis,
            extra_clearance=cfg.target_obstacle_clearance,
        ):
            return theta
    raise RuntimeError("Failed to sample target start away from obstacles.")


def generate_robot_start(
    cfg: SimConfig,
    rng: np.random.Generator,
    obstacles: tuple[tuple[float, float, float], ...],
    p_target0: np.ndarray,
) -> np.ndarray:
    """Random robot start outside ellipse and away from obstacles."""
    for _ in range(1200):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        scale = float(rng.uniform(*cfg.robot_spawn_scale_range))
        p = ellipse_point(theta, cfg, scale=scale)
        if ellipse_level(p, cfg) <= 1.01:
            continue
        if np.linalg.norm(p - p_target0) < cfg.min_target_robot_start_distance:
            continue
        if not safe_from_obstacles(
            p,
            obstacles,
            entity_radius=cfg.robot_radius_calc,
            extra_clearance=cfg.robot_spawn_clearance,
        ):
            continue
        return p
    raise RuntimeError("Failed to sample robot start outside ellipse and obstacle-free.")


def randomize_scenario(cfg: SimConfig, rng: np.random.Generator) -> None:
    if not (cfg.robot_radius_vis < cfg.robot_radius_calc):
        raise ValueError("robot_radius_vis must be smaller than robot_radius_calc.")

    obstacles, block_gap, pass_gap = generate_obstacles(cfg, rng)
    cfg.obstacles = obstacles
    cfg.blocking_gap = block_gap
    cfg.passable_gap = pass_gap

    cfg.theta0 = generate_target_theta(cfg, rng, cfg.obstacles)
    cfg.omega = float(rng.uniform(*cfg.omega_range))
    cfg.t_final = cfg.num_cycles * (2.0 * np.pi / abs(cfg.omega))

    p_target0 = ellipse_point(cfg.theta0, cfg)
    p_robot0 = generate_robot_start(cfg, rng, cfg.obstacles, p_target0)
    cfg.p0 = (float(p_robot0[0]), float(p_robot0[1]))
    cfg.v0 = (0.0, 0.0)


def reference_state(t: float, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Reference position/velocity on an ellipse."""
    theta = cfg.theta0 + cfg.omega * t
    p_ref = ellipse_point(theta, cfg)
    v_ref = np.array(
        [-cfg.a * cfg.omega * np.sin(theta), cfg.b * cfg.omega * np.cos(theta)],
        dtype=float,
    )
    return p_ref, v_ref


def obstacle_repulsion(p: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Gradient-based repulsive term from nearby obstacles."""
    u_rep = np.zeros(2, dtype=float)
    for ox, oy, radius in cfg.obstacles:
        c = np.array([ox, oy], dtype=float)
        delta = p - c
        center_dist = np.linalg.norm(delta)
        if center_dist < 1e-12:
            continue

        # Signed clearance from obstacle boundary to robot safety boundary.
        d = center_dist - (radius + cfg.robot_radius_calc)
        if d >= cfg.influence_radius:
            continue

        d_eff = max(d - cfg.safe_margin, cfg.eps_dist)
        influence_eff = max(cfg.influence_radius - cfg.safe_margin, cfg.eps_dist)
        grad_d = delta / center_dist
        mag = cfg.k_obs * (1.0 / d_eff - 1.0 / influence_eff) * (1.0 / (d_eff**2))
        u_rep += mag * grad_d

    return u_rep


def enforce_obstacle_clearance(
    p: np.ndarray,
    v: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Project robot outside all obstacle safety boundaries.

    This guarantees non-penetration for the robot's computed (safety) radius.
    """
    p_new = p.copy()
    v_new = v.copy()
    tol = 1e-4

    for _ in range(3):
        changed = False
        for ox, oy, r in cfg.obstacles:
            c = np.array([ox, oy], dtype=float)
            delta = p_new - c
            dist = np.linalg.norm(delta)
            min_dist = r + cfg.robot_radius_calc

            if dist < max(min_dist, 1e-12):
                if dist < 1e-9:
                    normal = np.array([1.0, 0.0], dtype=float)
                else:
                    normal = delta / dist

                p_new = c + (min_dist + tol) * normal
                vn = float(np.dot(v_new, normal))
                if vn < 0.0:
                    v_new = v_new - vn * normal
                changed = True
        if not changed:
            break

    return p_new, v_new


def control_input(
    p: np.ndarray,
    v: np.ndarray,
    p_ref: np.ndarray,
    v_ref: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Lyapunov-inspired control with speed shaping and anti-stall catch-up."""
    e_p = p - p_ref
    e_v = v - v_ref

    err_dist = float(np.linalg.norm(e_p))
    v_norm = float(np.linalg.norm(v))
    v_ref_norm = float(np.linalg.norm(v_ref))

    # Adaptive position gain: react faster as distance to target increases.
    kp_eff = cfg.k_p * (1.0 + cfg.k_p_err_boost * np.tanh(err_dist / cfg.err_boost_dist))
    u_track = -kp_eff * e_p - cfg.k_v * e_v

    # Speed-shaping term:
    # keep minimum speed around target speed and increase desired speed when far.
    v_des_min = max(cfg.min_speed_abs, cfg.min_speed_ratio_to_target * v_ref_norm)
    v_des = min(cfg.v_max, v_des_min + cfg.speed_err_gain * err_dist)
    move_dir = safe_normalize((p_ref - p) + 0.7 * v_ref, fallback=v)
    if v_norm > 1e-6:
        vel_dir = v / v_norm
    else:
        vel_dir = move_dir
    u_speed = cfg.k_speed * (v_des - v_norm) * vel_dir

    # Direct catch-up along line-of-sight to reduce large lag quickly.
    u_catch = cfg.k_catch * np.tanh(err_dist / cfg.err_boost_dist) * safe_normalize(
        p_ref - p, fallback=v_ref
    )

    u = u_track + u_speed + u_catch + obstacle_repulsion(p, cfg)

    # Anti-stall: if almost stopped near an obstacle while still far from target,
    # inject tangential motion to escape local minima and go around obstacles.
    if v_norm < cfg.stall_speed_threshold and err_dist > cfg.stall_err_threshold:
        min_clear = np.inf
        nearest_normal: np.ndarray | None = None
        for ox, oy, r in cfg.obstacles:
            c = np.array([ox, oy], dtype=float)
            delta = p - c
            center_dist = float(np.linalg.norm(delta))
            clearance = center_dist - (r + cfg.robot_radius_calc)
            if clearance < min_clear:
                min_clear = clearance
                nearest_normal = safe_normalize(delta)

        if nearest_normal is not None and min_clear < cfg.stall_obstacle_clearance:
            tangent = np.array([-nearest_normal[1], nearest_normal[0]], dtype=float)
            guide = (p_ref - p) + cfg.escape_vref_weight * v_ref
            if float(np.dot(tangent, guide)) < 0.0:
                tangent = -tangent
            u += cfg.k_escape * tangent

    return unit_clip(u, cfg.a_max)


def run_simulation(cfg: SimConfig) -> dict[str, np.ndarray]:
    n = int(np.floor(cfg.t_final / cfg.dt)) + 1
    t = np.linspace(0.0, cfg.t_final, n)

    p = np.zeros((n, 2), dtype=float)
    v = np.zeros((n, 2), dtype=float)
    u = np.zeros((n, 2), dtype=float)
    p_ref = np.zeros((n, 2), dtype=float)
    v_ref = np.zeros((n, 2), dtype=float)

    p[0] = np.array(cfg.p0, dtype=float)
    v[0] = np.array(cfg.v0, dtype=float)
    p_ref[0], v_ref[0] = reference_state(t[0], cfg)

    for k in range(n - 1):
        p_ref[k], v_ref[k] = reference_state(t[k], cfg)
        u[k] = control_input(p[k], v[k], p_ref[k], v_ref[k], cfg)
        v[k + 1] = unit_clip(v[k] + cfg.dt * u[k], cfg.v_max)
        p[k + 1] = p[k] + cfg.dt * v[k + 1]
        p[k + 1], v[k + 1] = enforce_obstacle_clearance(p[k + 1], v[k + 1], cfg)

    p_ref[-1], v_ref[-1] = reference_state(t[-1], cfg)
    u[-1] = control_input(p[-1], v[-1], p_ref[-1], v_ref[-1], cfg)

    err = np.linalg.norm(p - p_ref, axis=1)
    speed = np.linalg.norm(v, axis=1)
    accel = np.linalg.norm(u, axis=1)

    return {
        "t": t,
        "p": p,
        "v": v,
        "u": u,
        "p_ref": p_ref,
        "v_ref": v_ref,
        "err": err,
        "speed": speed,
        "accel": accel,
    }


def make_trajectory_animation(
    data: dict[str, np.ndarray],
    cfg: SimConfig,
    frame_step: int = 3,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    t = data["t"]
    p = data["p"]
    v = data["v"]
    p_ref = data["p_ref"]
    speed = data["speed"]

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    fig.canvas.manager.set_window_title("Tracking & Obstacle Avoidance")

    theta_grid = np.linspace(0.0, 2.0 * np.pi, 500)
    ellipse_x = cfg.cx + cfg.a * np.cos(theta_grid)
    ellipse_y = cfg.cy + cfg.b * np.sin(theta_grid)
    ax.plot(
        ellipse_x,
        ellipse_y,
        linestyle=(0, (1.5, 3.0)),
        linewidth=1.6,
        color="#d9b454",
        alpha=0.55,
        label="Reference ellipse",
    )

    for ox, oy, r in cfg.obstacles:
        ax.add_patch(
            plt.Circle((ox, oy), r, facecolor="#f05b6a", edgecolor="#7a2630", alpha=0.7)
        )
        ax.plot(ox, oy, marker="P", markersize=10, color="#4b1ea6")

    (line_ref,) = ax.plot([], [], color="#e6b94f", linewidth=2.4, label="Target trajectory")
    (line_robot,) = ax.plot([], [], color="#2ca9ff", linewidth=2.8, label="Robot trajectory")

    target_patch = Circle(
        (p_ref[0, 0], p_ref[0, 1]),
        radius=cfg.target_radius_vis,
        facecolor="#f4c757",
        edgecolor="#a88523",
        linewidth=1.2,
        zorder=5,
    )
    robot_patch = Circle(
        (p[0, 0], p[0, 1]),
        radius=cfg.robot_radius_vis,
        facecolor="#1f88d8",
        edgecolor="#0f4f80",
        linewidth=1.2,
        zorder=6,
        alpha=0.5
    )
    # Dashed ring for computed/safety radius (larger than visible radius).
    robot_safety_ring = Circle(
        (p[0, 0], p[0, 1]),
        radius=cfg.robot_radius_calc,
        fill=False,
        edgecolor="#1f88d8",
        linestyle=(0, (3.0, 3.0)),
        linewidth=1.1,
        alpha=0.5,
        zorder=5,
    )
    ax.add_patch(target_patch)
    ax.add_patch(robot_safety_ring)
    ax.add_patch(robot_patch)

    speed_arrow = FancyArrowPatch(
        (0.0, 0.0),
        (0.0, 0.0),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2.2,
        color="black",
    )
    ax.add_patch(speed_arrow)

    speed_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12)

    x_pad = 8.0
    y_pad = 8.0
    ax.set_xlim(min(np.min(p[:, 0]), np.min(ellipse_x)) - x_pad, max(np.max(p[:, 0]), np.max(ellipse_x)) + x_pad)
    ax.set_ylim(min(np.min(p[:, 1]), np.min(ellipse_y)) - y_pad, max(np.max(p[:, 1]), np.max(ellipse_y)) + y_pad)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Lyapunov Tracking with Obstacle Avoidance (2 Cycles)")
    ax.legend(loc="upper right", frameon=True)

    frames = np.arange(0, len(t), frame_step, dtype=int)
    if frames[-1] != len(t) - 1:
        frames = np.append(frames, len(t) - 1)

    last_dir = {"vec": np.array([1.0, 0.0], dtype=float)}

    def update(idx: int):
        k = int(frames[idx])
        line_ref.set_data(p_ref[: k + 1, 0], p_ref[: k + 1, 1])
        line_robot.set_data(p[: k + 1, 0], p[: k + 1, 1])

        target_patch.center = (p_ref[k, 0], p_ref[k, 1])
        robot_patch.center = (p[k, 0], p[k, 1])
        robot_safety_ring.center = (p[k, 0], p[k, 1])

        v_vec = v[k]
        v_norm = float(np.linalg.norm(v_vec))
        if v_norm > 1e-6:
            last_dir["vec"] = v_vec / v_norm

        # Make arrow length strongly speed-dependent (nonlinear scaling).
        arrow_len = 1.1 + 0.75 * (v_norm**1.60)
        arrow_len = float(np.clip(arrow_len, 1.1, 16.0))
        p0 = p[k]
        p1 = p0 + arrow_len * last_dir["vec"]
        speed_arrow.set_positions((p0[0], p0[1]), (p1[0], p1[1]))
        speed_arrow.set_mutation_scale(13.0 + 2.0 * v_norm)

        speed_text.set_text(f"t = {t[k]:.2f} s   |v| = {speed[k]:.3f}")
        return (
            line_ref,
            line_robot,
            target_patch,
            robot_patch,
            robot_safety_ring,
            speed_arrow,
            speed_text,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=30,
        blit=True,
        repeat=True,
    )
    return fig, ani


def make_metrics_animation(
    data: dict[str, np.ndarray],
    frame_step: int = 3,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    t = data["t"]
    speed = data["speed"]
    accel = data["accel"]
    err = data["err"]

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.8), sharex=True)
    fig.canvas.manager.set_window_title("Robot Metrics")
    fig.suptitle("Animated Robot Metrics")

    colors = ["#1f88d8", "#d64f4f", "#2f9d67"]
    y_data = [speed, accel, err]
    y_labels = ["Speed |v|", "Acceleration |u|", "Tracking Error ||p - p_ref||"]

    lines = []
    dots = []
    for ax, y, c, label in zip(axes, y_data, colors, y_labels):
        (line,) = ax.plot([], [], color=c, linewidth=2.3)
        (dot,) = ax.plot([], [], marker="o", color=c)
        ax.set_ylabel(label)
        y_max = max(1e-3, float(np.max(y)) * 1.12)
        ax.set_ylim(0.0, y_max)
        ax.grid(alpha=0.24, linewidth=0.7)
        lines.append(line)
        dots.append(dot)

    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(float(t[0]), float(t[-1]))
    time_text = axes[0].text(0.02, 0.88, "", transform=axes[0].transAxes, fontsize=11)

    frames = np.arange(0, len(t), frame_step, dtype=int)
    if frames[-1] != len(t) - 1:
        frames = np.append(frames, len(t) - 1)

    def update(idx: int):
        k = int(frames[idx])
        tk = t[: k + 1]
        for line, dot, y in zip(lines, dots, y_data):
            line.set_data(tk, y[: k + 1])
            dot.set_data([t[k]], [y[k]])
        time_text.set_text(f"t = {t[k]:.2f} s")
        return (*lines, *dots, time_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=30,
        blit=True,
        repeat=True,
    )
    return fig, ani


def save_animation(ani: animation.FuncAnimation, path: Path, fps: int = 30) -> None:
    """Save animation using Pillow writer for broad compatibility."""
    ext = path.suffix.lower()
    if ext not in {".gif", ".mp4"}:
        raise ValueError("Animation path must end with .gif or .mp4")
    if ext == ".gif":
        ani.save(path, writer=animation.PillowWriter(fps=fps))
    else:
        # Requires ffmpeg to be installed.
        ani.save(path, writer=animation.FFMpegWriter(fps=fps))


def print_summary(data: dict[str, np.ndarray], cfg: SimConfig, seed: int | None) -> None:
    err = data["err"]
    speed = data["speed"]
    accel = data["accel"]
    tgt_speed = np.linalg.norm(data["v_ref"], axis=1)

    print("Scenario")
    print(f"  Seed                 : {seed if seed is not None else 'random'}")
    print(f"  Target start theta   : {cfg.theta0:.4f} rad")
    print(f"  Target omega         : {cfg.omega:.4f} rad/s")
    print(f"  Sim horizon          : {cfg.t_final:.2f} s ({cfg.num_cycles:.1f} cycles)")
    print(f"  Robot start          : ({cfg.p0[0]:.3f}, {cfg.p0[1]:.3f})")
    print(f"  Robot radius (calc)  : {cfg.robot_radius_calc:.3f}")
    print(f"  Robot radius (vis)   : {cfg.robot_radius_vis:.3f}")
    block_eff = cfg.blocking_gap - 2.0 * cfg.robot_radius_calc
    pass_eff = cfg.passable_gap - 2.0 * cfg.robot_radius_calc
    print(f"  Block pair gap       : {cfg.blocking_gap:.3f} (surface)")
    print(f"  Pass pair gap        : {cfg.passable_gap:.3f} (surface)")
    print(f"  Block pair gap       : {block_eff:.3f} (effective for calc body)")
    print(f"  Pass pair gap        : {pass_eff:.3f} (effective for calc body)")
    print()
    print("Simulation summary")
    print(f"  Final tracking error : {err[-1]:.4f}")
    print(f"  Mean tracking error  : {np.mean(err):.4f}")
    print(f"  Max tracking error   : {np.max(err):.4f}")
    print(f"  Max robot speed      : {np.max(speed):.4f}")
    print(f"  Mean target speed    : {np.mean(tgt_speed):.4f}")
    print(f"  Max acceleration     : {np.max(accel):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lyapunov-based tracking and obstacle-avoidance animation demo."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible scenario generation.",
    )
    parser.add_argument(
        "--save-traj",
        type=Path,
        default=None,
        help="Optional output path (.gif/.mp4) for trajectory animation.",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        default=None,
        help="Optional output path (.gif/.mp4) for metrics animation.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows (useful when only saving).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = SimConfig()
    randomize_scenario(cfg, rng)
    data = run_simulation(cfg)
    print_summary(data, cfg, args.seed)

    fig_traj, ani_traj = make_trajectory_animation(data, cfg)
    fig_metrics, ani_metrics = make_metrics_animation(data)

    if args.save_traj is not None:
        save_animation(ani_traj, args.save_traj)
        print(f"Saved trajectory animation to: {args.save_traj}")

    if args.save_metrics is not None:
        save_animation(ani_metrics, args.save_metrics)
        print(f"Saved metrics animation to: {args.save_metrics}")

    if args.no_show:
        plt.close(fig_traj)
        plt.close(fig_metrics)
    else:
        plt.show()


if __name__ == "__main__":
    main()
