# Project Brief: Trajectory Tracking with APF
**Lyapunov-Based Control | Advanced Control Methods**

---

## 1. Project Overview

Control a 2D point mass (double integrator) to navigate from a start position to a goal while avoiding static circular obstacles, using Artificial Potential Fields (APF) as the Lyapunov-based control law. This maps cleanly onto the energy-based control topic in the course.

---

## 2. System: 2D Double Integrator

**State vector:** `q = [x, y, x_dot, y_dot]` in R^4

- `p = (x, y)` — position
- `v = (x_dot, y_dot)` — velocity

**Dynamics (Newton, unit mass):**

```
p_dot = v
v_dot = u
```

where `u = (u_x, u_y)` is the control force input (2D vector). No friction/damping is present in the plant — the controller actively commands a braking force `-k_d * v` to dissipate kinetic energy near the goal.

---

## 3. Control Method: Artificial Potential Fields (APF)

The total potential function is a sum of an attractive term (pulls toward goal) and repulsive terms (push away from each obstacle):

```
V(p) = V_att(p) + sum_i V_rep_i(p)
```

### Attractive Potential

```
V_att = (1/2) * k_att * ||p - p*||^2
grad V_att = k_att * (p - p*)
```

where `p*` is the goal position and `k_att > 0` is a tunable gain.

### Repulsive Potential (per obstacle i at position o_i)

```
rho_i = ||p - o_i||   (distance to obstacle center)

V_rep_i = (1/2) * k_rep * (1/rho_i - 1/rho_0)^2   if rho_i <= rho_0
V_rep_i = 0                                          if rho_i >  rho_0
```

where `rho_0` is the obstacle influence radius and `k_rep > 0`.

### Full Control Law (with velocity damping)

```
u = -grad V_att(p) - grad V_rep(p) - k_d * v
```

> **Note:** The damping term `-k_d * v` is essential. Without it the double integrator will oscillate around the goal indefinitely since the plant has no natural dissipation.

---

## 4. Lyapunov Function and Stability Argument

Define the total energy-like Lyapunov candidate for the full dynamic system:

```
V_total(p, v) = V(p) + (1/2) * ||v||^2
             = V_att(p) + V_rep(p) + (1/2) * ||v||^2
```

**Time derivative along trajectories:**

```
V_total_dot = grad_p V . v  +  v . u
            = grad_p V . v  +  v . (-grad_p V - k_d * v)
            = -k_d * ||v||^2  <= 0
```

`V_total_dot = 0` only when `v = 0`. By LaSalle's invariance principle this implies convergence to the goal in obstacle-free or well-separated obstacle scenarios.

**Key deliverable:** plot `V_total(t)` over time — it must be monotonically non-increasing, which is your visual proof of the stability argument.

---

## 5. Known Limitation: Local Minima

APF can trap the agent in local minima — points where `grad V_att` and `grad V_rep` cancel. This happens most commonly when the goal is directly behind an obstacle. This **must** be discussed in the README and video as an honest limitation. Demonstrate at least one such failure case in experiments.

---

## 6. Required Code Architecture

Per project requirements, strictly separate into the following files. Do NOT put everything in one file or notebook.

```
repo_root/
├── README.md
├── main.py                      # Entry point: load config, run, save outputs
├── uv.lock                      # Dependency lockfile
├── pyproject.toml               # Project metadata and dependencies
├── LICENSE
└── lyapunov_apf/                # Project 1 package
    ├── config.py                # All parameters as Python dataclasses
    ├── system.py                # PointMass2D: state, dynamics step (Euler/RK4)
    ├── controller.py            # APFController: compute_force(state, goal, obstacles)
    ├── simulation.py            # Run loop: integrate dynamics, log state + V_total
    └── visualization.py         # Plots + animation — no dynamics logic here
stubs/                           # All media for the report
    ├── figures/                 # Exported .png plots
    └── animations/              # Exported .gif or .mp4
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `lyapunov_apf/system.py` | `PointMass2D` class: holds state `[x, y, vx, vy]`, implements `step(u, dt)` |
| `lyapunov_apf/controller.py` | `APFController` class: computes `u` from current state, goal, obstacle list |
| `lyapunov_apf/simulation.py` | Time loop: calls system step, logs history, returns trajectory data |
| `lyapunov_apf/visualization.py` | All plotting and animation — no dynamics logic here |
| `lyapunov_apf/config.py` | Dataclasses for all parameters: `SimConfig`, `APFConfig`, `EnvConfig` |
| `main.py` | Wires everything together, instantiates configs, runs sim + visualization |
| `stubs/` | All figures and animations referenced in README and video |

### Example `config.py` structure

```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class APFConfig:
    k_att: float = 1.0        # Attractive gain
    k_rep: float = 100.0      # Repulsive gain
    rho_0: float = 5.0        # Obstacle influence radius
    k_d: float = 2.0          # Velocity damping gain
    u_max: float = float('inf')  # Control force limit (inf = unconstrained)

@dataclass
class SimConfig:
    dt: float = 0.01          # Integration time step (s)
    T: float = 30.0           # Max simulation duration (s)
    goal_tol: float = 0.1     # Stop when ||p - p*|| < goal_tol

@dataclass
class EnvConfig:
    start: Tuple[float, float] = (0.0, 0.0)
    goal:  Tuple[float, float] = (40.0, 40.0)
    obstacles: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (20.0, 18.0, 3.0),   # (x, y, radius)
            (25.0, 27.0, 2.5),
            (20.0, 38.0, 2.0),
        ]
    )
```

---

## 7. Required Results & Artifacts

| Output | Description |
|---|---|
| 2D trajectory plot | Agent path, obstacles (circles), goal and start marked clearly |
| `V_total(t)` plot | Must be non-increasing — proves the stability argument visually |
| Control signal plot | `\|\|u(t)\|\|` or `(u_x(t), u_y(t))` over time |
| Animation (.gif/.mp4) | Agent moving through field, current position + trail visible |
| Local minima demo | One scenario where APF fails — agent gets trapped |

---

## 8. Suggested Initial Parameters

| Parameter | Symbol | Suggested Value | Role |
|---|---|---|---|
| Attractive gain | `k_att` | `1.0` | Strength of pull toward goal |
| Repulsive gain | `k_rep` | `100.0` | Strength of obstacle repulsion |
| Influence radius | `rho_0` | `5.0` (world units) | Obstacle detection range |
| Damping gain | `k_d` | `2.0` | Velocity dissipation — tune to remove oscillation |
| Time step | `dt` | `0.01` s | Euler integration step |
| Sim duration | `T` | `30.0` s | Max simulation time |

---

## 9. README Checklist (from project requirements)

- [ ] Problem definition: trajectory tracking + collision avoidance for 2D point mass
- [ ] System description: state variables `(x, y, x_dot, y_dot)`, input `u`, dynamics equations
- [ ] Mathematical spec: define ALL symbols — `p`, `v`, `u`, `V_att`, `V_rep`, `rho`, `k_att`, `k_rep`, `k_d`, `rho_0`
- [ ] Method description: APF derivation, gradient computation, Lyapunov argument with `V_total`
- [ ] Algorithm listing: pseudocode or numbered pipeline of the control loop
- [ ] Experimental setup: initial conditions, goal position, obstacle positions/radii, all parameter values
- [ ] Reproducibility: exact run commands, `requirements.txt`, description of all outputs
- [ ] Results summary: what works, what fails (local minima), interpretation

---

*Generated from conversation with Claude | Advanced Control Methods — Skoltech | April 2026*