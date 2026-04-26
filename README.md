# Project 2: Adaptive Control of a 3-DOF Planar Manipulator

## Problem Definition

The project idea is taken from Team 8's repository:
https://github.com/14Og/RL-projects-team-8.

Control objective:

- move the end effector of a 3-link planar manipulator toward a moving target
- avoid moving circular obstacles
- track an obstacle-aware joint reference under unknown joint inertia and damping
- compare adaptive control against non-adaptive Lyapunov baselines

Important scope note: this implementation does not copy Team 8's code and does
not reproduce their full rigid-body `M(q), C(q, q_dot), G(q)` model. It uses a
simplified diagonal joint-space model because the adaptive-control proof from
the lecture requires dynamics that are linear in unknown constant parameters.

## System Description

The arm has three revolute joints and link lengths:

```text
L = [90, 70, 40] px
```

The base is fixed at:

```text
p_base = [400, 600] px
```

State variables:

```math
q =
\begin{bmatrix} q_1 & q_2 & q_3 \end{bmatrix}^T,
\qquad
\dot q =
\begin{bmatrix} \dot q_1 & \dot q_2 & \dot q_3 \end{bmatrix}^T.
```

Control input:

```math
\tau =
\begin{bmatrix} \tau_1 & \tau_2 & \tau_3 \end{bmatrix}^T.
```

Torque bounds:

```text
|tau| <= [22, 16, 12]
```

Forward kinematics for joint endpoint `i`:

```math
p_i(q) = p_base +
\sum_{k=1}^{i}
L_k
\begin{bmatrix}
\cos(q_1 + ... + q_k) \\
\sin(q_1 + ... + q_k)
\end{bmatrix}.
```

The end effector is `p_3(q)`.

Moving target:

```math
p_T(t) =
c_T +
\begin{bmatrix}
a_x \cos(\omega_T t) \\
a_y \sin(\omega_T t)
\end{bmatrix}.
```

Moving obstacles:

```math
c_j(t) =
c_{j,0} +
\begin{bmatrix}
A_{jx} \cos(\omega_j t + \phi_j) \\
A_{jy} \sin(\omega_j t + \phi_j)
\end{bmatrix}.
```

Each obstacle is a circle with radius `30 px`.

## Plant Dynamics

The simulated torque-level plant is:

```math
H \ddot q + D \dot q = \tau.
```

Here:

- `H = diag(H_1, H_2, H_3)` is the true joint inertia matrix
- `D = diag(D_1, D_2, D_3)` is the true viscous damping matrix
- both `H` and `D` are unknown to the adaptive controller

Default true parameters:

```text
H = [18, 10, 5]
D = [16, 9, 4]
```

This model is deliberately simpler than a full manipulator model, but it is
linear in unknown constant parameters and therefore fits the adaptive-control
theory used here.

## Obstacle-Aware Reference Generator

The controller tracks a generated joint reference `q_r(t)`. The reference
generator is a kinematic artificial-potential-field planner.

Goal velocity in task space:

```math
v_g =
k_g (p_T - p_3(q_r)) + \dot p_T.
```

The goal contribution to joint velocity uses damped least squares:

```math
\dot q_g =
J_3(q_r)^T
\left(J_3(q_r)J_3(q_r)^T + \eta^2 I\right)^{-1}
v_g.
```

For each obstacle and each controlled point on the arm, a repulsive velocity is
added when the clearance is below an influence distance. The code converts that
repulsive point velocity to joint velocity through the corresponding point
Jacobian.

The planner is heuristic. It is included to preserve the obstacle-avoidance
problem idea from Team 8. The formal adaptive proof applies to tracking the
bounded generated reference, not to global obstacle-avoidance optimality.

## Adaptive Control Theory

The YouTube lecture develops adaptive control for systems with dynamics linear
in unknown parameters:

```math
u = Y(q, \dot q, q_r, \dot q_r, \ddot q_r) \hat a - K s,
```

with filtered tracking error:

```math
e = q - q_r,
\qquad
\dot e = \dot q - \dot q_r,
\qquad
s = \dot e + \lambda e.
```

For this project:

```math
\dot q_r^* = \dot q_r - \lambda e,
\qquad
\ddot q_r^* = \ddot q_r - \lambda \dot e.
```

The adaptive torque law is:

```math
\tau =
\hat H \ddot q_r^*
+ \hat D \dot q
- K_s s.
```

Because:

```math
H \dot s =
-K_s s
+ \tilde H \ddot q_r^*
+ \tilde D \dot q,
```

where:

```math
\tilde H = \hat H - H,
\qquad
\tilde D = \hat D - D,
```

use the augmented Lyapunov candidate:

```math
V =
\frac{1}{2}s^T Hs
+ \sum_i \frac{\tilde H_i^2}{2\gamma_{H_i}}
+ \sum_i \frac{\tilde D_i^2}{2\gamma_{D_i}}.
```

The update laws are:

```math
\dot{\hat H}_i = -\gamma_{H_i} s_i \ddot q_{r,i}^*,
\qquad
\dot{\hat D}_i = -\gamma_{D_i} s_i \dot q_i.
```

Substitution gives:

```math
\dot V = -s^T K_s s <= 0.
```

Under the standard lecture assumptions of bounded reference signals, positive
inertia, no actuator saturation, and exact velocity measurement, the filtered
error `s` converges to zero. Since `s = dot e + lambda e` is a stable first
order filter, `e -> 0` and `dot e -> 0`.

Parameter convergence is not guaranteed without persistent excitation. The
results therefore focus on tracking improvement, not exact identification.

## Comparison Controllers

### Fixed Lyapunov Baseline

The fixed baseline uses the same filtered-error structure but keeps incorrect
parameters:

```math
\tau =
H_0 \ddot q_r^*
+ D_0 \dot q
- K_s s.
```

Defaults:

```text
H_0 = [1.0, 0.8, 0.5]
D_0 = [0.1, 0.1, 0.1]
```

### Plain PD Lyapunov Baseline

The second baseline is classical joint-space PD tracking:

```math
\tau = -K_p e - K_d \dot e.
```

This is a simple Lyapunov controller for joint errors, but it has no adaptive
model compensation.

## Algorithm Listing

For each simulation step:

1. Compute the moving target `p_T(t)` and moving obstacles `c_j(t)`.
2. Update the obstacle-aware joint reference `q_r, dot q_r, ddot q_r`.
3. Measure the plant state `q, dot q`.
4. Compute `e = q - q_r`, `dot e = dot q - dot q_r`.
5. Compute `s = dot e + lambda e`.
6. Compute `ddot q_r_star = ddot q_r - lambda dot e`.
7. Apply adaptive torque:
   `tau = H_hat ddot q_r_star + D_hat dot q - K_s s`.
8. Update `H_hat` and `D_hat` using the adaptive laws.
9. Clip torque to the configured actuator limits.
10. Integrate the true plant dynamics.
11. Record target error, joint error, torque, parameter estimates, and obstacle clearance.

The fixed Lyapunov baseline skips step 8. The PD baseline uses its own torque
law in step 7.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- main.py
|-- configs/
|   `-- default.json
|-- figures/
|-- animations/
`-- src/
    |-- __init__.py
    |-- config.py
    |-- system.py
    |-- controller.py
    |-- simulation.py
    |-- visualization.py
    `-- main.py
```

Separation:

- `system.py`: kinematics, obstacles, dynamics, collision clearance
- `controller.py`: reference generator and controllers
- `simulation.py`: rollout and metrics
- `visualization.py`: plots and animation only
- `main.py`: command-line orchestration

## Experimental Setup

Default setup:

| Quantity | Value |
|---|---:|
| duration | 60 s |
| integration step | 0.02 s |
| initial angles | `[pi, -0.5, 0.7]` rad |
| target threshold | 30 px |
| obstacle radius | 30 px |
| true inertia | `[18, 10, 5]` |
| true damping | `[16, 9, 4]` |
| initial inertia estimate | `[1.0, 0.8, 0.5]` |
| initial damping estimate | `[0.1, 0.1, 0.1]` |
| torque limits | `[22, 16, 12]` |

## Reproducibility

Install:

```bash
pip install -r requirements.txt
```

Run the complete experiment:

```bash
python main.py
```

Run without animation:

```bash
python main.py --no-animation
```

Run a shorter smoke test:

```bash
python main.py --duration 5 --no-plots --no-animation
```

Equivalent module entry point:

```bash
python -m src.main
```

Produced outputs:

- `figures/summary_metrics.json`
- `figures/workspace_trajectories.png`
- `figures/tracking_errors.png`
- `figures/adaptive_parameter_estimates.png`
- `figures/control_and_clearance.png`
- `animations/adaptive_manipulator.gif`

## Results Summary

Default run metrics:

| Controller | Tail mean target error | Minimum clearance | Tail success |
|---|---:|---:|---:|
| adaptive | 4.636 px | 28.686 px | 1.000 |
| fixed Lyapunov | 6.300 px | 63.028 px | 1.000 |
| plain PD | 13.507 px | 28.187 px | 1.000 |

The adaptive controller tracks the generated reference more tightly than the
fixed-parameter Lyapunov controller, and both outperform the plain PD baseline
under the chosen uncertain dynamics. All controllers avoid collision in the
default run; the comparison is primarily about tracking quality under parameter
mismatch.

The final adaptive estimates are not equal to the true physical parameters.
This is expected: the lecture theory guarantees tracking under the stated
assumptions, while parameter convergence additionally requires persistent
excitation.

## Final Artifacts

![Workspace trajectories](figures/workspace_trajectories.png)

Figure 1. End-effector paths, moving target path, obstacles at the initial
frame, and final arm configurations.

![Tracking errors](figures/tracking_errors.png)

Figure 2. End-effector target error and joint-reference tracking error. The
adaptive controller has the lowest tail target error.

![Adaptive parameter estimates](figures/adaptive_parameter_estimates.png)

Figure 3. Online inertia and damping estimates. Dashed lines mark true values.
The estimates improve tracking but do not exactly identify all parameters.

![Control and clearance](figures/control_and_clearance.png)

Figure 4. Adaptive torques and minimum link-obstacle clearance. The clearance
stays positive in the default run.

Animation:

![Adaptive manipulator animation](animations/adaptive_manipulator.gif)

## Limitations

- The dynamics are diagonal joint-space dynamics, not the full rigid-body model
  from Team 8.
- The obstacle-aware planner is heuristic APF plus damped least squares. It is
  not a formal CBF safety proof.
- The Lyapunov proof ignores actuator saturation; the implementation records
  saturation so this limitation is visible.
- No sensor noise is simulated. A real noisy system would need projection,
  leakage, dead zones, or filtering to avoid parameter drift.
