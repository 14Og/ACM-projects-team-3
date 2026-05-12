# Backstepping Tracking for a 3DOF Planar Manipulator

This project demonstrates high-precision trajectory tracking for a three-link planar manipulator using a backstepping controller.

The robot dynamics are modeled as:

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau$$

where $q,\dot{q},\ddot{q}\in\mathbb{R}^3$ are joint angles, velocities, and accelerations, and $\tau$ is the vector of motor torques.

## What It Does

By default, the end-effector follows an ellipse in the XY plane:

$$p_d(t)=
\begin{bmatrix}
60 + 165\cos(0.7t)\\
45\sin(0.7t)
\end{bmatrix}$$

If the requested ellipse is too large for the manipulator, the code scales it to the nearest fully reachable ellipse while preserving the center and shape. The reachable desired path is converted to a joint-space reference $q_d(t)$ using planar 3DOF inverse kinematics.

The controller tracks $q_d(t)$ with:

$$\tau = M(q)(\dot{\alpha} - z_1 - K_2z_2) + C(q,\dot{q})\dot{q} + G(q)$$

where:

$$z_1 = q - q_d$$

$$\alpha = \dot{q}_d - K_1z_1$$

$$z_2 = \dot{q} - \alpha$$

## Lyapunov Stability

The Lyapunov candidate is:

$$V=\frac{1}{2}z_1^Tz_1+\frac{1}{2}z_2^Tz_2$$

The backstepping law gives the nominal closed-loop derivative:

$$\dot{V}=-z_1^TK_1z_1-z_2^TK_2z_2\le0$$

For positive definite gains $K_1$ and $K_2$, the tracking error is asymptotically stable in the nominal manipulator model.

## Files

- `main.py` - command-line entry point
- `config.py` - robot, scenario, and controller parameters
- `system.py` - real full-physics plant, kinematics, and dynamics helpers
- `controller.py` - `AdaptiveLyapunovController`, `BacksteppingFull`, and `BacksteppingSimplified`
- `simulation.py` - rollout loop and CSV export
- `visualisation.py` - comparison plots and animation helpers
- `ppo/physics_robot.py` and `ppo/backstepping_tracking.py` - earlier standalone full-physics backstepping demo
- `simplified_backstepping_demo.py` - additional standalone demo with simplified decentralized dynamics: constant diagonal inertia and $C(q,\dot{q})=0$

## Install

```bash
pip install -r requirements.txt
```

## Run

Headless run with plots/logs saved:

```bash
python main.py
```

Run only one controller:

```bash
python main.py --controller backstepping_full
```

Run only the unknown-payload scenario:

```bash
python main.py --payload
```

By default, `main.py` compares `adaptive`, `adaptive_simp`, `backstepping_full`, and `backstepping_simp` in both nominal and payload scenarios. In `config.py`, `SIMULATE_PAYLOAD_ERROR=True` makes the real third-link mass 3x heavier while the backstepping controllers keep the nominal mass model. `adaptive_simp` uses the simplified model `M_diag`, `C=0`, `G(q,m_hat)` and adapts link-mass estimates.

Each rollout also saves its own GIF:

```text
outputs/comparison/nominal/animations/adaptive.gif
outputs/comparison/nominal/animations/adaptive_simp.gif
outputs/comparison/nominal/animations/backstepping_full.gif
outputs/comparison/nominal/animations/backstepping_simp.gif
outputs/comparison/payload/animations/adaptive.gif
outputs/comparison/payload/animations/adaptive_simp.gif
outputs/comparison/payload/animations/backstepping_full.gif
outputs/comparison/payload/animations/backstepping_simp.gif
```

Run the additional simplified model for comparison on the same end-effector ellipse:

```bash
python simplified_backstepping_demo.py
```

This script uses `scipy.integrate.solve_ivp`, the same requested/reachable ellipse and IK reference, and the simplified model:

$$M_{diag}\ddot{q} + G(q) = \tau,\qquad C(q,\dot{q})=0$$

with the controller:

$$\tau = M_{diag}(\dot{\alpha} - z_1 - K_2z_2) + G(q)$$

It saves plots under `outputs/simplified_backstepping/`.
It also saves `simplified_tracking.gif` with the manipulator motion.

## Outputs

Results are saved under `outputs/backstepping_tracking/`:

- `joint_tracking.png` - actual $q(t)$ vs desired $q_d(t)$
- `tracking_error.png` - joint tracking error $z_1(t)$
- `tracking_error_norm.png` - $\|z_1(t)\|$
- `torques.png` - control torques $\tau(t)$
- `end_effector_trajectory.png` - requested ellipse, reachable desired path, and actual end-effector path
- `lyapunov.png` - Lyapunov function $V(t)$
- `lyapunov_derivative.png` - nominal derivative $\dot{V}(t)\le0$
- `simulation_log.npz` - NumPy log with time, states, references, torques, errors, and end-effector positions

## Tuning

Controller gains and trajectory parameters are in `BacksteppingConfig`:

```python
k1 = (10.0, 10.0, 10.0)
k2 = (10.0, 10.0, 10.0)
ellipse_center_xy = (60.0, 0.0)
ellipse_radii_xy = (165.0, 45.0)
omega = 0.7
```

Use `trajectory_mode = "joint_sine"` to switch from end-effector ellipse tracking to the analytical joint-space sine reference.

The default simulation starts with a small nonzero tracking error:

```python
initial_q_error = (0.15, -0.10, 0.08)
initial_q_dot_error = (0.0, 0.0, 0.0)
```

This makes the Lyapunov plot useful for a report: $V(t)$ starts above zero and decays. If the initial state is exactly equal to the reference, the theoretical value is $V(0)=0$ and any tiny visible peaks are only numerical residue from integration, inverse kinematics, and finite-difference derivatives.
