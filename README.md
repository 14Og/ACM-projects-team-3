# Project 3: Backstepping and Adaptive Backstepping for a 3-DOF Planar Manipulator

## Quick Visual Summary

Nominal scenario (`backstepping_full`):

![Nominal backstepping full](animations/comparison/nominal/backstepping_full.gif)

The full-model backstepping controller tracks the reference ellipse almost perfectly when model parameters match the plant.

Payload scenario (`adaptive_simp`):

![Payload adaptive simplified](animations/comparison/payload/adaptive_simp.gif)

Adaptive simplified backstepping preserves useful tracking when third-link mass/inertia are tripled in the real plant.

## 1. Problem Definition

We study trajectory tracking for a fully actuated 3-link planar manipulator with joint dynamics

$$M(q)\ddot q + C(q,\dot q)\dot q + G(q) + D\dot q = \tau.$$

Goal: track a moving elliptical end-effector target while comparing four controllers under:
- `nominal`: controller model matches plant masses.
- `payload`: true third-link mass/inertia is tripled while controllers keep nominal masses.

Controllers compared:
- `backstepping_full`
- `backstepping_simp`
- `adaptive_simp`
- `adaptive` (Project-2 Slotine-Li baseline, intentionally model-mismatched here)

## 2. System Description

- Manipulator: 3R planar arm, link lengths `[90, 70, 40]` px.
- State: $q\in\mathbb{R}^3$, $\dot q\in\mathbb{R}^3$.
- Input: joint torque $\tau\in\mathbb{R}^3$, clipped to `[80, 60, 40]`.
- Simulation: $dt=0.01$ s, horizon $12$ s, RK4 integration in plant model.
- Reference: moving ellipse (target frequency $\omega=0.7$).
- Payload perturbation: third-link mass/inertia multiplied by `payload_multiplier = 3.0`.

All numeric settings are in `configs/default.json` (and mirrored in `default_config()` in `src/config.py`).

## 3. Controllers (Methods) Description

Define backstepping coordinates:

$$z_1=q-q_d,\qquad \alpha=\dot q_d-K_1 z_1,\qquad z_2=\dot q-\alpha.$$

### 3.1 Full Backstepping (`backstepping_full`)

$$\tau = M(q)(\dot\alpha-z_1-K_2z_2)+C(q,\dot q)\dot q+G(q).$$

Uses full model feedforward and feedback stabilization in $(z_1,z_2)$.

### 3.2 Simplified Backstepping (`backstepping_simp`)

$$\tau = M_{\mathrm{diag}}(q)(\dot\alpha-z_1-K_2z_2)+G(q).$$

Neglects Coriolis term and off-diagonal inertia coupling.

### 3.3 Adaptive Simplified Backstepping (`adaptive_simp`)

Uses $G(q,m)=Y_g(q)m$ and online mass adaptation:

$$\dot{\hat m}=-\Gamma_mY_g(q)^\top z_2.$$

Implementation also adapts damping and applies projection bounds to estimates.

### 3.4 Slotine-Li Baseline (`adaptive`)

Legacy adaptive law from Project 2 for diagonal dynamics; included as a negative baseline for model-class mismatch against full manipulator dynamics.

## 4. Algorithm

For each `(scenario, controller)` pair:

1. Build plant dynamics (`nominal` or `payload`) and initialize controller state.
2. At each time step, generate reference $(q_d,\dot q_d,\ddot q_d)$.
3. Compute controller torque $\tau_{raw}$ from current $(q,\dot q)$.
4. Apply torque saturation to get $\tau$.
5. Integrate plant one step and log states, errors, torques, Lyapunov metrics.
6. For adaptive controllers, update parameter estimates.
7. After rollout, export CSV summaries and generate figures/GIFs.

## 5. Experimental Setup

The experiments use shared initial conditions and controller gains from `configs/default.json`.

| Item | Value |
|---|---|
| Initial joint angles | `[0.2, 0.1, -0.2]` rad |
| Initial joint velocities | `[0, 0, 0]` rad/s |
| Simulation step / duration | `0.01 s` / `12 s` |
| Tail window (metrics) | `2 s` |
| Torque limits | `[80, 60, 40]` |
| Reference target | moving ellipse, `omega = 0.7` |
| Backstepping gains | `K1 = diag(10,10,10)`, `K2 = diag(10,10,10)` |
| Adaptive-simplified gains | `gamma_mass = [0.08, 0.08, 0.08]`, `gamma_damping = [0.5, 0.5, 0.5]` |
| Adaptive (Slotine-Li) gains | `lambda = 6.0`, `sliding_gain = [18, 16, 12]` |
| Payload scenario change | third-link mass/inertia scaled by `3.0` |

## 6. Results

### 6.1 Quantitative Summary

Headline metric: tail mean $\|z_1\|$ over last 2 seconds.

| Scenario | Controller | Tail mean $\|z_1\|$ [rad] | Tail success | Final $V_e$ | RMS torque |
|---|---|---:|---:|---:|---:|
| nominal | `backstepping_full` | **0.011** | **1.00** | **0.020** | 7.32 |
| nominal | `backstepping_simp` | **0.008** | **1.00** | **0.006** | 37.77 |
| nominal | `adaptive_simp` | 0.304 | 0.42 | 5.11 | 52.73 |
| nominal | `adaptive` | 2.264 | 0.00 | 417.5 | 104.99 |
| payload | `backstepping_full` | 1.087 | 0.00 | 98.0 | 14.70 |
| payload | `backstepping_simp` | 1.231 | 0.00 | 105.1 | 39.00 |
| payload | `adaptive_simp` | **0.196** | **0.77** | **7.01** | 50.62 |
| payload | `adaptive` | 2.110 | 0.00 | 496.0 | 104.61 |

Main observations:
- In `nominal`, both backstepping variants track very accurately.
- In `payload`, non-adaptive backstepping degrades due to model mismatch.
- `adaptive_simp` is the only controller that retains useful payload tracking.

### 6.2 Visual Results

Nominal (`backstepping_full`):

![Nominal backstepping full](animations/comparison/nominal/backstepping_full.gif)

Reference tracking is tight and smooth with low residual oscillation.

Payload (`adaptive_simp`):

![Payload adaptive simplified](animations/comparison/payload/adaptive_simp.gif)

Tracking remains stable and practically useful under strong mass mismatch.

Selected plots:

![Nominal workspace trajectories](figures/comparison/nominal/workspace_trajectories.png)

Workspace view in nominal conditions: end-effector path remains close to the desired trajectory.

![Nominal joint tracking errors](figures/comparison/nominal/joint_tracking_errors_z1.png)

Per-joint tracking errors in nominal conditions: backstepping variants converge to very small errors.

![Payload workspace trajectories](figures/comparison/payload/workspace_trajectories.png)

Workspace view in payload conditions: non-adaptive methods drift more, while adaptive simplified remains closer.

![Payload joint tracking errors](figures/comparison/payload/joint_tracking_errors_z1.png)

Per-joint tracking errors in payload conditions: `adaptive_simp` has the smallest sustained errors.

## 7. Project Structure

```text
.
├── README.md
├── pyproject.toml
├── main.py
├── configs/default.json
├── src/
│   ├── config.py
│   ├── controller.py
│   ├── simulation.py
│   ├── system.py
│   └── visualisation.py
├── figures/
├── animations/
└── data/
```

## 8. Reproducibility

Install:

```bash
uv sync
```

Run full study:

```bash
uv run main.py
```

Typical variants:

```bash
uv run main.py --controller backstepping_full --nominal-only
uv run main.py --payload
uv run main.py --controller adaptive_simp
uv run main.py --live
```

Generated artifacts are saved under:
- `figures/comparison/{nominal,payload}/`
- `animations/comparison/{nominal,payload}/`
- `data/comparison/{nominal,payload}/`

## 9. Limitations

- The README presents a compact Lyapunov sketch; full formal derivations are omitted for brevity.
- Torque saturation is simulated but not fully embedded in theoretical guarantees.
- Elliptical references are not persistently exciting, so parameter estimates need not converge to true masses.
- Noise, delay, and actuator dynamics are not modeled.

## AI Usage

AI assistance was used for drafting and consistency checks; all equations, settings, and reported metrics were cross-checked against code and generated outputs.
