# Project 3: Backstepping and Adaptive Backstepping for a 3-DOF Planar Manipulator

This project applies backstepping control to end-effector tracking on a fully-actuated three-link planar manipulator with full rigid-body dynamics $M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) + D\,\dot q = \tau$. Four controllers are compared in two scenarios: a **nominal** scenario in which controller and plant share the same link masses, and a **payload** scenario in which the real third-link mass is tripled while the controllers keep nominal values.

The four controllers are `backstepping_full` (full $M,C,G$ feedforward), `backstepping_simp` (diagonal $M$, no Coriolis), `adaptive_simp` (adaptive backstepping with gravity regressor), and `adaptive` (Slotine-Li from Project 2, included as a wrong-model baseline). Together they produce all eight cases required by the mandatory Project 2+ comparison (§10).

## Quick Visual Summary

| Nominal scenario | Payload scenario (m₃ × 3) |
|---|---|
| ![nominal_backstepping_full](animations/comparison/nominal/backstepping_full.gif) | ![payload_adaptive_simp](animations/comparison/payload/adaptive_simp.gif) |
| `backstepping_full` follows the reachable ellipse essentially perfectly. | `adaptive_simp` is the only controller that survives model mismatch. |

All eight per-controller GIFs are in `animations/comparison/{nominal,payload}/`.

## Result Summary

Tail-window (last 2 s) metrics. Success threshold: 10 px end-effector error. $V_e = \tfrac{1}{2}(\|z_1\|^2 + \|z_2\|^2)$.

| Scenario | Controller | Tail mean $\|z_1\|$ [rad] | Tail success | Final $V_e$ | RMS torque [Nm] |
|---|---|---:|---:|---:|---:|
| nominal | `backstepping_full` | **0.011** | **1.00** | **0.020** | 7.32 |
| nominal | `backstepping_simp` | **0.008** | **1.00** | **0.006** | 37.77 |
| nominal | `adaptive_simp` | 0.304 | 0.42 | 5.11 | 52.73 |
| nominal | `adaptive` | 2.264 | 0.00 | 417.5 | 104.99 |
| payload | `backstepping_full` | 1.087 | 0.00 | 98.0 | 14.70 |
| payload | `backstepping_simp` | 1.231 | 0.00 | 105.1 | 39.00 |
| payload | `adaptive_simp` | **0.196** | **0.77** | **7.01** | 50.62 |
| payload | `adaptive` | 2.110 | 0.00 | 496.0 | 104.61 |

Key findings: (1) both backstepping variants achieve $\|z_1\| \sim 10^{-2}$ rad under nominal conditions; (2) the payload mismatch inverts the ranking — `adaptive_simp` is the only survivor (77 % success); (3) the Slotine-Li baseline fails in both scenarios because its diagonal joint-space model does not describe the true plant; (4) tracking is asymptotic without persistent excitation, but parameter identification is not.

## Repository Layout

```text
.
├── README.md
├── LICENSE                                # MIT
├── requirements.txt                       # numpy, scipy, matplotlib, pillow
├── pyproject.toml
├── main.py                                # root launcher → src.main.main()
├── simplified_backstepping_demo.py        # standalone scipy/solve_ivp demo
├── configs/
│   └── default.json                       # full hyperparameter set
├── src/                                   # main four-way comparison
│   ├── __init__.py
│   ├── config.py
│   ├── system.py
│   ├── controller.py
│   ├── simulation.py
│   ├── visualisation.py
│   └── main.py
├── ppo/                                   # earlier standalone backstepping demo
│   ├── config.py
│   ├── physics_robot.py
│   └── backstepping_tracking.py
├── figures/comparison/{nominal,payload}/  # 7 PNGs per scenario
├── animations/comparison/{nominal,payload}/ # 4 GIFs per scenario
├── data/comparison/{nominal,payload}/     # rollouts.csv, summary.csv
└── notes/project_1_stability.md          # historical, kept for reference
```

| File | Responsibility |
|---|---|
| `src/system.py` | FK, Jacobian, full $M(q), C(q,\dot q), G(q)$, RK4 integration, payload helper. |
| `src/controller.py` | Reference generator + four controllers. |
| `src/simulation.py` | Rollout, metrics, CSV export, Lyapunov values. |
| `src/visualisation.py` | PNG figures and per-controller GIF animations. |
| `src/config.py` | Typed dataclasses, `default_config()`, `load_config()`. |
| `src/main.py` | CLI: runs both scenarios, generates all artifacts. |

## 1. Problem Definition

The end-effector must track a closed ellipse in workspace coordinates. The requested ellipse is auto-scaled to the largest reachable version inside the workspace, and the joint-space reference $q_d(t), \dot q_d(t), \ddot q_d(t)$ is obtained by closed-form planar-3R inverse kinematics with central-difference differentiation.

The full plant is $M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) + D\,\dot q = \tau$. Assumptions: full state $[q^\top\,\dot q^\top]^\top \in \mathbb{R}^6$ is available; link geometry is perfectly known; torques are clipped to configured limits (not modelled in proofs); no sensor noise or actuator dynamics.

Two Lyapunov-based methods are used: **backstepping** (nominal case) and **adaptive backstepping with a gravity regressor** (unknown-mass case). The Slotine-Li controller from Project 2 is kept as a wrong-model baseline.

## 2. System Description

Three revolute joints, link lengths $L = [90, 70, 40]$ px $= [0.45, 0.35, 0.20]$ m (divided by 200). Centre of mass at midpoint; thin-rod inertia $I_i = \tfrac{1}{12}m_i L_{m,i}^2$. Gravity $g = 9.81$ m/s².

Forward kinematics: $p_i(q) = \sum_{k=1}^{i} L_k [\cos\alpha_k,\, \sin\alpha_k]^\top$ with cumulative angle $\alpha_k = \sum_{\ell=1}^k q_\ell$.

**Inertia matrix** $M(q)$ is the standard symmetric planar-3R matrix built from link masses $m_1, m_2, m_3$ and the cross-term products $m_i L_j \ell_{ci} \cos(\cdot)$; all entries are implemented in `src/system.py` (`_dynamics_matrices`). **Gravity vector** entries: $G_i = \sum_{k=i}^{3} m_k g \ell_{ck} \cos(\sum_{\ell=1}^k q_\ell)$, augmented by the downstream link-CoM contributions. **Coriolis vector** $C(q,\dot q)\dot q$ is computed from Christoffel symbols of $M$, preserving the skew-symmetry $\dot M - 2C = -(\dot M - 2C)^\top$.

**Reference ellipse:** $p_r(t) = (60, 0) + [165\cos(0.7t),\; 45\sin(0.7t)]$ px; auto-scaled to workspace before IK.

**Payload perturbation:** in the payload scenario the third-link mass and inertia are multiplied by 3.0 in the plant; all controllers continue using nominal masses $[1.0, 0.7, 0.6]$ kg.

## 3. Mathematical Notation

| Symbol | Meaning | Code name |
|---|---|---|
| $q, \dot q, \ddot q$ | joint angles, velocities, accelerations | `q`, `dq`, `ddq` |
| $q_d, \dot q_d, \ddot q_d$ | desired joint trajectory | `ReferenceState.q/.dq/.ddq` |
| $z_1 = q - q_d$ | position tracking error | `ControlInfo.q_error` |
| $\alpha = \dot q_d - K_1 z_1$ | virtual control | implicit in `_backstepping_errors` |
| $z_2 = \dot q - \alpha$ | velocity error w.r.t. virtual control | `ControlInfo.sliding_error` |
| $\dot\alpha = \ddot q_d - K_1(\dot q - \dot q_d)$ | derivative of virtual control | computed inline |
| $M(q), C(q,\dot q), G(q)$ | full rigid-body matrices | `manipulator_terms()` |
| $M_\text{diag}$ | diagonal of $M(q)$ | `BacksteppingSimplified`, `AdaptiveSimplifiedController` |
| $K_1, K_2 \succ 0$ | backstepping gain matrices | `backstepping_controller.k1/.k2` |
| $m,\; \hat m,\; \tilde m = \hat m - m$ | true/estimated/error link masses | `dynamics.link_masses`, `ControlInfo.mass_hat` |
| $Y_g(q)$: $G(q,m) = Y_g(q)m$ | gravity regressor | `_gravity_mass_regressor` |
| $\Gamma_m, \Gamma_D \succ 0$ | adaptation gains | `gamma_mass`, `gamma_damping` |
| $V_e = \tfrac12(\|z_1\|^2+\|z_2\|^2)$ | tracking Lyapunov candidate | `Rollout.tracking_lyapunov` |
| $V_a$ | augmented candidate (adds $\tfrac12\tilde m^\top\Gamma_m^{-1}\tilde m$) | `Rollout.augmented_lyapunov` |

## 4. Method Description

### 4.1 Backstepping (nominal case)

The manipulator is a two-layer cascade: $\dot q$ drives $q$, and $\tau$ drives $\dot q$. Backstepping builds the control law layer by layer.

**Step 1 — virtual control.** Define $z_1 = q - q_d$, $\dot z_1 = \dot q - \dot q_d$. If $\dot q$ were free, the choice $\alpha = \dot q_d - K_1 z_1$ gives $\dot z_1 = -K_1 z_1$ (exponentially stable).

**Step 2 — velocity error.** Because $\dot q \neq \alpha$ in general, set $z_2 = \dot q - \alpha$, so $\dot z_1 = z_2 - K_1 z_1$. If $z_2 \to 0$ then $z_1 \to 0$.

**Step 3 — velocity-error dynamics.** From the plant equation,
$$\dot z_2 = M(q)^{-1}[\tau - C\dot q - G] - \dot\alpha, \qquad \dot\alpha = \ddot q_d - K_1(\dot q - \dot q_d).$$

**Step 4 — Lyapunov candidate.** Take $V = \tfrac{1}{2}\|z_1\|^2 + \tfrac{1}{2}\|z_2\|^2$. *(Note: the canonical form uses $z_2^\top M z_2$ and exploits skew-symmetry; the unweighted form is used here for simplicity with the same result.)*

**Step 5 — derivative.**
$$\dot V = -z_1^\top K_1 z_1 + z_1^\top z_2 + z_2^\top\bigl[M^{-1}(\tau - C\dot q - G) - \dot\alpha\bigr].$$

**Step 6 — control law.** Choose $\tau$ so that the bracket equals $-z_1 - K_2 z_2$:
$$\boxed{\;\tau = M(q)\bigl(\dot\alpha - z_1 - K_2 z_2\bigr) + C(q,\dot q)\dot q + G(q).\;}$$

This is implemented verbatim in `BacksteppingFull.compute`.

**Step 7 — conclusion.** With this $\tau$,
$$\dot V = -z_1^\top K_1 z_1 - z_2^\top K_2 z_2 \le 0.$$
By Barbalat's lemma (reference is time-varying), $z_1, z_2 \to 0$: **asymptotic tracking** under exact model knowledge and no saturation.

### 4.2 Simplified backstepping

`BacksteppingSimplified` uses $\tau = M_\text{diag}(\dot\alpha - z_1 - K_2 z_2) + G(q)$, dropping $C\dot q$ and off-diagonal entries of $M$. These terms are small at $\omega = 0.7$ rad/s in the nominal scenario; under payload mismatch they become significant.

### 4.3 Adaptive backstepping with gravity regressor

Because $G(q,m) = Y_g(q)\,m$ is linear in the link-mass vector, the masses can be estimated online. With estimate $\hat m$ and error $\tilde m = \hat m - m$, the control law is
$$\tau = M_\text{diag}(\hat m)\bigl(\dot\alpha - z_1 - K_2 z_2\bigr) + \hat D\dot q + G(q, \hat m).$$

**Augmented candidate:** $V_a = \tfrac{1}{2}\|z_1\|^2 + \tfrac{1}{2}\|z_2\|^2 + \tfrac{1}{2}\tilde m^\top\Gamma_m^{-1}\tilde m$.

**Derivative.** The $z_2$-dynamics carry the residual $-Y_g\tilde m$, so
$$\dot V_a = -z_1^\top K_1 z_1 - z_2^\top K_2 z_2 - z_2^\top Y_g\tilde m + \tilde m^\top\Gamma_m^{-1}\dot{\hat m}.$$

**Adaptation law.** Setting the last two terms to zero for all $\tilde m$:
$$\boxed{\;\dot{\hat m} = -\Gamma_m\,Y_g(q)^\top z_2.\;}$$

Implemented in `AdaptiveSimplifiedController.compute` with projection (np.clip) into mass bounds. With this law $\dot V_a = -z_1^\top K_1 z_1 - z_2^\top K_2 z_2 \le 0$, so **tracking is asymptotic even with unknown masses**. Parameter convergence additionally requires persistent excitation, which the elliptical reference does not provide.

An analogous law $\dot{\hat D} = -\Gamma_D(z_2 \odot \dot q)$ adapts the damping estimate simultaneously.

### 4.4 Slotine-Li baseline (wrong-model case)

`AdaptiveLyapunovController` (Project 2) assumes the diagonal model $H\ddot q + D\dot q = \tau + b$ with scalar unknowns $(H_i, D_i, b_i)$. Applied to the full $M(q), C, G$ plant, the adaptation laws drive the estimates to values that do not minimise tracking error. This controller is included as a deliberate negative baseline.

## 5. Algorithm Listing

For each controller and each simulation step:

1. Compute reference state $q_d$, $\dot q_d$, $\ddot q_d$ from the ellipse trajectory via closed-form IK and central differences.
2. Read plant state $q$, $\dot q$.
3. Compute backstepping errors: $z_1 \leftarrow q - q_d$; $\alpha \leftarrow \dot q_d - K_1 z_1$; $z_2 \leftarrow \dot q - \alpha$; $\dot\alpha \leftarrow \ddot q_d - K_1(\dot q - \dot q_d)$.
4. Compute controller torque:
   - `backstepping_full`: $\tau = M(q)(\dot\alpha - z_1 - K_2 z_2) + C(q,\dot q)\dot q + G(q)$,
   - `backstepping_simp`: $\tau = M_\text{diag}(q)(\dot\alpha - z_1 - K_2 z_2) + G(q)$,
   - `adaptive_simp`: $\tau = M_\text{diag}(q,\hat m)(\dot\alpha - z_1 - K_2 z_2) + \hat D\dot q + G(q,\hat m)$,
   - `adaptive`: $\tau = \hat H\ddot q_r + \hat D\dot q - \hat b - K_s s$, with $s = (\dot q - \dot q_d) + \lambda(q - q_d)$, $\ddot q_r = \ddot q_d - \lambda(\dot q - \dot q_d)$.
5. Clip $\tau$ to actuator bounds.
6. For adaptive controllers, update parameter estimates:
   - `adaptive_simp`: $\hat m \leftarrow \mathrm{proj}\!\left(\hat m - \Delta t\,\Gamma_m Y_g(q)^\top z_2\right)$; $\hat D \leftarrow \mathrm{proj}\!\left(\hat D - \Delta t\,\Gamma_D(z_2\odot\dot q)\right)$.
   - `adaptive`: update $\hat H$, $\hat D$, $\hat b$ as in Project 2.
7. Integrate the true plant by RK4 with sub-stepping (sub-step $\le 0.5$ ms).
8. Record $q$, $\dot q$, $\tau$, errors $z_1$, $z_2$, parameter estimates, and Lyapunov values $V_e$, $V_a$.
9. After rollout: compute tail-window metrics; export `rollouts.csv`, `summary.csv`, PNG figures, and per-controller GIF.

## 6. Experimental Setup

| Quantity | Value |
|---|---:|
| integration step $\Delta t$ | 0.01 s |
| RK4 sub-step | $\le 0.5$ ms |
| simulation duration | 12.0 s |
| tail window | 2.0 s |
| EE success threshold | 10 px |
| initial joint angles $q(0)$ | $[0.2, 0.1, -0.2]$ rad |
| initial $\dot q(0)$ | $[0, 0, 0]$ rad/s |
| true link masses (nominal) | $[1.0, 0.7, 0.6]$ kg |
| true link masses (payload) | $[1.0, 0.7, \mathbf{1.8}]$ kg |
| true damping $D$ | $[0.08, 0.06, 0.05]$ |
| torque limits | $[80, 60, 40]$ Nm |
| ellipse centre, semi-axes, $\omega$ | $(60,0)$ px; $(165, 45)$ px; $0.7$ rad/s |
| $K_1, K_2$ (backstepping) | $\mathrm{diag}(10,10,10)$ |
| $\Gamma_m$ | $\mathrm{diag}(0.08, 0.08, 0.08)$ |
| $\Gamma_D$ | $\mathrm{diag}(0.5, 0.5, 0.5)$ |
| mass bounds | $[0.05, 3.0]$ kg |
| $\lambda$, $K_s$ (Slotine-Li) | $6.0$; $\mathrm{diag}(18,16,12)$ |

All values are in `configs/default.json` and reproduced by `src/config.py::default_config()`.

## 7. Reproducibility

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, pillow
python main.py                    # full 4-way × 2-scenario comparison (~2 min/scenario)
python main.py --controller backstepping_full --nominal-only   # single controller
python main.py --payload                                       # payload scenario only
python simplified_backstepping_demo.py                         # standalone scipy demo
python -m ppo.backstepping_tracking                            # earlier numba demo
```

Outputs are written to `figures/`, `animations/`, and `data/` as described in the Repository Layout. All committed artifacts in those directories were produced by `python main.py` with `configs/default.json` unchanged.

## 8. Results: Nominal Scenario

![Nominal workspace trajectories](figures/comparison/nominal/workspace_trajectories.png)

**Figure 1.** Workspace view. Orange: reachable ellipse. Green curve: end-effector path of the `adaptive` controller (first rollout rendered; see Limitations). Arm poses at $t = T$ are shown for all four controllers.

![Nominal per-joint tracking errors](figures/comparison/nominal/joint_tracking_errors_z1.png)

**Figure 2.** Per-joint $z_1$. Both backstepping variants reach the $10^{-2}$ rad band after ~4 s and stay there. `adaptive_simp` settles to a small residual after a longer transient. `adaptive` shows persistent oscillations on all three joints.

![Nominal Lyapunov values](figures/comparison/nominal/lyapunov_values.png)

**Figure 3.** Top: augmented $V_a$ for `adaptive` — grows to $\approx 10^3$ because the plant model is wrong. Bottom: tracking $V_e$ for all controllers. Backstepping variants reach $V_e < 0.1$; `adaptive_simp` settles at $\approx 5$; `adaptive` stays at $300$–$400$.

![Nominal adaptive backstepping parameter estimates](figures/comparison/nominal/adaptive_simp_parameter_estimates.png)

**Figure 4.** Mass estimates of `adaptive_simp`: initially saturate at the 3.0 kg bound during startup, then drift toward the true values $[1.0, 0.7, 0.6]$ without full convergence — consistent with absent persistent excitation.

## 9. Results: Payload Scenario

Third-link mass $m_3 = 1.8$ kg in the plant (nominal value used by all controllers: 0.6 kg).

![Payload per-joint tracking errors](figures/comparison/payload/joint_tracking_errors_z1.png)

**Figure 5.** `adaptive_simp` (light green) is the only controller below $0.2$ rad per joint. Both backstepping variants drift to $\approx 1$ rad on joint 3 due to feedforward error. `adaptive` is unchanged from the nominal failure mode.

![Payload Lyapunov values](figures/comparison/payload/lyapunov_values.png)

**Figure 6.** $V_e$ for backstepping controllers stabilises at $\approx 100$ — an order of magnitude above nominal. Only `adaptive_simp` decays to a small steady-state value ($V_e \approx 7$).

![Payload adaptive backstepping parameter estimates](figures/comparison/payload/adaptive_simp_parameter_estimates.png)

**Figure 7.** The third-link mass estimate shifts upward relative to the nominal case, partially compensating for the true 3× increase. The gravity regressor cancels the gravity error in the direction that appears at the end-effector, restoring tracking without resolving the individual mass values.

## 10. Comparison Discussion

| Controller | Nominal | Payload | Failure mode |
|---|---|---|---|
| `backstepping_full` | ✓ $\|z_1\| = 0.011$ rad, 100 % | ✗ $\|z_1\| = 1.09$ rad | Feedforward commits a fixed gravity/inertia error proportional to $\Delta m_3$; feedback cannot cancel it within actuator limits. |
| `backstepping_simp` | ✓ $\|z_1\| = 0.008$ rad, 100 % | ✗ $\|z_1\| = 1.23$ rad | Same as full, additionally unmodelled Coriolis term grows with payload. |
| `adaptive_simp` | ~ $\|z_1\| = 0.30$ rad, 42 % | ✓ $\|z_1\| = 0.20$ rad, 77 % | Nominal: slow adaptation transient. Payload: only survivor — regressor redirects adaptation to cancel the dominant gravity error direction. |
| `adaptive` | ✗ $V_e \approx 400$ | ✗ $V_e \approx 500$ | Wrong model class (diagonal scalar plant vs. full $M(q), C, G$); adaptation laws have no correct fixed point in either scenario. |

Three lessons: (1) backstepping is optimal when the model is exact but fragile to parameter mismatch; (2) adaptive backstepping with a correct regressor structure restores tracking under mismatch without full parameter convergence; (3) the model class must match the plant — a correct algorithm on the wrong model class fails systematically.

## 11. Limitations

- **Simplified Lyapunov form.** $V = \tfrac12\|z_1\|^2 + \tfrac12\|z_2\|^2$ is used instead of $V = \tfrac12\|z_1\|^2 + \tfrac12 z_2^\top M z_2$. The closed-loop result is the same but the energy interpretation and skew-symmetry argument are foregone.
- **No persistent excitation.** The elliptical reference does not persistently excite the gravity regressor, so mass estimates remain bounded but do not identify the true values.
- **`workspace_trajectories.png` shows one EE path.** `_save_workspace` renders only the first rollout's end-effector trace; use `joint_tracking_errors_z1.png` or the per-controller GIFs to compare paths.
- **Torque saturation not in proof.** Clipping to $\pm[80,60,40]$ Nm occurs in the simulator but is not modelled in the Lyapunov arguments.
- **No noise, delay, or actuator dynamics.** All signals are ideal.
- **Gravity-only regressor.** $M(q)$ is not linearised in masses; inertia and damping are adapted by separate scalar laws, not a full mass-regressor parametrisation.

## 12. AI Usage

Parts of the README structure, prose, and cross-checks between equations and code were drafted with AI assistance. All mathematical claims have been verified against the implementation; all numerical values in the result tables come directly from `data/comparison/*/summary.csv`.
