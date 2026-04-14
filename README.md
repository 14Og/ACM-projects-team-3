# Lyapunov & Energy-Based Control: Safe Ellipse Tracking

> Advanced Control Methods course project by **Team 3**.
> Topic: *Lyapunov and Energy-Based Control* — planar double-integrator
> tracking a moving target on an ellipse, with obstacle avoidance.

The final controller is a **Control Lyapunov Function (CLF)** combined
with a **Higher-Order Control Barrier Function (HOCBF)** safety filter,
solved as a small **slack-relaxed quadratic program** at every step.
The project went through several controller iterations; this README
also documents the design history and the mathematical issues we
encountered, so the trade-offs we made are visible.

For full mathematical details — system model, CLF construction,
Lyapunov derivative, HOCBF derivation, QP formulation, and
guarantees — see [`stability.md`](stability.md).

---

## Notation

We follow the course notation.

- $s \in \mathbb{R}^n$ — system state.
- $a \in \mathbb{R}^m$ — control action (input).
- $\pi(s)$ — controller / policy.
- $\dot s = P(s, a)$ — plant dynamics (continuous, deterministic).
- $L(s)$ — Lyapunov function candidate.
- $\kappa_{\text{low}}, \kappa_{\text{up}}, \kappa_{\text{dec}} \in \mathcal{K}_\infty$ — class-$\mathcal{K}_\infty$ comparison functions.
- $\beta \in \mathcal{KL}$ — class-$\mathcal{KL}$ function used for asymptotic-stability bounds $\|s(t)\| \le \beta(\|s_0\|, t)$.
- $\mathbb{S}_{\text{safe}} \subset \mathbb{S}$ — safe state set; $B(s)$ — barrier certificate.

For this project the state is

$$
s \;=\; \begin{bmatrix} p \\ v \end{bmatrix} \in \mathbb{R}^4,
\qquad p,\,v \in \mathbb{R}^2,
$$

the action is $a = u \in \mathbb{R}^2$, and the plant is the planar
double integrator

$$
\dot p = v, \qquad \dot v = u.
$$

This is a deterministic, continuous-time, **non-autonomous** control
system (the reference is time-varying), satisfying the course's
definition $\dot s = P(s, a, t)$ once we write $P$ around the moving
reference $p_r(t)$.

---

## Problem statement

The reference trajectory is the ellipse

$$
p_r(t) \;=\;
\begin{bmatrix}
c_x + a\,\cos(\theta_0 + \omega t) \\
c_y + b\,\sin(\theta_0 + \omega t)
\end{bmatrix},
$$

with derivatives $v_r = \dot p_r$ and $a_r = \ddot p_r$.

Tracking errors:

$$
e_p \;=\; p - p_r,
\qquad
e_v \;=\; v - v_r.
$$

The plant must asymptotically track the reference,

$$
\|e_p(t)\| + \|e_v(t)\| \;\xrightarrow[t \to \infty]{}\; 0,
$$

while remaining outside a finite collection of circular obstacles
$\{(c_i, r_i)\}_{i=1}^{N_{\text{obs}}}$ and respecting the box control
bound $|u_j| \le u_{\max}$.

---

## Demo

### Pure Lyapunov tracking, no obstacles

Closed loop with the nominal CLF controller and reference feed-forward;
$L_{\text{tot}}$ in the bottom-right panel descends monotonically.

![Pure Lyapunov tracking](assets/gif/lyapunov_stability.gif)

### CLF + HOCBF safety filter, four obstacles

The QP overrides the nominal control only where the safety constraint
is active.

![CLF + HOCBF QP](assets/gif/clf_cbf_qp.gif)

### Constrained $u_{\max}$, feasible

Tight box bound on $u$ with the QP still strictly feasible
($s^\star = 0$).

![Constrained, feasible](assets/gif/constrained.gif)

### Constrained $u_{\max}$, stalled

Same scenario with $u_{\max}$ too small to satisfy the CBF strictly: the
slack $s^\star > 0$ keeps the QP feasible but the plant gets blocked
near the obstacle.

![Constrained, stalled](assets/gif/constrained_stalled.gif)

---

## Final controller

The implementation is in
[`lyapunov_apf/controller.py`](lyapunov_apf/controller.py).

### 1. CLF nominal control

Pick the Lyapunov function candidate

$$
L(e_p, e_v) \;=\; \tfrac{1}{2}\,k_p\,\|e_p\|^2 \;+\; \tfrac{1}{2}\,\|e_v\|^2.
$$

Bounds with $\kappa_{\text{low}}, \kappa_{\text{up}} \in \mathcal{K}_\infty$ are immediate
($L$ is a positive-definite quadratic form on the error coordinates).
Choose the nominal action

$$
u_{\text{des}} \;=\; a_r \;-\; k_p\,e_p \;-\; k_v\,e_v.
$$

The error dynamics become autonomous,

$$
\dot e_p = e_v,
\qquad
\dot e_v = -\,k_p\,e_p - k_v\,e_v,
$$

and the Lyapunov derivative is

$$
\dot L \;=\; -\,k_v\,\|e_v\|^2 \;\le\; 0.
$$

Applying LaSalle's invariance principle to the largest invariant subset
of $\{\dot L = 0\} = \{e_v = 0\}$ — on which $\dot e_v = 0$ forces
$e_p = 0$ — gives global asymptotic stability of $(e_p, e_v) = 0$, i.e.
the *asymptotic stability* characterisation $\|s(t)\| \le \beta(\|s_0\|, t)$
with $\beta \in \mathcal{KL}$.

### 2. HOCBF safety constraint

Per obstacle $i$, define the per-obstacle safety radius

$$
r_{s,i} \;=\; r_{\text{obs},i} \;+\; r_{\text{plant}} \;+\; \delta_{\text{safe}}.
$$

Take the barrier function

$$
B_i(p) \;=\; r_{s,i}^2 \;-\; \|p - c_i\|^2,
$$

so the safe set is $\mathbb{S}_{\text{safe},i} = \{ s : B_i(p) \le 0 \}$.
The implementation works with $h_i := -B_i \ge 0$ (the sign convention
used in stability.md):

$$
h_i(p) = \|p - c_i\|^2 - r_{s,i}^2,
\qquad h_i \ge 0 \;\Leftrightarrow\; s \in \mathbb{S}_{\text{safe},i}.
$$

Because $h_i$ has relative degree 2 with respect to $u$, we use the
HOCBF condition

$$
\ddot h_i + \alpha_1 \,\dot h_i + \alpha_2 \, h_i \;\ge\; 0,
\qquad
s^2 + \alpha_1 s + \alpha_2 \text{ Hurwitz},
$$

which is **linear** in $u$:

$$
2\,(p - c_i)^{\!\top} u
\;\ge\;
-2\,\|v\|^2 \;-\; 2\alpha_1 (p - c_i)^{\!\top} v \;-\; \alpha_2\, h_i.
$$

### 3. Slack-relaxed CLF–CBF–QP

The final controller is the convex program

$$
\min_{u,\;s_{\text{slk}}}\;
\tfrac{1}{2}\,\|u - u_{\text{des}}\|^2 \;+\; \rho\, s_{\text{slk}}
$$

subject to

$$
A\,u + s_{\text{slk}}\,\mathbf{1} \;\ge\; b,
\qquad
|u_j| \le u_{\max},
\qquad
s_{\text{slk}} \ge 0,
$$

where each row of $(A, b)$ is one HOCBF constraint, and $\rho \gg 0$
ensures the slack is used only when no strictly safe control exists.
We solve with `scipy.optimize.minimize` using `LinearConstraint` and
`Bounds`, with an analytic objective gradient.

### Guarantees

- **Forward invariance** of $\mathbb{S}_{\text{safe}}$ while $s_{\text{slk}}^\star = 0$: $h_i(p(t)) \ge 0$ for all $i$.
- **Asymptotic tracking** $e_p, e_v \to 0$ when the CBF is inactive (no override): $u^\star = u_{\text{des}}$ and $\dot L \le 0$, so the existence of $\beta \in \mathcal{KL}$ with $\|s(t)\| \le \beta(\|s_0\|, t)$ follows.
- **Practical / ultimate boundedness** when the CBF is active: the filter modifies $u$ away from $u_{\text{des}}$, monotone descent of $L$ is no longer guaranteed, and only a vicinity of the origin can be reached (analogous to the sample-and-hold practical-stability result in the course notes).
- **Graceful degradation** when no safe control exists ($s_{\text{slk}}^\star > 0$): the QP remains feasible, the result is best-effort, and safety must be restored by control authority $u_{\max}$ and parameter choice $\alpha_1, \alpha_2, \delta_{\text{safe}}$.

---

## Development history

### Stage 1 — APF (artificial potential field)

Repulsive potential

$$
L_{\text{rep},i} \;=\; \tfrac{1}{2}\,k_{\text{rep}}\,\Bigl(\tfrac{1}{d_{\text{eff},i}} - \tfrac{1}{I_{\text{eff}}}\Bigr)^2
$$

added to the attractive controller
$u = -k_p e_p - k_v e_v - \nabla L_{\text{rep}}$.

We abandoned this because:

- *Local minima* (Koren–Borenstein 1991): repulsive and attractive forces can cancel away from the goal.
- The candidate $L = \tfrac{1}{2} k_p \|e_p\|^2 + L_{\text{rep}} + \tfrac{1}{2}\|v\|^2$ gives only *local* asymptotic stability, and only *uniform ultimate boundedness* for moving targets.
- Heuristic add-ons (speed shaping, "catch" boost, gain inflation near obstacles) restored visual performance but broke the Lyapunov argument.

### Stage 2 — Pure energy-based PBC + reference feed-forward

Removing all heuristics leaves the textbook passivity-based law

$$
u \;=\; a_r \;-\; k_p\,e_p \;-\; k_v\,e_v.
$$

With the **error-coordinate** Lyapunov function $L = \tfrac{1}{2} k_p \|e_p\|^2 + \tfrac{1}{2}\|e_v\|^2$
the closed loop becomes autonomous and $L$ is monotonically
non-increasing along every trajectory, $\dot L = -k_v\|e_v\|^2 \le 0$.
LaSalle gives global asymptotic tracking of the moving target —
provided there are no obstacles. This is what the **Pure Lyapunov
tracking** demo above shows.

### Stage 3 — CLF + HOCBF as a QP

Adding obstacles back, we wanted formal *safety* rather than the
heuristic repulsion of stage 1. The HOCBF approach treats safety as a
*hard constraint* and embeds it in a per-step QP. The CLF nominal
control of stage 2 becomes the QP's reference, and the safety filter
overrides it only when necessary. This is the controller currently in
[`lyapunov_apf/controller.py`](lyapunov_apf/controller.py).

---

## Mathematical issues we hit (and how we resolved them)

These are the non-obvious traps we encountered. They are documented
here because they shaped the final implementation.

1. **$L_{\text{kin}} = \tfrac{1}{2}\|v\|^2$ is not monotone for moving targets.**
   The "obvious" candidate $L_{\text{att}} + L_{\text{rep}} + \tfrac{1}{2}\|v\|^2$
   only decreases when the target is *static*. For a moving target $v$
   tracks $v_r \neq 0$, so $L$ cannot reach zero. The fix is to use the
   **error-coordinate** kinetic energy $\tfrac{1}{2}\|e_v\|^2$, which makes
   $L$ a genuine Lyapunov function for the tracking problem.

2. **Asymptotic tracking of a moving target needs reference feed-forward.**
   Without $a_r$ the error dynamics are non-autonomous and the best
   guarantee is uniform ultimate boundedness. Adding $a_r$ cancels the
   drift term and produces the autonomous error system

   $$
   \ddot e_p + k_v\,\dot e_p + k_p\,e_p \;=\; 0,
   $$

   which gives global asymptotic stability via LaSalle. Toggle with
   `--feedforward`.

3. **APF local minima are a structural problem, not a tuning problem.**
   No choice of $k_{\text{rep}}, k_p, I$ removes them on non-convex
   obstacle layouts. CBF avoids this entirely: safety becomes a
   *constraint* on $u$, not a counter-force, so it never cancels the
   tracking force.

4. **CLF–CBF–QP guarantees split.** When the CBF is *inactive*,
   $\dot L \le 0$ and asymptotic tracking holds; when *active*, the
   filter can break monotone descent of $L$ and only practical
   stability / ultimate boundedness can be claimed. The
   `stability.md` *Guarantees* section makes this split explicit
   (early drafts overclaimed).

5. **HOCBF coefficients must give a Hurwitz polynomial.** The
   condition $\ddot h + \alpha_1 \dot h + \alpha_2 h \ge 0$ enforces
   forward invariance of $\{h \ge 0\}$ only if
   $s^2 + \alpha_1 s + \alpha_2$ has roots in the open LHP. We use
   $\alpha_1 = 2,\ \alpha_2 = 1$ (double pole at $s = -1$).

6. **Per-obstacle safety radius.** Using a single global $r_{\text{safe}}$
   discards each obstacle's individual radius. We compute
   $r_{s,i} = r_{\text{obs},i} + r_{\text{plant}} + \delta_{\text{safe}}$
   per obstacle, so the geometry is honest.

7. **QP infeasibility can occur** when the control authority $u_{\max}$ is
   tight relative to the required avoidance acceleration. We handle
   this with a single shared **slack variable** $s_{\text{slk}} \ge 0$
   penalised by $\rho$ in the objective. With a strictly safe control
   available, $s_{\text{slk}}^\star = 0$ and all guarantees hold;
   otherwise the slack absorbs infeasibility and the controller does
   its best instead of crashing. The "constrained, stalled" demo above
   shows a regime where the slack is occasionally non-zero.

8. **Box vs ball control bounds.** The stability analysis specifies a
   box $|u_j| \le u_{\max}$ (linear in $u$, keeps the program a true
   QP). An earlier implementation enforced a ball $\|u\| \le u_{\max}$,
   which is tighter but quadratic in $u$ and forced a nonlinear
   program. Box bounds are now used everywhere.

---

## Repository layout

```
lyapunov_apf/
├── __init__.py
├── config.py          # APFConfig, EnvConfig, SimConfig, EpisodeState
├── controller.py      # CLFCBFController (CLF nominal + HOCBF-QP filter)
├── plant.py           # 2D double integrator with RK4 step
├── simulation.py      # Episode randomisation + closed-loop rollout
└── visualization.py   # Combined trajectory + 5-panel metrics animation
main.py                # Entry point with episode loop and key handlers
stability.md           # Mathematical analysis (CLF, HOCBF, QP, guarantees)
README.md              # This file
```

---

## Running

```
uv sync
uv run python main.py
```

Useful flags:

| Flag | Effect |
|---|---|
| `--seed <int>` | Reproducible scenario generation. |
| `--feedforward` / `--no-feedforward` | Enable / disable the $a_r$ term in $u_{\text{des}}$. |
| `--constrain-control` / `--no-constrain-control` | Enable / disable the box bound $\lvert u_j\rvert \le u_{\max}$. |

Window controls:

- **ENTER** — start the next episode (re-randomises obstacles and plant start).
- **q** — quit.

To compare the no-obstacle, monotone-$L$ regime against the obstacle
regime, run

```
uv run python main.py --feedforward
```

with `obstacle_bases` in `config.py` empty vs populated.

---
