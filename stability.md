# Safe Ellipse Tracking with CLF–CBF–QP

Mathematical analysis for the project. Notation follows the course:
state $`s \in \mathbb{R}^n`$, action $`a \in \mathbb{R}^m`$, controller
$`\pi(s)`$, plant dynamics $`\dot{s} = P(s, a)`$, Lyapunov function
$`L(s)`$, class-$`\mathcal{K}_\infty`$ comparison functions
$`\kappa_{\mathrm{low}}, \kappa_{\mathrm{up}}, \kappa_{\mathrm{dec}}`$,
class-$`\mathcal{KL}`$ function $`\beta`$, safe set
$`\mathbb{S}_{\mathrm{safe}} \subset \mathbb{S}`$, barrier certificate
$`B(s)`$.

## 1. System dynamics

Planar double integrator:

```math
\dot p = v, \qquad \dot v = u \in \mathbb{R}^2.
```

State $`s = (p, v) \in \mathbb{R}^4`$, action $`a = u \in \mathbb{R}^2`$,
plant map $`P(s, u) = (v, u)`$.

## 2. Reference trajectory

The reference is an ellipse:

```math
p_r(t) =
\begin{bmatrix}
c_x + a \cos(\theta_0 + \omega t) \\
c_y + b \sin(\theta_0 + \omega t)
\end{bmatrix}.
```

Velocity and acceleration:

```math
v_r = \dot p_r =
\begin{bmatrix}
- a\, \omega \sin(\theta_0 + \omega t) \\
  b\, \omega \cos(\theta_0 + \omega t)
\end{bmatrix},
\qquad
a_r = \ddot p_r =
\begin{bmatrix}
- a\, \omega^2 \cos(\theta_0 + \omega t) \\
- b\, \omega^2 \sin(\theta_0 + \omega t)
\end{bmatrix}.
```

Tracking errors:

```math
e_p = p - p_r, \qquad e_v = v - v_r.
```

## 3. Control Lyapunov function

Pick the positive-definite candidate

```math
L(e_p, e_v) = \tfrac{1}{2}\,k_p\,\|e_p\|^2 + \tfrac{1}{2}\,\|e_v\|^2.
```

$`L`$ is a quadratic form on the error coordinates, so there exist
$`\kappa_{\mathrm{low}}, \kappa_{\mathrm{up}} \in \mathcal{K}_\infty`$ with
$`\kappa_{\mathrm{low}}(\|e\|) \le L(e) \le \kappa_{\mathrm{up}}(\|e\|)`$ where
$`e = (e_p, e_v)`$.

## 4. Error dynamics

```math
\dot e_p = e_v, \qquad \dot e_v = u - a_r.
```

Lyapunov derivative along the flow:

```math
\dot L = e_v^{\!\top}(u - a_r) + k_p\,e_p^{\!\top} e_v.
```

## 5. Nominal (CLF) controller

Choose

```math
u_{\mathrm{des}} = a_r - k_p\,e_p - k_v\,e_v.
```

Substituting into $`\dot L`$:

```math
\dot L = -\,k_v\,\|e_v\|^2 \le 0.
```

## 6. Stability analysis

The closed-loop error system under $`u = u_{\mathrm{des}}`$ is

```math
\dot e_p = e_v, \qquad \dot e_v = -k_p\,e_p - k_v\,e_v,
```

equivalently the second-order system

```math
\ddot e_p + k_v\,\dot e_p + k_p\,e_p = 0.
```

$`\dot L`$ is negative semi-definite (zero on $`\{e_v = 0\}`$), so
asymptotic stability does not follow from the decrease condition
alone. By LaSalle's invariance principle the largest invariant subset
of $`\{\dot L = 0\} = \{e_v = 0\}`$ satisfies $`\dot e_v = 0`$, which
forces $`e_p = 0`$. Hence the only invariant point in that set is the
origin, and $`(e_p, e_v) = 0`$ is **globally asymptotically stable**:
there exists $`\beta \in \mathcal{KL}`$ with

```math
\|e(t)\| \le \beta(\|e(0)\|,\, t) \quad \text{for all } t \ge 0.
```

## 7. Barrier certificate and the safe set

For each obstacle $`i`$ with centre $`c_i`$ and radius $`r_{\mathrm{obs},i}`$,
define the per-obstacle safety radius

```math
r_{s,i} = r_{\mathrm{obs},i} + r_{\mathrm{plant}} + \delta_{\mathrm{safe}},
```

and the safe set

```math
\mathbb{S}_{\mathrm{safe},i} = \{\, s = (p, v) : \|p - c_i\| \ge r_{s,i} \,\}.
```

Using the course's barrier-certificate convention, a valid certificate
satisfies $`B_i(s) \le 0`$ on $`\mathbb{S}_{\mathrm{safe},i}`$ and
$`B_i(s) > 0`$ on $`\mathbb{S} \setminus \mathbb{S}_{\mathrm{safe},i}`$; a
natural choice here is

```math
B_i(s) = r_{s,i}^2 - \|p - c_i\|^2.
```

The implementation (and the CBF literature) prefers the sign-flipped
form

```math
h_i(p) = \|p - c_i\|^2 - r_{s,i}^2 = -\,B_i(s),
```

so $`h_i(p) \ge 0 \Leftrightarrow s \in \mathbb{S}_{\mathrm{safe},i}`$. Both
are equivalent.

## 8. Higher-order control barrier function

$`h_i`$ has relative degree 2 with respect to $`u`$. First and second
derivatives along the flow:

```math
\dot h_i = 2\,(p - c_i)^{\!\top} v,
```

```math
\ddot h_i = 2\,\|v\|^2 + 2\,(p - c_i)^{\!\top} u.
```

The HOCBF condition, with $`\alpha_1, \alpha_2 > 0`$ chosen so
$`s^2 + \alpha_1 s + \alpha_2`$ is Hurwitz, is

```math
\ddot h_i + \alpha_1\,\dot h_i + \alpha_2\,h_i \ge 0.
```

This is **linear in $`u`$**:

```math
2\,(p - c_i)^{\!\top} u
\;\ge\;
-2\,\|v\|^2
- 2\,\alpha_1\,(p - c_i)^{\!\top} v
- \alpha_2\,h_i.
```

## 9. Slack-relaxed CLF–CBF quadratic program

The CLF appears implicitly through the reference $`u_{\mathrm{des}}`$; the
QP acts as a minimally invasive safety filter. A non-negative slack
$`s_{\mathrm{slk}} \ge 0`$ keeps the program feasible when no strictly
safe control exists, penalised heavily so $`s_{\mathrm{slk}}^\star = 0`$
whenever a safe control is available:

```math
\min_{u,\,s_{\mathrm{slk}}}\;
\tfrac{1}{2}\,\|u - u_{\mathrm{des}}\|^2 + \rho\, s_{\mathrm{slk}}
```

subject to

```math
A\,u + s_{\mathrm{slk}}\,\mathbf{1} \ge b,
\qquad
s_{\mathrm{slk}} \ge 0,
```

with

```math
A_i = 2\,(p - c_i)^{\!\top},
\qquad
b_i = -2\,\|v\|^2 - 2\,\alpha_1\,(p - c_i)^{\!\top} v - \alpha_2\,h_i,
```

and box control bounds

```math
A_{\mathrm{box}} =
\begin{bmatrix}
 1 & 0 \\
-1 & 0 \\
 0 & 1 \\
 0 & -1
\end{bmatrix},
\qquad
b_{\mathrm{box}} =
\begin{bmatrix}
 u_{\max} \\
-u_{\max} \\
 u_{\max} \\
-u_{\max}
\end{bmatrix}.
```

## 10. Final controller

```math
(u^\star,\, s_{\mathrm{slk}}^\star) \;=\;
\arg\min_{u,\, s_{\mathrm{slk}}}
\tfrac{1}{2}\,\|u - u_{\mathrm{des}}\|^2 + \rho\, s_{\mathrm{slk}}
```

```math
\text{s.t. } A\,u + s_{\mathrm{slk}}\,\mathbf{1} \ge b,\;\;
s_{\mathrm{slk}} \ge 0,\;\;
|u_j| \le u_{\max}.
```

## 11. Guarantees

- **Forward invariance of $`\mathbb{S}_{\mathrm{safe}}`$.** While
  $`s_{\mathrm{slk}}^\star = 0`$, the CBF constraint is strictly
  satisfied and $`h_i(p(t)) \ge 0`$ for all $`i`$ and all $`t`$, so the
  plant never enters any unsafe set.
- **Asymptotic tracking** $`e_p, e_v \to 0`$ holds whenever the CBF
  constraint is inactive (no safety override is required); in that
  regime $`u^\star = u_{\mathrm{des}}`$ and $`\dot L = -k_v\|e_v\|^2 \le 0`$,
  so there exists $`\beta \in \mathcal{KL}`$ with
  $`\|e(t)\| \le \beta(\|e(0)\|, t)`$.
- **Practical stability / ultimate boundedness** when the CBF is
  active. The safety filter modifies $`u`$ away from $`u_{\mathrm{des}}`$,
  monotone descent of $`L`$ is no longer guaranteed, and only a
  vicinity of the origin can be reached — analogous to the
  sample-and-hold practical-stability result: from the decrease bound
  $`\dot L \le -\kappa_{\mathrm{dec}}(\|e\|)`$ holding only when the filter
  is inactive, one obtains $`\|e\| \le \kappa_{\mathrm{low}}^{-1}(\bar\ell)`$
  for some level $`\bar\ell`$ set by the activity of the filter.
- **Graceful degradation** if no safe control exists
  ($`s_{\mathrm{slk}}^\star > 0`$): the slack keeps the QP feasible and
  yields a best-effort response; safety must then be re-established by
  control authority ($`u_{\max}`$) and parameter choice
  ($`\alpha_1, \alpha_2, \delta_{\mathrm{safe}}`$).
