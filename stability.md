\section{Safe Ellipse Tracking with CLF-CBF-QP}

\subsection{System Dynamics}

Consider a planar double integrator:
\begin{equation}
\dot{p} = v, \quad \dot{v} = u \in \mathbb{R}^2
\end{equation}

\subsection{Reference Trajectory}

The reference trajectory is an ellipse:
\begin{equation}
p_r(t) =
\begin{bmatrix}
c_x + a \cos(\theta_0 + \omega t) \\
c_y + b \sin(\theta_0 + \omega t)
\end{bmatrix}
\end{equation}

Velocity and acceleration:
\begin{equation}
v_r = \dot{p}_r =
\begin{bmatrix}
- a \omega \sin(\theta_0 + \omega t) \\
b \omega \cos(\theta_0 + \omega t)
\end{bmatrix}
\end{equation}

\begin{equation}
a_r = \ddot{p}_r =
\begin{bmatrix}
- a \omega^2 \cos(\theta_0 + \omega t) \\
- b \omega^2 \sin(\theta_0 + \omega t)
\end{bmatrix}
\end{equation}

Tracking errors:
\begin{equation}
e_p = p - p_r, \quad e_v = v - v_r
\end{equation}

\subsection{Control Lyapunov Function}

Define:
\begin{equation}
V(e_p, e_v) = \frac{1}{2} \|e_v\|^2 + \frac{k_p}{2} \|e_p\|^2
\end{equation}

\subsection{Error Dynamics}

\begin{equation}
\dot e_p = e_v, \quad \dot e_v = u - a_r
\end{equation}

Lyapunov derivative:
\begin{equation}
\dot V = e_v^\top (u - a_r) + k_p e_p^\top e_v
\end{equation}

\subsection{Nominal Controller}

Choose:
\begin{equation}
u_{\text{des}} = a_r - k_p e_p - k_v e_v
\end{equation}

Then:
\begin{equation}
\dot V = -k_v \|e_v\|^2 \le 0
\end{equation}

\subsection{Stability Analysis}

The closed-loop system becomes:
\begin{equation}
\dot e_p = e_v, \quad
\dot e_v = -k_p e_p - k_v e_v
\end{equation}

Equivalent second-order system:
\begin{equation}
\ddot e_p + k_v \dot e_p + k_p e_p = 0
\end{equation}

Using LaSalle's invariance principle, the equilibrium $(e_p, e_v) = (0,0)$ is globally asymptotically stable.

\subsection{Control Barrier Functions}

Define safety constraints for obstacles. The per-obstacle safety radius
combines the obstacle radius, the plant radius, and a fixed margin:
\begin{equation}
r_{s,i} = r_{\text{obs},i} + r_{\text{plant}} + \delta_{\text{safe}}
\end{equation}
\begin{equation}
h_i(p) = \|p - c_i\|^2 - r_{s,i}^2
\end{equation}

First and second derivatives:
\begin{equation}
\dot h_i = 2(p - c_i)^\top v
\end{equation}

\begin{equation}
\ddot h_i = 2\|v\|^2 + 2(p - c_i)^\top u
\end{equation}

HOCBF condition:
\begin{equation}
\ddot h_i + \alpha_1 \dot h_i + \alpha_0 h_i \ge 0
\end{equation}

Linear constraint in $u$:
\begin{equation}
2(p - c_i)^\top u \ge
-2\|v\|^2
-2\alpha_1 (p - c_i)^\top v
- \alpha_0 h_i
\end{equation}

\subsection{Quadratic Program}

The CLF appears implicitly through the nominal control $u_{\text{des}}$;
the QP acts as a minimally invasive safety filter. A non-negative slack
$s \ge 0$ keeps the program feasible when no strictly safe input exists,
penalised heavily so $s = 0$ whenever a safe control is available:
\begin{equation}
\min_{u,\,s} \frac{1}{2} \|u - u_{\text{des}}\|^2 + \rho\, s
\end{equation}

subject to:
\begin{equation}
A u + s\,\mathbf{1} \ge b, \qquad s \ge 0
\end{equation}

where:
\begin{equation}
A_i = 2(p - c_i)^\top
\end{equation}

\begin{equation}
b_i =
-2\|v\|^2
-2\alpha_1 (p - c_i)^\top v
- \alpha_0 h_i
\end{equation}

Control bounds:
\begin{equation}
A_{\text{box}} =
\begin{bmatrix}
1 & 0 \\
-1 & 0 \\
0 & 1 \\
0 & -1
\end{bmatrix},
\quad
b_{\text{box}} =
\begin{bmatrix}
u_{\max} \\
- u_{\max} \\
u_{\max} \\
- u_{\max}
\end{bmatrix}
\end{equation}

\subsection{Final Controller}

\begin{equation}
(u^*, s^*) = \arg\min_{u,s} \frac{1}{2} \|u - u_{\text{des}}\|^2 + \rho\, s
\quad \text{s.t. } A u + s\,\mathbf{1} \ge b,\ s \ge 0,\ |u_j| \le u_{\max}
\end{equation}

\subsection{Guarantees}

\begin{itemize}
\item Forward invariance: while $s^* = 0$ (CBF strictly satisfied),
$h_i(p(t)) \ge 0$ for all $i$, so the plant never enters any unsafe set.
\item Asymptotic tracking $e_p, e_v \to 0$ holds whenever the CBF
constraint is inactive (i.e.\ no safety override is required); in that
regime $u^* = u_{\text{des}}$ and $\dot V = -k_v\|e_v\|^2 \le 0$.
\item When the CBF constraint is active, the safety filter modifies $u$
away from $u_{\text{des}}$, and only ultimate boundedness of the
tracking error can be claimed.
\item If no safe control exists ($s^* > 0$), the slack provides a
graceful, best-effort response while preserving QP feasibility; safety
must then be re-established by control authority ($u_{\max}$) and
parameter choice ($\alpha_1, \alpha_2, \delta_{\text{safe}}$).
\end{itemize}