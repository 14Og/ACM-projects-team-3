"""Command-line entry point for the adaptive manipulator project."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .config import load_config
from .controller import AdaptiveLyapunovController, FixedLyapunovController, PlainPDController
from .simulation import run_rollout, save_rollout_data_csv, summarize_rollout
from .system import PlanarArm
from .visualization import save_all_plots, save_animation, show_live_animation


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive control for a 3-DOF planar manipulator with moving obstacles."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to JSON config.")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration.")
    parser.add_argument("--dt", type=float, default=None, help="Override integration step.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF animation generation.")
    parser.add_argument("--animation-stride", type=int, default=15, help="Use every Nth sample.")
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Show real-time animation during simulation (requires interactive display).",
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Disable moving obstacles in simulation.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=30,
        help="Frames per second for live animation (default: 30).",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use robust adaptive controller with sliding mode term (default: use adaptive).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.duration is not None:
        config = replace(config, simulation=replace(config.simulation, duration=float(args.duration)))
    if args.dt is not None:
        config = replace(config, simulation=replace(config.simulation, dt=float(args.dt)))
    if args.no_obstacles:
        from .config import ObstacleConfig
        import numpy as np
        config = replace(
            config,
            obstacles=ObstacleConfig(
                radius=config.obstacles.radius,
                base_centers=np.empty((0, 2)),  # No obstacles
                amplitudes=np.empty((0, 2)),
                omegas=np.empty(0),
                phases=np.empty(0),
            ),
        )

    from .controller import RobustAdaptiveController

    controllers = {
        "adaptive": AdaptiveLyapunovController(
            config.adaptive_controller,
            config.dynamics.torque_limits,
        ),
        "fixed_lyapunov": FixedLyapunovController(
            config.fixed_lyapunov_controller,
            config.dynamics.torque_limits,
        ),
        "plain_pd": PlainPDController(
            config.pd_controller,
            config.dynamics.torque_limits,
        ),
    }

    # Replace adaptive with robust if --robust flag is set
    if args.robust:
        controllers["adaptive"] = RobustAdaptiveController(
            config.adaptive_controller,
            config.dynamics.torque_limits,
        )

    rollouts = {name: run_rollout(config, controller) for name, controller in controllers.items()}
    metrics = {name: summarize_rollout(config, rollout) for name, rollout in rollouts.items()}

    figures_dir = Path(config.output.figures_dir)
    animations_dir = Path(config.output.animations_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = figures_dir / "summary_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    rollout_data_path = save_rollout_data_csv(rollouts, figures_dir / "rollout_timeseries.csv")

    arm = PlanarArm(config.robot)
    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = save_all_plots(
            config=config,
            arm=arm,
            rollouts=rollouts,
            figures_dir=figures_dir,
        )

    animation_path: Path | None = None
    if args.animation:
        # Live real-time animation mode
        show_live_animation(
            config=config,
            arm=arm,
            rollouts=rollouts,
            fps=args.animation_fps,
        )
    elif not args.no_animation:
        animation_path = save_animation(
            config=config,
            arm=arm,
            rollouts=rollouts,
            animations_dir=animations_dir,
            frame_stride=max(1, int(args.animation_stride)),
        )

    print("Adaptive manipulator simulation complete")
    for name in ["adaptive", "fixed_lyapunov", "plain_pd"]:
        print(
            f"  {name:15s} tail_error={metrics[name]['tail_mean_target_error_px']:.3f} px  "
            f"min_clearance={metrics[name]['min_clearance_px']:.3f} px  "
            f"tail_success={metrics[name]['tail_success_fraction']:.3f}"
        )
    print(
        "  final estimates: "
        f"I_hat={rollouts['adaptive'].inertia_hat[-1].round(3).tolist()}  "
        f"D_hat={rollouts['adaptive'].damping_hat[-1].round(3).tolist()}  "
        f"b_hat={rollouts['adaptive'].bias_hat[-1].round(3).tolist()}"
    )
    print(f"  metrics: {metrics_path}")
    print(f"  rollout data: {rollout_data_path}")
    for path in plot_paths:
        print(f"  plot: {path}")
    if animation_path is not None:
        print(f"  animation: {animation_path}")


if __name__ == "__main__":
    main()
