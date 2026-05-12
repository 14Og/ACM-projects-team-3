from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from config import ProjectConfig, default_config, load_config
from controller import (
    AdaptiveLyapunovController,
    AdaptiveSimplifiedController,
    BacksteppingFull,
    BacksteppingSimplified,
)
from simulation import run_rollout, save_rollout_data_csv, summarize_rollout
from system import PlanarArm, with_payload_error


CONTROLLER_CHOICES = ("adaptive", "adaptive_simp", "backstepping_full", "backstepping_simp", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adaptive, full backstepping, and simplified backstepping controllers."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")
    parser.add_argument(
        "--controller",
        choices=CONTROLLER_CHOICES,
        default=None,
        help="Controller to run. Default comes from config, usually 'all'.",
    )
    parser.add_argument(
        "--payload",
        action="store_true",
        help="Run only the unknown-payload scenario.",
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Run only the nominal scenario.",
    )
    return parser.parse_args()


def make_controller(config: ProjectConfig, name: str):
    if name == "adaptive":
        return AdaptiveLyapunovController(
            config.adaptive_controller,
            config.dynamics.torque_limits,
        )
    if name == "adaptive_simp":
        return AdaptiveSimplifiedController(
            config.adaptive_controller,
            config.robot,
            config.dynamics.torque_limits,
            config.backstepping_controller,
        )
    if name == "backstepping_full":
        return BacksteppingFull(
            config.backstepping_controller,
            config.robot,
            config.dynamics.torque_limits,
        )
    if name == "backstepping_simp":
        return BacksteppingSimplified(
            config.backstepping_controller,
            config.robot,
            config.dynamics.torque_limits,
        )
    raise ValueError(f"Unknown controller: {name}")


def selected_controller_names(config: ProjectConfig, override: str | None) -> list[str]:
    selected = override or config.controller_selection.controller_type
    if selected == "all":
        return ["adaptive", "adaptive_simp", "backstepping_full", "backstepping_simp"]
    return [selected]


def selected_scenarios(args: argparse.Namespace, config: ProjectConfig) -> list[tuple[str, bool]]:
    if args.payload and args.nominal_only:
        raise ValueError("Use either --payload or --nominal-only, not both.")
    if args.payload:
        return [("payload", True)]
    if args.nominal_only:
        return [("nominal", False)]
    if config.controller_selection.simulate_payload_error:
        return [("payload", True)]
    return [("nominal", False), ("payload", True)]


def run_scenario(
    *,
    base_config: ProjectConfig,
    scenario_label: str,
    use_payload: bool,
    controller_names: list[str],
) -> dict[str, object]:
    real_dynamics = (
        with_payload_error(
            base_config.dynamics,
            base_config.controller_selection.payload_multiplier,
        )
        if use_payload
        else base_config.dynamics
    )

    output_root = Path(base_config.output.figures_dir).parent / scenario_label
    scenario_config = replace(
        base_config,
        output=replace(
            base_config.output,
            figures_dir=str(output_root / "figures"),
            animations_dir=str(output_root / "animations"),
        ),
    )

    rollouts = {}
    summaries = {}
    for controller_name in controller_names:
        controller = make_controller(scenario_config, controller_name)
        rollout = run_rollout(scenario_config, controller, real_dynamics=real_dynamics)
        rollouts[controller_name] = rollout
        summaries[controller_name] = summarize_rollout(
            scenario_config,
            rollout,
            real_dynamics=real_dynamics,
        )

    figures = []
    animations = []
    try:
        from visualisation import save_all_animations, save_all_plots

        arm = PlanarArm(scenario_config.robot)
        figures = save_all_plots(
            config=scenario_config,
            arm=arm,
            rollouts=rollouts,
            figures_dir=Path(scenario_config.output.figures_dir),
            real_dynamics=real_dynamics,
        )
        animations = save_all_animations(
            config=scenario_config,
            arm=arm,
            rollouts=rollouts,
            animations_dir=Path(scenario_config.output.animations_dir),
        )
    except ModuleNotFoundError as exc:
        print(f"plots/animations skipped: {exc}")
    csv_path = save_rollout_data_csv(rollouts, output_root / "rollouts.csv")
    summary_path = _save_summary_csv(summaries, output_root / "summary.csv")

    return {
        "label": scenario_label,
        "payload": use_payload,
        "rollouts": rollouts,
        "summaries": summaries,
        "figures": figures,
        "animations": animations,
        "csv": csv_path,
        "summary": summary_path,
    }


def _save_summary_csv(summaries: dict[str, dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for summary in summaries.values() for key in summary})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["controller", *keys])
        writer.writeheader()
        for controller, summary in summaries.items():
            writer.writerow({"controller": controller, **summary})
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else default_config()
    controller_names = selected_controller_names(config, args.controller)
    scenarios = selected_scenarios(args, config)

    print(f"controllers: {', '.join(controller_names)}")
    for scenario_label, use_payload in scenarios:
        result = run_scenario(
            base_config=config,
            scenario_label=scenario_label,
            use_payload=use_payload,
            controller_names=controller_names,
        )
        print(f"\nscenario: {scenario_label}")
        print(f"summary: {result['summary']}")
        print(f"rollout csv: {result['csv']}")
        if result["animations"]:
            print("gifs:")
            for path in result["animations"]:
                print(f"  {path}")
        for controller, summary in result["summaries"].items():
            print(
                f"  {controller}: final ||z1||={summary['final_q_error_norm_rad']:.4f}, "
                f"tail mean ||z1||={summary['tail_mean_q_error_norm_rad']:.4f}, "
                f"rms tau={summary['rms_torque']:.2f}"
            )


if __name__ == "__main__":
    main()
