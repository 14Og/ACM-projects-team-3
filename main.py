"""Entry point: load config, randomise episode, run simulation, show outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lyapunov_apf.config import APFConfig, EnvConfig, SimConfig
from lyapunov_apf.controller import APFController
from lyapunov_apf.simulation import SimulationEngine
from lyapunov_apf.visualization import Visualizer


def print_summary(data: dict, env: EnvConfig, episode, seed: int | None) -> None:
    err = data["err"]
    speed = data["speed"]
    accel = data["accel"]
    V = data["V"]
    tgt_speed = np.linalg.norm(data["v_ref"], axis=1)

    print("Scenario")
    print(f"  Seed                 : {seed if seed is not None else 'random'}")
    print(f"  Target start theta   : {episode.theta0:.4f} rad")
    print(f"  Target omega         : {episode.omega:.4f} rad/s")
    print(f"  Sim horizon          : {episode.t_final:.2f} s")
    print(f"  Robot start          : ({episode.p0[0]:.3f}, {episode.p0[1]:.3f})")
    print(f"  Obstacles            : {len(episode.obstacles)}")
    print()
    print("Simulation summary")
    print(f"  Final tracking error : {err[-1]:.4f}")
    print(f"  Mean tracking error  : {np.mean(err):.4f}")
    print(f"  Max tracking error   : {np.max(err):.4f}")
    print(f"  Max plant speed      : {np.max(speed):.4f}")
    print(f"  Mean target speed    : {np.mean(tgt_speed):.4f}")
    print(f"  Max acceleration     : {np.max(accel):.4f}")
    print(f"  V_total initial      : {V[0]:.4f}")
    print(f"  V_total final        : {V[-1]:.4f}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lyapunov-based trajectory tracking demo on an elliptical reference path."
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible scenario generation.")
    parser.add_argument("--constrain-control", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable (default) or disable (--no-constrain-control) the "
                             "||u|| <= u_max control clipping.")
    parser.add_argument("--feedforward", action=argparse.BooleanOptionalAction, default=None,
                        help="Add reference acceleration a_ref to control (asymptotic "
                             "tracking of moving targets in the no-obstacle case).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = EnvConfig()
    sim = SimConfig()
    apf = APFConfig()
    if args.constrain_control is not None:
        apf.constrain_control = args.constrain_control
    if args.feedforward is not None:
        apf.feedforward = args.feedforward
    controller = APFController(apf, env.plant_radius)
    simulation = SimulationEngine(env, sim)
    visualizer = Visualizer()

    episode = simulation.randomize_episode(rng)
    ep = 0

    plt.ion()

    # Shared mutable state for key handlers
    key_state = {"next_ep": False, "quit": False}
    fig: plt.Figure | None = None
    ani = None

    def on_key(event) -> None:
        if event.key == "enter":
            key_state["next_ep"] = True
        elif event.key == "q":
            key_state["quit"] = True

    while True:
        ep += 1
        data = simulation.run_simulation(episode, controller)
        print_summary(data, env, episode, args.seed)

        if ani is not None:
            ani.event_source.stop()

        fig, ani = visualizer.make_animation(data, env, episode, ep_num=ep, fig=fig)

        if ep == 1:
            fig.canvas.mpl_connect("key_press_event", on_key)

        # Spin until ENTER (next episode) or q (quit)
        key_state["next_ep"] = False
        while not key_state["next_ep"] and not key_state["quit"]:
            plt.pause(0.05)

        if key_state["quit"]:
            sys.exit(0)

        episode = simulation.next_episode(episode, rng)


if __name__ == "__main__":
    main()
