"""
plot_timeseries_eval.py — Queue Length time-series trong evaluation episode.

Chay 1 episode day du (Cologne8, 7–8 AM) voi:
  - Best RL controller (mac dinh DDQN/wait-clip s42)
  - Fixed-time controller
  - Webster controller

Ve tong queue (so xe dung tren toan mang) theo thoi gian mo phong (giay).

Su dung:
    python plot_timeseries_eval.py
    python plot_timeseries_eval.py --models_dir exp_cologne8_real \
        --algo ddqn --reward wait-clip --seed s42 --out figures
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if "SUMO_HOME" not in os.environ:
    sys.exit("[plot_timeseries_eval] SUMO_HOME chua duoc khai bao")

_HERE    = Path(__file__).parent
C8_NET   = _HERE / "nets" / "cologne8" / "cologne8.net.xml"
C8_ROUTE = _HERE / "nets" / "cologne8" / "cologne8.rou.xml"

SUMO_BEGIN = 25200
SUMO_END   = 28800

from rl_controller.traffic_env   import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env      import MultiAgentVecEnv
from rl_controller.webster       import DynamicWebsterController

USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

PALETTE = {
    "DQN":        "#1976D2",
    "DDQN":       "#F57C00",
    "PPO":        "#388E3C",
    "Fixed-time": "#757575",
    "Webster":    "#455A64",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _total_queue(sumo_conn, signal_ids) -> float:
    total = 0
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        for lane in lanes:
            try:
                total += sumo_conn.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
    return float(total)


def smooth(arr, w=24):
    """Rolling mean, w=24 → 2-minute window at 5-second resolution."""
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode="same")


def _make_env():
    return TrafficControlEnv(
        net_file        = str(C8_NET),
        route_file      = str(C8_ROUTE),
        sim_duration    = SUMO_END + 99999,
        reward_fn       = "queue",
        obs_class       = BaselineObservation,
        single_agent    = False,
        show_warnings   = False,
        extra_sumo_args = f"--begin {SUMO_BEGIN}",
    )


def _make_fixed_env():
    return TrafficControlEnv(
        net_file        = str(C8_NET),
        route_file      = str(C8_ROUTE),
        sim_duration    = SUMO_END + 99999,
        reward_fn       = "queue",
        obs_class       = BaselineObservation,
        single_agent    = False,
        fixed_signal    = True,
        show_warnings   = False,
        extra_sumo_args = f"--begin {SUMO_BEGIN}",
    )


# ── Collectors ────────────────────────────────────────────────────────────────

def collect_rl(model_path: str, algo: str):
    """Returns (times_s, queues) relative to episode start."""
    from stable_baselines3 import PPO, DQN
    print(f"  [RL] Loading {algo.upper()} from {Path(model_path).name} ...", flush=True)
    agent   = (PPO if algo == "ppo" else DQN).load(model_path, env=None, device="cpu")
    vec_env = MultiAgentVecEnv(_make_env)
    obs     = vec_env.reset()

    times, queues = [], []
    while True:
        sumo = vec_env.env.sumo
        if sumo is None:
            break
        t = sumo.simulation.getTime()
        if t >= SUMO_END:
            break
        times.append(t - SUMO_BEGIN)
        queues.append(_total_queue(sumo, vec_env.signal_ids))
        actions, _ = agent.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(actions)
        if dones.any():
            break
    try:
        vec_env.close()
    except Exception:
        pass
    print(f"  [RL] Done — {len(times)} steps collected.", flush=True)
    return np.array(times), np.array(queues)


def collect_fixed():
    """Returns (times_s, queues) for Fixed-time controller."""
    print("  [Fixed] Running ...", flush=True)
    vec_env  = MultiAgentVecEnv(_make_fixed_env)
    obs      = vec_env.reset()
    n_agents = vec_env.num_envs

    times, queues = [], []
    while True:
        sumo = vec_env.env.sumo
        if sumo is None:
            break
        t = sumo.simulation.getTime()
        if t >= SUMO_END:
            break
        times.append(t - SUMO_BEGIN)
        queues.append(_total_queue(sumo, vec_env.signal_ids))
        _, _, dones, _ = vec_env.step(np.zeros(n_agents, dtype=int))
        if dones.any():
            break
    try:
        vec_env.close()
    except Exception:
        pass
    print(f"  [Fixed] Done — {len(times)} steps collected.", flush=True)
    return np.array(times), np.array(queues)


def collect_webster():
    """Returns (times_s, queues) for Webster controller."""
    import sumolib
    import traci as _traci

    print("  [Webster] Running ...", flush=True)
    label = f"wsb_ts_{int(time.time()*1000) % 100000}"
    cmd   = [
        sumolib.checkBinary("sumo"),
        "-n", str(C8_NET),
        "-r", str(C8_ROUTE),
        "--begin",            str(SUMO_BEGIN),
        "--time-to-teleport", "-1",
        "--no-warnings",
    ]
    if USE_LIBSUMO:
        _traci.start(cmd); sumo = _traci
    else:
        _traci.start(cmd, label=label); sumo = _traci.getConnection(label)

    signal_ids = list(sumo.trafficlight.getIDList())

    def green_phases(ts):
        logics = sumo.trafficlight.getAllProgramLogics(ts)
        if not logics:
            return [0]
        return [i for i, ph in enumerate(logics[0].phases)
                if "G" in ph.state.upper() and "Y" not in ph.state.upper()]

    controllers = {ts: DynamicWebsterController(ts, green_phases(ts), sumo)
                   for ts in signal_ids}

    times, queues = [], []
    try:
        while sumo.simulation.getTime() < SUMO_END:
            sumo.simulationStep()
            t = sumo.simulation.getTime()
            for ctrl in controllers.values():
                ctrl.step(t)
            times.append(t - SUMO_BEGIN)
            queues.append(_total_queue(sumo, signal_ids))
    except Exception as e:
        print(f"  [Webster] Error: {e}", flush=True)
    finally:
        try:
            if not USE_LIBSUMO:
                _traci.switch(label)
            _traci.close()
        except Exception:
            pass
    print(f"  [Webster] Done — {len(times)} steps collected.", flush=True)
    return np.array(times), np.array(queues)


# ── Model discovery ───────────────────────────────────────────────────────────

def find_model(models_dir: Path, algo: str, reward: str, seed: str):
    target = f"{algo}_{reward}_{seed}"
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parts = subdir.name.split("_")
        a     = parts[0]
        r     = "_".join(parts[1:-1])
        s     = parts[-1]
        if a != algo or r != reward or s != seed:
            continue
        for pat in [f"*_{algo}_final_model.zip", f"*_{algo}_final_model"]:
            candidates = sorted(subdir.glob(pat))
            if candidates:
                return str(candidates[0])
    return None


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_timeseries(series: dict, algo_label: str, out: Path):
    """
    series = {label: (times, queues)}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, (times, queues) in series.items():
        color = next(
            (v for k, v in PALETTE.items() if label.upper().startswith(k.upper())),
            "#999999"
        )
        # Smooth signal
        qs = smooth(queues, w=24)

        # Dashed style for baselines
        ls = "-" if label.upper() not in ("FIXED-TIME", "WEBSTER") else (
            "--" if "Fixed" in label else ":"
        )
        lw = 2.0 if ls == "-" else 1.5

        ax.plot(times, qs, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.9)

    ax.set_xlabel("Simulation Time (s) — Episode 07:00–08:00", fontsize=11)
    ax.set_ylabel("Total Queue Length (halting vehicles)", fontsize=11)
    ax.set_title(f"Queue Length over Time — Cologne8\n"
                 f"RL: {algo_label} vs Fixed-time vs Webster", fontsize=12)
    ax.set_xlim(0, SUMO_END - SUMO_BEGIN)

    # x-tick as HH:MM
    ticks = np.arange(0, 3601, 600)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{7 + t//3600:02d}:{(t%3600)//60:02d}" for t in ticks])

    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="./exp_cologne8_real")
    p.add_argument("--algo",       default="ddqn",      choices=["dqn", "ddqn", "ppo"])
    p.add_argument("--reward",     default="wait-clip",
                   choices=["queue", "pressure", "wait-clip"])
    p.add_argument("--seed",       default="s42")
    p.add_argument("--out",        default="./figures")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models_dir)
    model_path = find_model(models_dir, args.algo, args.reward, args.seed)
    if model_path is None:
        sys.exit(f"[ERROR] Khong tim thay model: {args.algo}/{args.reward}/{args.seed} "
                 f"trong {models_dir}")

    algo_label = f"{args.algo.upper()} ({args.reward}, {args.seed})"
    print("=" * 60)
    print("  plot_timeseries_eval.py — Cologne8")
    print(f"  RL model : {model_path}")
    print(f"  Output   : {args.out}/")
    print("=" * 60)

    series = {}

    # RL
    times, queues = collect_rl(model_path, args.algo)
    series[algo_label] = (times, queues)

    # Fixed-time
    times, queues = collect_fixed()
    series["Fixed-time"] = (times, queues)

    # Webster
    times, queues = collect_webster()
    series["Webster"] = (times, queues)

    out = out_dir / "fig_timeseries_c8.png"
    plot_timeseries(series, algo_label, out)
    print(f"\nHoan tat. -> {out}")


if __name__ == "__main__":
    main()
