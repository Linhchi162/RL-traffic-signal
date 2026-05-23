"""
eval_timeseries_c8.py — Per-step queue time series for Cologne-8.

Runs PPO-pressure (5 seeds) and Fixed-time, records total queue at every
decision step, then plots mean ± std as a time-series with the demand-surge
window highlighted.

Anomalous window (from flow profile): 07:13–07:21 (demand surge peak at 07:15,
78 veh/min vs. background ~40 veh/min).

Output: figures/fig_timeseries_c8.png
"""

import os, sys, logging
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("Set SUMO_HOME first.")

_HERE     = Path(__file__).parent
NET_FILE  = str(_HERE / "nets" / "cologne8" / "cologne8.net.xml")
ROU_FILE  = str(_HERE / "nets" / "cologne8" / "cologne8.rou.xml")
MODELS_DIR = _HERE / "exp_cologne8_direct"
OUT        = "figures/fig_timeseries_c8.png"

SUMO_BEGIN = 25200   # 7:00 AM in seconds from midnight
SUMO_END   = 28800   # 8:00 AM

# Demand-surge window (from flow profile)
SURGE_START = 25200 + 13 * 60   # 07:13
SURGE_END   = 25200 + 21 * 60   # 07:21

# Lazy imports — deferred to avoid loading torch at startup
def _get_env_classes():
    from rl_controller.traffic_env   import TrafficControlEnv
    from rl_controller.state_builder import BaselineObservation
    from rl_controller.grid_env      import MultiAgentVecEnv
    return TrafficControlEnv, BaselineObservation, MultiAgentVecEnv


def make_env(fixed_signal=False):
    TrafficControlEnv, BaselineObservation, _ = _get_env_classes()
    return TrafficControlEnv(
        net_file        = NET_FILE,
        route_file      = ROU_FILE,
        sim_duration    = SUMO_END + 99999,
        reward_fn       = "pressure",
        obs_class       = BaselineObservation,
        single_agent    = False,
        fixed_signal    = fixed_signal,
        show_warnings   = False,
        extra_sumo_args = f"--begin {SUMO_BEGIN}",
    )


def make_vec_env(fixed_signal=False):
    _, _, MultiAgentVecEnv = _get_env_classes()
    return MultiAgentVecEnv(lambda: make_env(fixed_signal=fixed_signal))


def collect_step_queue(sumo_conn, signal_ids):
    total_q = 0
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        for lane in lanes:
            try:
                total_q += sumo_conn.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
    return total_q


def run_fixed_time():
    vec_env = make_vec_env(fixed_signal=True)
    vec_env.reset()
    times, queues = [], []
    while True:
        sumo = vec_env.env.sumo
        if sumo is None:
            break
        t = sumo.simulation.getTime()
        if t >= SUMO_END:
            break
        q = collect_step_queue(sumo, vec_env.signal_ids)
        times.append(t)
        queues.append(q)
        n = vec_env.num_envs
        obs, _, dones, _ = vec_env.step(np.zeros(n, dtype=int))
        if dones.any():
            break
    try: vec_env.close()
    except: pass
    return times, queues


def run_ppo_seed(model_path: str):
    from stable_baselines3 import PPO
    agent = PPO.load(model_path, env=None, device="cpu")
    vec_env = make_vec_env(fixed_signal=False)
    obs = vec_env.reset()
    times, queues = [], []
    while True:
        sumo = vec_env.env.sumo
        if sumo is None:
            break
        t = sumo.simulation.getTime()
        if t >= SUMO_END:
            break
        q = collect_step_queue(sumo, vec_env.signal_ids)
        times.append(t)
        queues.append(q)
        actions, _ = agent.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(actions)
        if dones.any():
            break
    try: vec_env.close()
    except: pass
    return times, queues


def resample_to_grid(times, queues, t_start, t_end, bin_sec=60):
    """Bin queue values into fixed-width intervals, return (bin_centres, means)."""
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    counts = np.zeros(len(bins) - 1)
    sums   = np.zeros(len(bins) - 1)
    for t, q in zip(times, queues):
        idx = int((t - t_start) / bin_sec)
        if 0 <= idx < len(counts):
            sums[idx]   += q
            counts[idx] += 1
    means = np.where(counts > 0, sums / counts, np.nan)
    centres = (bins[:-1] + bins[1:]) / 2
    return centres, means


def to_hhmm(sec):
    h, m = divmod(int(sec), 3600)
    return f"{h:02d}:{m//60:02d}"


def main():
    Path("figures").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.WARNING)

    ppo_seeds = sorted(MODELS_DIR.glob("ppo_pressure_s*/cologne8_ppo_final_model.zip"))
    print(f"Found {len(ppo_seeds)} PPO-pressure seeds")

    # ── Fixed-time (single run, deterministic) ────────────────────────────────
    print("Running Fixed-time ...", flush=True)
    ft_times, ft_queues = run_fixed_time()
    print(f"  {len(ft_times)} steps, queue range [{min(ft_queues)}, {max(ft_queues)}]")

    # ── PPO-pressure (one run per seed) ───────────────────────────────────────
    ppo_series = []
    for i, model_path in enumerate(ppo_seeds):
        seed_name = model_path.parent.name
        print(f"Running PPO-pressure {seed_name} ({i+1}/{len(ppo_seeds)}) ...", flush=True)
        t, q = run_ppo_seed(str(model_path))
        ppo_series.append((t, q))
        print(f"  {len(t)} steps, queue range [{min(q)}, {max(q)}]")

    # ── Resample to 1-minute bins ─────────────────────────────────────────────
    BIN = 60
    ft_centres, ft_means = resample_to_grid(ft_times, ft_queues, SUMO_BEGIN, SUMO_END, BIN)

    ppo_grid = []
    for t, q in ppo_series:
        centres, means = resample_to_grid(t, q, SUMO_BEGIN, SUMO_END, BIN)
        ppo_grid.append(means)
    ppo_arr  = np.array(ppo_grid)
    ppo_mean = np.nanmean(ppo_arr, axis=0)
    ppo_std  = np.nanstd(ppo_arr,  axis=0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(9, 4.5))

    x_ft  = ft_centres
    x_ppo = centres   # same grid

    # Fixed-time line
    ax.plot(x_ft, ft_means, color="#E53935", linewidth=2, label="Fixed-time")

    # PPO-pressure line + std band
    ax.plot(x_ppo, ppo_mean, color="#1976D2", linewidth=2, label="PPO-pressure")
    ax.fill_between(x_ppo, ppo_mean - ppo_std, ppo_mean + ppo_std,
                    color="#1976D2", alpha=0.18)

    # Highlight demand-surge window
    ax.axvspan(SURGE_START, SURGE_END, color="#4CAF50", alpha=0.15,
               label="Demand surge (07:13–07:21)")
    ax.axvline(SURGE_START, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(SURGE_END,   color="gray", linewidth=0.8, linestyle=":")

    # X-axis: HH:MM every 5 min
    tick_sec = np.arange(SUMO_BEGIN, SUMO_END + 1, 5 * 60)
    ax.set_xticks(tick_sec)
    ax.set_xticklabels([to_hhmm(s) for s in tick_sec], rotation=45, ha="right")
    ax.set_xlim(SUMO_BEGIN, SUMO_END)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Mean Queue (vehicles)", fontsize=11)
    ax.set_title("Queue During Demand Surge — Cologne-8", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
