"""
evaluate_per_junction.py — Danh gia queue trung binh theo tung nut giao (Cologne8).

Chay tren 6 controller: DQN / DDQN / PPO (best seed theo queue) + Fixed / Webster / Random.
Luu ket qua vao: results_cologne8_real/junction_queue.csv
Ve bar chart:    figures/fig6_per_junction_c8.png

Su dung:
    python evaluate_per_junction.py
    python evaluate_per_junction.py --models_dir exp_cologne8_real \
                                     --save_dir results_cologne8_real \
                                     --out figures
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("[evaluate_per_junction] SUMO_HOME chua duoc khai bao")

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


# ── Env factory ───────────────────────────────────────────────────────────────

def _make_env(fixed_signal=False):
    return TrafficControlEnv(
        net_file        = str(C8_NET),
        route_file      = str(C8_ROUTE),
        sim_duration    = SUMO_END + 99999,
        reward_fn       = "queue",
        obs_class       = BaselineObservation,
        single_agent    = False,
        fixed_signal    = fixed_signal,
        show_warnings   = False,
        extra_sumo_args = f"--begin {SUMO_BEGIN}",
    )


# ── Per-junction queue collection ─────────────────────────────────────────────

def _collect_junction_queue(sumo_conn, signal_ids):
    """Dict {signal_id: queue_count} tại bước hiện tại."""
    result = {}
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        q = 0
        for lane in lanes:
            try:
                q += sumo_conn.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
        result[sid] = q
    return result


# ── Eval functions ────────────────────────────────────────────────────────────

def eval_model(model_path: str, algo: str) -> dict:
    """Returns {signal_id: mean_queue} over simulation."""
    from stable_baselines3 import PPO, DQN
    agent   = (PPO if algo == "ppo" else DQN).load(model_path, env=None, device="cpu")
    vec_env = MultiAgentVecEnv(lambda: _make_env())
    obs     = vec_env.reset()

    acc = {}; counts = 0
    while True:
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= SUMO_END:
            break
        jq = _collect_junction_queue(sumo_conn, vec_env.signal_ids)
        for sid, q in jq.items():
            acc[sid] = acc.get(sid, 0.0) + q
        counts += 1
        actions, _ = agent.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(actions)
        if dones.any():
            break

    try:
        vec_env.close()
    except Exception:
        pass
    return {sid: acc[sid] / max(1, counts) for sid in acc}


def eval_fixed() -> dict:
    vec_env = MultiAgentVecEnv(lambda: _make_env(fixed_signal=True))
    vec_env.reset()
    acc = {}; counts = 0
    n_agents = vec_env.num_envs
    while True:
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= SUMO_END:
            break
        jq = _collect_junction_queue(sumo_conn, vec_env.signal_ids)
        for sid, q in jq.items():
            acc[sid] = acc.get(sid, 0.0) + q
        counts += 1
        _, _, dones, _ = vec_env.step(np.zeros(n_agents, dtype=int))
        if dones.any():
            break
    try:
        vec_env.close()
    except Exception:
        pass
    return {sid: acc[sid] / max(1, counts) for sid in acc}


def eval_webster() -> dict:
    import sumolib
    import traci as _traci

    label = f"wsb_pj_{int(time.time()*1000) % 100000}"
    cmd   = [
        sumolib.checkBinary("sumo"),
        "-n", str(C8_NET),
        "-r", str(C8_ROUTE),
        "--begin", str(SUMO_BEGIN),
        "--time-to-teleport", "-1",
        "--no-warnings",
    ]
    if USE_LIBSUMO:
        _traci.start(cmd); sumo = _traci
    else:
        _traci.start(cmd, label=label); sumo = _traci.getConnection(label)

    signal_ids = list(sumo.trafficlight.getIDList())

    def get_green_phases(ts):
        logics = sumo.trafficlight.getAllProgramLogics(ts)
        if not logics:
            return [0]
        return [i for i, ph in enumerate(logics[0].phases)
                if "G" in ph.state.upper() and "Y" not in ph.state.upper()]

    controllers = {ts: DynamicWebsterController(ts, get_green_phases(ts), sumo)
                   for ts in signal_ids}
    acc = {}; counts = 0
    try:
        while sumo.simulation.getTime() < SUMO_END:
            sumo.simulationStep()
            t = sumo.simulation.getTime()
            for ctrl in controllers.values():
                ctrl.step(t)
            jq = _collect_junction_queue(sumo, signal_ids)
            for sid, q in jq.items():
                acc[sid] = acc.get(sid, 0.0) + q
            counts += 1
    except Exception as e:
        print(f"  [Webster] Loi: {e}")
    finally:
        try:
            if not USE_LIBSUMO:
                _traci.switch(label)
            _traci.close()
        except Exception:
            pass
    return {sid: acc[sid] / max(1, counts) for sid in acc}


def eval_random(seed: int) -> dict:
    import random
    rng     = random.Random(seed)
    vec_env = MultiAgentVecEnv(lambda: _make_env())
    vec_env.reset()
    n_agents = vec_env.num_envs
    acc = {}; counts = 0
    while True:
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= SUMO_END:
            break
        jq = _collect_junction_queue(sumo_conn, vec_env.signal_ids)
        for sid, q in jq.items():
            acc[sid] = acc.get(sid, 0.0) + q
        counts += 1
        actions = np.array([rng.randint(0, 3) for _ in range(n_agents)])
        _, _, dones, _ = vec_env.step(actions)
        if dones.any():
            break
    try:
        vec_env.close()
    except Exception:
        pass
    return {sid: acc[sid] / max(1, counts) for sid in acc}


# ── Model discovery: best seed per algo ──────────────────────────────────────

def find_best_model(models_dir: Path, algo: str, seed: str = "s42"):
    """Tim model cho algo + seed (mac dinh seed s42)."""
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parts = subdir.name.split("_")
        if parts[0] != algo or parts[-1] != seed:
            continue
        candidates = (sorted(subdir.glob(f"*_{algo}_final_model.zip")) +
                      sorted(subdir.glob(f"*_{algo}_final_model")))
        if candidates:
            return str(candidates[0]), "_".join(parts[1:-1])  # (path, reward)
    return None, None


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_per_junction(rows: list, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return

    controllers = list(dict.fromkeys(r["controller"] for r in rows))
    junctions   = sorted({r["junction"] for r in rows})

    # Map controller → color
    COLORS = {
        "DQN":       "#1976D2",
        "DDQN":      "#F57C00",
        "PPO":       "#388E3C",
        "Fixed-time":"#757575",
        "Webster":   "#455A64",
        "Random":    "#C62828",
    }

    data = {ctrl: [] for ctrl in controllers}
    for jid in junctions:
        for ctrl in controllers:
            q = next((float(r["mean_queue"]) for r in rows
                      if r["junction"] == jid and r["controller"] == ctrl), 0.0)
            data[ctrl].append(q)

    x   = np.arange(len(junctions))
    n   = len(controllers)
    w   = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(10, len(junctions) * 1.5), 5))
    ax.set_title("Per-Intersection Queue — Cologne8 (7:00–8:00 AM)",
                 fontsize=12, fontweight="bold")

    for i, ctrl in enumerate(controllers):
        offset = (i - n / 2 + 0.5) * w
        ax.bar(x + offset, data[ctrl], w,
               color=COLORS.get(ctrl, "#999"), alpha=0.85, label=ctrl)

    ax.set_xticks(x)
    ax.set_xticklabels([f"J{j}" if str(j).isdigit() else str(j)
                         for j in junctions], fontsize=8)
    ax.set_ylabel("Mean Queue (veh/step)")
    ax.set_xlabel("Junction ID")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.rcParams.update({"figure.dpi": 150})
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

# ── Worker (chay trong subprocess rieng biet) ─────────────────────────────────

def _worker(task: tuple):
    """
    task = ("model",   ctrl_name, model_path, algo)
         | ("fixed",   ctrl_name)
         | ("webster", ctrl_name)
         | ("random",  ctrl_name, seed)
    """
    import os, logging
    logging.basicConfig(level=logging.WARNING, force=True)
    kind = task[0]
    try:
        if kind == "model":
            _, ctrl_name, model_path, algo = task
            jq = eval_model(model_path, algo)
        elif kind == "fixed":
            _, ctrl_name = task
            jq = eval_fixed()
        elif kind == "webster":
            _, ctrl_name = task
            jq = eval_webster()
        elif kind == "random":
            _, ctrl_name, seed = task
            jq = eval_random(seed)
        else:
            raise ValueError(f"Unknown task kind: {kind}")
        return ctrl_name, jq, None
    except Exception as e:
        import traceback
        return task[1], {}, traceback.format_exc()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="./exp_cologne8_real")
    p.add_argument("--save_dir",   default="./results_cologne8_real")
    p.add_argument("--out",        default="./figures")
    p.add_argument("--seed",       default="s42",
                   help="Seed de chon model RL (mac dinh: s42)")
    p.add_argument("--workers",    type=int, default=6,
                   help="So process song song (mac dinh: 6 = 3 RL + 3 baselines)")
    p.add_argument("--skip_rl",    action="store_true")
    return p.parse_args()


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args       = parse_args()
    models_dir = Path(args.models_dir)
    save_dir   = Path(args.save_dir)
    out_dir    = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not C8_NET.exists() or not C8_ROUTE.exists():
        sys.exit(f"[ERR] Thieu net/route Cologne8 tai {C8_NET.parent}")

    print("=" * 60)
    print("  evaluate_per_junction.py — Cologne8")
    print(f"  Models dir : {models_dir}")
    print(f"  Seed       : {args.seed}")
    print(f"  Workers    : {args.workers}")
    print(f"  Sim window : {SUMO_BEGIN}s–{SUMO_END}s")
    print("=" * 60)

    # ── Build task list ──
    tasks = []
    if not args.skip_rl:
        for algo in ["dqn", "ddqn", "ppo"]:
            model_path, reward = find_best_model(models_dir, algo, args.seed)
            if model_path is None:
                print(f"  [WARN] Khong tim thay model: {algo} seed={args.seed}")
                continue
            ctrl_name = f"{algo.upper()} ({args.seed}, {reward})"
            tasks.append(("model", ctrl_name, model_path, algo))
            print(f"  Task: {ctrl_name}  model={Path(model_path).name}")

    tasks += [
        ("fixed",   "Fixed-time"),
        ("webster", "Webster"),
        ("random",  "Random", 0),
    ]

    total   = len(tasks)
    workers = min(args.workers, total)
    print(f"\nChay {total} tasks voi {workers} workers song song...\n")

    label_result = {}
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        fut_map = {pool.submit(_worker, t): t[1] for t in tasks}
        done = 0
        for fut in as_completed(fut_map):
            done += 1
            ctrl_name_expected = fut_map[fut]
            ctrl_name, jq, err = fut.result()
            if err:
                print(f"[{done}/{total}] ERR  {ctrl_name}: {err.splitlines()[-1]}")
            else:
                mean_q = np.mean(list(jq.values())) if jq else 0
                print(f"[{done}/{total}] OK   {ctrl_name}  —  {len(jq)} junctions, mean_queue={mean_q:.2f}")
                label_result[ctrl_name] = jq

    # ── Build rows ──
    all_rows = []
    for ctrl, jq in label_result.items():
        for sid, mq in sorted(jq.items()):
            all_rows.append({"controller": ctrl, "junction": sid,
                              "mean_queue": round(mq, 3)})

    if not all_rows:
        print("[ERR] Khong co ket qua nao")
        return

    # ── Save CSV ──
    csv_path = save_dir / "junction_queue.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["controller", "junction", "mean_queue"])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved: {csv_path}")

    # ── Plot ──
    fig_path = out_dir / "fig6_per_junction_c8.png"
    plot_per_junction(all_rows, fig_path)

    # ── Console summary ──
    print("\n=== Per-junction summary ===")
    junctions = sorted({r["junction"] for r in all_rows})
    print(f"  {'Controller':<30}" +
          "".join(f"  {str(j)[:5]:5s}" for j in junctions))
    for ctrl in label_result:
        vals = [label_result[ctrl].get(j, 0) for j in junctions]
        print(f"  {ctrl:<30}" + "".join(f"  {v:5.1f}" for v in vals))


if __name__ == "__main__":
    main()
