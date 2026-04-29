"""
evaluate_grid.py — Danh gia toan bo model tren grid 2x2 (4 junction)

Su dung:
    python evaluate_grid.py
    python evaluate_grid.py --models_dir ./exp_grid --save_dir ./results_grid --workers 4
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("[evaluate_grid] SUMO_HOME chua duoc khai bao")

_HERE        = Path(__file__).parent
GRID_NET     = _HERE / "nets"            / "grid2x2.net.xml"
GRID_TEST    = _HERE / "generated_flows" / "grid_test_s31415.xml"
EVAL_DURATION = int(os.environ.get("EVAL_DURATION_OVERRIDE", 7_200))

from rl_controller.traffic_env  import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env     import MultiAgentVecEnv
from rl_controller.webster      import DynamicWebsterController

USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

class _TravelTracker:
    def __init__(self):
        self._dep:     dict = {}
        self._tt:      list = []
        self._arrived:  int = 0

    def update(self, sumo_conn):
        t = sumo_conn.simulation.getTime()
        try:
            for vid in sumo_conn.simulation.getDepartedVehiclesIDs():
                self._dep[vid] = t
            for vid in sumo_conn.simulation.getArrivedVehiclesIDs():
                self._arrived += 1
                if vid in self._dep:
                    self._tt.append(t - self._dep.pop(vid))
        except AttributeError:
            pass

    @property
    def throughput(self):        return self._arrived

    @property
    def mean_travel_time(self):
        return float(np.mean(self._tt)) if self._tt else 0.0


def _collect_step_metrics(sumo_conn, signal_ids: list) -> dict:
    all_vehs = sumo_conn.vehicle.getIDList()
    speeds   = [sumo_conn.vehicle.getSpeed(v) for v in all_vehs] if all_vehs else []
    waits    = [sumo_conn.vehicle.getWaitingTime(v) for v in all_vehs] if all_vehs else []

    total_q = 0
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        for lane in lanes:
            try:
                total_q += sumo_conn.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass

    n = max(1, len(all_vehs))
    return {
        "total_queue":       total_q,
        "total_wait":        sum(waits),
        "mean_wait_per_veh": sum(waits) / n,
        "mean_speed":        float(np.mean(speeds)) if speeds else 0.0,
        "vehicles_running":  len(all_vehs),
    }


def _empty_result():
    return {"mean_queue": 0, "mean_wait": 0, "mean_wait_per_veh": 0,
            "mean_speed": 0, "throughput": 0, "mean_travel_time": 0,
            "total_reward": 0}


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------

def run_grid_model_eval(model_path: str, algo: str,
                        net: str, route: str, duration: int) -> dict:
    from stable_baselines3 import PPO, DQN

    AgentCls = PPO if algo == "ppo" else DQN
    agent    = AgentCls.load(model_path, env=None)

    def _make():
        return TrafficControlEnv(
            net_file     = net,
            route_file   = route,
            sim_duration = duration + 500,
            reward_fn    = "queue",
            obs_class    = BaselineObservation,
            single_agent = False,
            show_warnings= False,
        )

    vec_env = MultiAgentVecEnv(_make)
    obs     = vec_env.reset()
    tracker = _TravelTracker()

    metrics_list = []
    total_reward = 0.0
    n_steps      = duration // 5 * 4 + 200

    for _ in range(n_steps):
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= duration:
            break
        tracker.update(sumo_conn)
        m = _collect_step_metrics(sumo_conn, vec_env.signal_ids)
        metrics_list.append(m)

        actions, _ = agent.predict(obs, deterministic=True)
        obs, rews, dones, _ = vec_env.step(actions)
        total_reward += float(rews.mean())
        if dones.any():
            break

    try:
        vec_env.close()
    except Exception:
        pass

    if not metrics_list:
        return _empty_result()

    return {
        "mean_queue":        float(np.mean([m["total_queue"]       for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"]         for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"]         for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      total_reward,
    }


def run_grid_random_eval(n_actions: int, seed: int,
                         net: str, route: str, duration: int) -> dict:
    import random
    rng = random.Random(seed)

    def _make():
        return TrafficControlEnv(
            net_file     = net,
            route_file   = route,
            sim_duration = duration + 500,
            reward_fn    = "queue",
            obs_class    = BaselineObservation,
            single_agent = False,
            show_warnings= False,
        )

    vec_env = MultiAgentVecEnv(_make)
    obs     = vec_env.reset()
    tracker = _TravelTracker()
    metrics_list = []
    n_agents = vec_env.num_envs

    for _ in range(duration // 5 * 4 + 200):
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        if sumo_conn.simulation.getTime() >= duration:
            break
        tracker.update(sumo_conn)
        metrics_list.append(_collect_step_metrics(sumo_conn, vec_env.signal_ids))
        actions = np.array([rng.randint(0, n_actions - 1) for _ in range(n_agents)])
        obs, _, dones, _ = vec_env.step(actions)
        if dones.any():
            break

    try:
        vec_env.close()
    except Exception:
        pass

    if not metrics_list:
        return _empty_result()

    return {
        "mean_queue":        float(np.mean([m["total_queue"]       for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"]         for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"]         for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      0.0,
    }


def run_grid_fixed_eval(net: str, route: str, duration: int) -> dict:
    def _make():
        return TrafficControlEnv(
            net_file     = net,
            route_file   = route,
            sim_duration = duration + 500,
            reward_fn    = "queue",
            obs_class    = BaselineObservation,
            single_agent = False,
            fixed_signal = True,
            show_warnings= False,
        )

    vec_env = MultiAgentVecEnv(_make)
    obs     = vec_env.reset()
    tracker = _TravelTracker()
    metrics_list = []
    n_agents = vec_env.num_envs

    for _ in range(duration // 5 * 4 + 200):
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        if sumo_conn.simulation.getTime() >= duration:
            break
        tracker.update(sumo_conn)
        metrics_list.append(_collect_step_metrics(sumo_conn, vec_env.signal_ids))
        actions = np.zeros(n_agents, dtype=int)
        obs, _, dones, _ = vec_env.step(actions)
        if dones.any():
            break

    try:
        vec_env.close()
    except Exception:
        pass

    if not metrics_list:
        return _empty_result()

    return {
        "mean_queue":        float(np.mean([m["total_queue"]       for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"]         for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"]         for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      0.0,
    }


def run_grid_webster_eval(net: str, route: str, duration: int) -> dict:
    import sumolib
    import traci as _traci

    label = f"wsb_grid_{int(time.time()*1000) % 100000}"
    cmd = [
        sumolib.checkBinary("sumo"),
        "-n", net,
        "-r", route,
        "--time-to-teleport", "-1",
        "--no-warnings",
    ]
    if USE_LIBSUMO:
        _traci.start(cmd)
        sumo = _traci
    else:
        _traci.start(cmd, label=label)
        sumo = _traci.getConnection(label)

    signal_ids = list(sumo.trafficlight.getIDList())
    tracker    = _TravelTracker()

    def get_green_phases(ts):
        logics = sumo.trafficlight.getAllProgramLogics(ts)
        if not logics:
            return [0]
        return [i for i, ph in enumerate(logics[0].phases)
                if "G" in ph.state.upper() and "Y" not in ph.state.upper()]

    controllers = {
        ts: DynamicWebsterController(ts, get_green_phases(ts), sumo)
        for ts in signal_ids
    }

    metrics_list = []
    try:
        while sumo.simulation.getTime() < duration:
            sumo.simulationStep()
            t = sumo.simulation.getTime()
            tracker.update(sumo)
            for ctrl in controllers.values():
                ctrl.step(t)
            metrics_list.append(_collect_step_metrics(sumo, signal_ids))
    except Exception as exc:
        print(f"  [Webster-grid] Loi: {exc}")
    finally:
        try:
            if not USE_LIBSUMO:
                _traci.switch(label)
            _traci.close()
        except Exception:
            pass

    if not metrics_list:
        return _empty_result()

    return {
        "mean_queue":        float(np.mean([m["total_queue"]       for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"]         for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"]         for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      0.0,
    }


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_grid_models(models_dir: Path) -> List[Tuple[str, str, str, str]]:
    """Returns list of (algo, reward, seed, model_path)."""
    results = []
    if not models_dir.exists():
        return results

    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        parts = name.split("_")
        if len(parts) < 3:
            continue

        algo = parts[0]
        seed = parts[-1]
        middle = [p for p in parts[1:-1]]
        reward = "_".join(middle)

        if algo == "ppo":
            model_path = subdir / "grid_ppo_final_model.zip"
        elif algo in ("dqn", "ddqn"):
            model_path = subdir / f"grid_{algo}_final_model.zip"
        else:
            continue

        if model_path.exists():
            results.append((algo, reward, seed, str(model_path)))

    return results


# ---------------------------------------------------------------------------
# Worker wrapper
# ---------------------------------------------------------------------------

def _worker(task):
    kind, args = task
    net, route, dur = args["net"], args["route"], args["duration"]

    if kind == "model":
        m = run_grid_model_eval(args["model_path"], args["algo"], net, route, dur)
        row = {"algo": args["algo"], "reward": args["reward"],
               "seed": args["seed"], **m}
    elif kind == "random":
        m = run_grid_random_eval(4, args["seed"], net, route, dur)
        row = {"algo": "random", "reward": "-", "seed": str(args["seed"]), **m}
    elif kind == "fixed":
        m = run_grid_fixed_eval(net, route, dur)
        row = {"algo": "fixed", "reward": "-", "seed": "-", **m}
    elif kind == "webster":
        m = run_grid_webster_eval(net, route, dur)
        row = {"algo": "webster", "reward": "-", "seed": "-", **m}

    return row, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="./exp_grid")
    p.add_argument("--save_dir",   default="./results_grid")
    p.add_argument("--skip_fixed",   action="store_true")
    p.add_argument("--skip_random",  action="store_true")
    p.add_argument("--skip_webster", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def main():
    args       = parse_args()
    models_dir = Path(args.models_dir)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not GRID_NET.exists():
        sys.exit(f"[ERR] Thieu file mang: {GRID_NET}")
    if not GRID_TEST.exists():
        sys.exit(f"[ERR] Thieu file flow: {GRID_TEST}")

    models = discover_grid_models(models_dir)
    print(f"\nTim thay {len(models)} grid model(s) trong {models_dir}")
    print(f"Test flow : {GRID_TEST.name}")
    print(f"Eval time : {EVAL_DURATION}s\n")

    _base = {"net": str(GRID_NET), "route": str(GRID_TEST), "duration": EVAL_DURATION}
    tasks, labels = [], []

    if not args.skip_random:
        tasks.append(("random", {**_base, "seed": 0}))
        labels.append("Random baseline (grid)")
    if not args.skip_fixed:
        tasks.append(("fixed", {**_base}))
        labels.append("Fixed-time baseline (grid)")
    if not args.skip_webster:
        tasks.append(("webster", {**_base}))
        labels.append("Webster baseline (grid)")
    for algo, reward, seed, model_path in models:
        tasks.append(("model", {**_base, "algo": algo, "reward": reward,
                                "seed": seed, "model_path": model_path}))
        labels.append(f"{algo.upper()} reward={reward} seed={seed}")

    total  = len(tasks)
    t0     = time.time()
    fields = ["algo", "reward", "seed",
              "mean_queue", "mean_wait", "mean_wait_per_veh", "mean_speed",
              "throughput", "mean_travel_time", "total_reward"]
    all_rows = []
    workers  = max(1, args.workers)
    print(f"Chay {total} jobs voi {workers} worker(s)...\n")

    if workers == 1:
        for idx, (task, label) in enumerate(zip(tasks, labels), 1):
            print(f"[{idx}/{total}] {label}")
            try:
                row, m = _worker(task)
                all_rows.append(row)
                print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}"
                      f"  speed={m['mean_speed']:.3f}")
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"  [ERR] {exc}")
    else:
        import multiprocessing
        ctx  = multiprocessing.get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            fut_map = {pool.submit(_worker, t): lbl
                       for t, lbl in zip(tasks, labels)}
            for fut in as_completed(fut_map):
                done += 1
                lbl   = fut_map[fut]
                try:
                    row, m = fut.result()
                    all_rows.append(row)
                    print(f"[{done}/{total}] DONE {lbl}")
                    print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}"
                          f"  speed={m['mean_speed']:.3f}")
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    print(f"[{done}/{total}] ERR  {lbl}: {exc}")

    # Write CSV
    all_csv = save_dir / "all_results.csv"
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)

    # Summary
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["algo"], row["reward"])].append(row)

    sum_rows = []
    for (algo, reward), rows in sorted(grouped.items()):
        def _mean(key): return round(float(np.mean([float(r[key]) for r in rows])), 3)
        def _std(key):  return round(float(np.std([float(r[key]) for r in rows])), 3)
        sum_rows.append({
            "algo": algo, "reward": reward, "n_runs": len(rows),
            "mean_queue": _mean("mean_queue"), "std_queue": _std("mean_queue"),
            "mean_wait":  _mean("mean_wait"),  "std_wait":  _std("mean_wait"),
            "mean_speed": _mean("mean_speed"), "std_speed": _std("mean_speed"),
            "mean_throughput":  _mean("throughput"),
            "mean_travel_time": _mean("mean_travel_time"),
        })

    sum_csv = save_dir / "summary.csv"
    sum_fields = ["algo", "reward", "n_runs",
                  "mean_queue", "std_queue", "mean_wait", "std_wait",
                  "mean_speed", "std_speed", "mean_throughput", "mean_travel_time"]
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader(); w.writerows(sum_rows)

    elapsed = round((time.time() - t0) / 60, 1)
    print(f"\n{'='*55}")
    print(f"  HOAN TAT ({elapsed} phut)")
    print(f"  Chi tiet : {all_csv}")
    print(f"  Tom tat  : {sum_csv}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
