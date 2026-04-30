"""
evaluate_cologne8.py — Zero-shot evaluation tren Cologne8 (8 giao lo that, CHLB Duc)

Map + traffic demand tu RESCO benchmark (Pi-Star-Lab/RESCO).
Thoi gian mo phong: 25200s-28800s (7am-8am gio cao diem, 3600s thuc te).

Model duoc train tren grid2x2 (7D obs, Discrete(4)) se duoc chay truc tiep
tren mang that nay de kiem tra kha nang generalize (zero-shot transfer).

Su dung:
    python evaluate_cologne8.py
    python evaluate_cologne8.py --models_dir ./exp_grid --save_dir ./results_cologne8 --workers 4
    python evaluate_cologne8.py --skip_models   # chi chay baselines
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("[evaluate_cologne8] SUMO_HOME chua duoc khai bao")

_HERE    = Path(__file__).parent
C8_NET   = _HERE / "nets" / "cologne8" / "cologne8.net.xml"
C8_ROUTE = _HERE / "nets" / "cologne8" / "cologne8.rou.xml"

# Cologne8 vehicles depart during this window (real Cologne peak hour)
SUMO_BEGIN = 25200   # 7:00 AM
SUMO_END   = 28800   # 8:00 AM
_EXTRA_ARGS = f"--begin {SUMO_BEGIN}"

from rl_controller.traffic_env   import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env      import MultiAgentVecEnv
from rl_controller.webster       import DynamicWebsterController

USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

class _TravelTracker:
    def __init__(self):
        self._dep:      dict = {}
        self._tt:       list = []
        self._arrived:  int  = 0
        self._prev_ids: set  = set()

    def update(self, sumo_conn):
        t = sumo_conn.simulation.getTime()
        try:
            current_ids = set(sumo_conn.vehicle.getIDList())
            for vid in current_ids - self._prev_ids:
                self._dep[vid] = t
            for vid in self._prev_ids - current_ids:
                self._arrived += 1
                if vid in self._dep:
                    self._tt.append(t - self._dep.pop(vid))
            self._prev_ids = current_ids
        except AttributeError:
            pass

    @property
    def throughput(self):     return self._arrived

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
    }


def _empty_result():
    return {"mean_queue": 0, "mean_wait": 0, "mean_wait_per_veh": 0,
            "mean_speed": 0, "throughput": 0, "mean_travel_time": 0,
            "total_reward": 0}


def _make_env(fixed_signal=False):
    """Factory tra ve TrafficControlEnv voi Cologne8 net/route."""
    return TrafficControlEnv(
        net_file        = str(C8_NET),
        route_file      = str(C8_ROUTE),
        sim_duration    = SUMO_END + 99999, # lon hon SUMO_END de done khong bao gio trigger
                                            # → tranh auto-reset lam libsumo hang
        reward_fn       = "queue",
        obs_class       = BaselineObservation,
        single_agent    = False,
        fixed_signal    = fixed_signal,
        show_warnings   = False,
        extra_sumo_args = _EXTRA_ARGS,     # SUMO bat dau tu 25200s
    )


def _run_loop(vec_env, get_actions, tracker):
    """Vong lap mo phong chung: tra ve (metrics_list, total_reward)."""
    metrics_list = []
    total_reward = 0.0

    while True:
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= SUMO_END:
            break
        tracker.update(sumo_conn)
        metrics_list.append(_collect_step_metrics(sumo_conn, vec_env.signal_ids))

        actions = get_actions(vec_env)
        obs_out, rews, dones, _ = vec_env.step(actions)
        total_reward += float(rews.mean())
        if dones.any():
            break

    try:
        vec_env.close()
    except Exception:
        pass

    return metrics_list, total_reward


def _aggregate(metrics_list, tracker, total_reward):
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


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------

def run_model_eval(model_path: str, algo: str) -> dict:
    from stable_baselines3 import PPO, DQN
    agent = (PPO if algo == "ppo" else DQN).load(model_path, env=None)

    vec_env = MultiAgentVecEnv(lambda: _make_env())
    obs     = vec_env.reset()
    tracker = _TravelTracker()

    # Closure de get_actions co the dung obs hien tai
    _obs_holder = [obs]

    def get_actions(ve):
        acts, _ = agent.predict(_obs_holder[0], deterministic=True)
        return acts

    # Override vong lap de cap nhat obs
    metrics_list = []
    total_reward = 0.0
    while True:
        sumo_conn = vec_env.env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= SUMO_END:
            break
        tracker.update(sumo_conn)
        metrics_list.append(_collect_step_metrics(sumo_conn, vec_env.signal_ids))
        actions, _ = agent.predict(_obs_holder[0], deterministic=True)
        _obs_holder[0], rews, dones, _ = vec_env.step(actions)
        total_reward += float(rews.mean())
        if dones.any():
            break

    try:
        vec_env.close()
    except Exception:
        pass

    return _aggregate(metrics_list, tracker, total_reward)


def run_random_eval(seed: int) -> dict:
    import random
    rng = random.Random(seed)

    vec_env = MultiAgentVecEnv(lambda: _make_env())
    vec_env.reset()
    tracker  = _TravelTracker()
    n_agents = vec_env.num_envs

    metrics_list, total_reward = _run_loop(
        vec_env,
        lambda ve: np.array([rng.randint(0, 3) for _ in range(n_agents)]),
        tracker,
    )
    return _aggregate(metrics_list, tracker, total_reward)


def run_fixed_eval() -> dict:
    vec_env = MultiAgentVecEnv(lambda: _make_env(fixed_signal=True))
    vec_env.reset()
    tracker  = _TravelTracker()
    n_agents = vec_env.num_envs

    metrics_list, total_reward = _run_loop(
        vec_env,
        lambda ve: np.zeros(n_agents, dtype=int),
        tracker,
    )
    return _aggregate(metrics_list, tracker, total_reward)


def run_webster_eval() -> dict:
    import sumolib
    import traci as _traci

    label = f"wsb_c8_{int(time.time()*1000) % 100000}"
    cmd = [
        sumolib.checkBinary("sumo"),
        "-n", str(C8_NET),
        "-r", str(C8_ROUTE),
        "--begin", str(SUMO_BEGIN),
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
        while sumo.simulation.getTime() < SUMO_END:
            sumo.simulationStep()
            t = sumo.simulation.getTime()
            tracker.update(sumo)
            for ctrl in controllers.values():
                ctrl.step(t)
            metrics_list.append(_collect_step_metrics(sumo, signal_ids))
    except Exception as exc:
        print(f"  [Webster-c8] Loi: {exc}")
    finally:
        try:
            if not USE_LIBSUMO:
                _traci.switch(label)
            _traci.close()
        except Exception:
            pass

    return _aggregate(metrics_list, tracker, 0.0)


# ---------------------------------------------------------------------------
# Model discovery (reuse logic tu evaluate_grid.py)
# ---------------------------------------------------------------------------

def discover_grid_models(models_dir: Path):
    results = []
    if not models_dir.exists():
        return results
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parts = subdir.name.split("_")
        if len(parts) < 3:
            continue
        algo   = parts[0]
        seed   = parts[-1]
        reward = "_".join(parts[1:-1])
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
# Worker
# ---------------------------------------------------------------------------

def _worker(task):
    kind, args = task
    if kind == "model":
        m   = run_model_eval(args["model_path"], args["algo"])
        row = {"algo": args["algo"], "reward": args["reward"], "seed": args["seed"], **m}
    elif kind == "random":
        m   = run_random_eval(args["seed"])
        row = {"algo": "random", "reward": "-", "seed": str(args["seed"]), **m}
    elif kind == "fixed":
        m   = run_fixed_eval()
        row = {"algo": "fixed",  "reward": "-", "seed": "-", **m}
    elif kind == "webster":
        m   = run_webster_eval()
        row = {"algo": "webster","reward": "-", "seed": "-", **m}
    return row, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir",    default="./exp_grid")
    p.add_argument("--save_dir",      default="./results_cologne8")
    p.add_argument("--skip_fixed",    action="store_true")
    p.add_argument("--skip_random",   action="store_true")
    p.add_argument("--skip_webster",  action="store_true")
    p.add_argument("--skip_models",   action="store_true")
    p.add_argument("--workers",       type=int, default=1)
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


def main():
    args       = parse_args()
    models_dir = Path(args.models_dir)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not C8_NET.exists():
        sys.exit(f"[ERR] Thieu file mang: {C8_NET}\n"
                 f"      Chay: bash setup_vast.sh  (se tu download)")
    if not C8_ROUTE.exists():
        sys.exit(f"[ERR] Thieu file route: {C8_ROUTE}")

    models = discover_grid_models(models_dir)
    print(f"\nCologne8 zero-shot evaluation")
    print(f"Net   : {C8_NET.name}  |  Route: {C8_ROUTE.name}")
    print(f"Window: {SUMO_BEGIN}s-{SUMO_END}s ({(SUMO_END-SUMO_BEGIN)//60} phut peak hour)")
    print(f"Models: {len(models)} grid model(s) tu {models_dir}\n")

    _base  = {}
    fields = ["algo", "reward", "seed",
              "mean_queue", "mean_wait", "mean_wait_per_veh", "mean_speed",
              "throughput", "mean_travel_time", "total_reward"]
    all_rows  = []
    tasks     = []
    labels    = []

    all_csv   = save_dir / "all_results.csv"
    done_keys: set = set()
    if args.resume and all_csv.exists():
        with open(all_csv, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                done_keys.add((row["algo"], row["reward"], row["seed"]))
        print(f"[--resume] Da co {len(all_rows)} ket qua, se bo qua.\n")

    if not args.skip_random and ("random", "-", "0") not in done_keys:
        tasks.append(("random", {"seed": 0}));  labels.append("Random baseline")
    if not args.skip_fixed and ("fixed", "-", "-") not in done_keys:
        tasks.append(("fixed",  {}));            labels.append("Fixed-time baseline")
    if not args.skip_webster and ("webster", "-", "-") not in done_keys:
        tasks.append(("webster", {}));           labels.append("Webster baseline")
    if not args.skip_models:
        for algo, reward, seed, model_path in models:
            if (algo, reward, seed) in done_keys:
                continue
            tasks.append(("model", {"algo": algo, "reward": reward,
                                    "seed": seed, "model_path": model_path}))
            labels.append(f"{algo.upper()} reward={reward} seed={seed}")

    total   = len(tasks)
    workers = max(1, args.workers)
    t0      = time.time()
    print(f"Chay {total} jobs voi {workers} worker(s)...\n")

    if workers == 1:
        for idx, (task, label) in enumerate(zip(tasks, labels), 1):
            print(f"[{idx}/{total}] {label}")
            try:
                row, m = _worker(task)
                all_rows.append(row)
                print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}"
                      f"  speed={m['mean_speed']:.3f}  throughput={m['throughput']}")
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"  [ERR] {exc}")
    else:
        import multiprocessing
        ctx  = multiprocessing.get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            fut_map = {pool.submit(_worker, t): lbl for t, lbl in zip(tasks, labels)}
            for fut in as_completed(fut_map):
                done += 1
                lbl = fut_map[fut]
                try:
                    row, m = fut.result()
                    all_rows.append(row)
                    print(f"[{done}/{total}] DONE {lbl}")
                    print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}"
                          f"  speed={m['mean_speed']:.3f}  throughput={m['throughput']}")
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    print(f"[{done}/{total}] ERR  {lbl}: {exc}")

    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)

    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["algo"], row["reward"])].append(row)

    sum_rows = []
    for (algo, reward), rows in sorted(grouped.items()):
        def _mean(key): return round(float(np.mean([float(r[key]) for r in rows])), 3)
        def _std(key):  return round(float(np.std( [float(r[key]) for r in rows])), 3)
        sum_rows.append({
            "algo": algo, "reward": reward, "n_runs": len(rows),
            "mean_queue": _mean("mean_queue"), "std_queue": _std("mean_queue"),
            "mean_wait":  _mean("mean_wait"),  "std_wait":  _std("mean_wait"),
            "mean_speed": _mean("mean_speed"), "std_speed": _std("mean_speed"),
            "mean_throughput":  _mean("throughput"),
            "mean_travel_time": _mean("mean_travel_time"),
        })

    sum_csv    = save_dir / "summary.csv"
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
