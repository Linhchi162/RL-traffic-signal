import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EXP_DIR = Path("experiments")

PPO_REWARDS = ["queue", "diff-waiting-time", "pressure", "average-speed"]
SEEDS       = [42, 123, 777]
TOTAL_STEPS = 200_000
PYTHON_EXEC = sys.executable

# Số luồng chạy song song — chỉnh xuống 1 nếu máy lag.
MAX_WORKERS = 3


def run_cmd(task_name, cmd):
    print(f"  [START] {task_name}")
    result = subprocess.run(cmd, check=False)
    return task_name, result.returncode


def main():
    EXP_DIR.mkdir(exist_ok=True)
    start_time = time.time()
    tasks = []

    # PPO — single intersection, all reward types
    for reward in PPO_REWARDS:
        for seed in SEEDS:
            save_dir   = EXP_DIR / f"ppo_{reward}_single_s{seed}"
            model_file = save_dir / "ppo_final_model.zip"
            if not model_file.exists():
                cmd = [
                    PYTHON_EXEC, "train_ppo.py",
                    "--save_dir",    str(save_dir),
                    "--reward_type", reward,
                    "--seed",        str(seed),
                    "--total_steps", str(TOTAL_STEPS),
                    "--obs_mode",    "raw",
                ]
                tasks.append((f"PPO | reward={reward} | seed={seed}", cmd))

    # DQN / DDQN — single intersection
    for algo in ["dqn", "ddqn"]:
        for reward in ["wait-clip", "queue", "pressure"]:
            for seed in SEEDS:
                save_dir   = EXP_DIR / f"{algo}_{reward}_s{seed}"
                model_file = save_dir / "dqn_final_model.zip"
                if not model_file.exists():
                    cmd = [
                        PYTHON_EXEC, "train_dqn.py",
                        "--save_dir",    str(save_dir),
                        "--algo",        algo,
                        "--reward_type", reward,
                        "--seed",        str(seed),
                        "--total_steps", str(TOTAL_STEPS),
                    ]
                    tasks.append((f"{algo.upper()} | reward={reward} | seed={seed}", cmd))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("Tat ca cac kich ban deu da hoan tat!")
        print("Chay danh gia: python evaluate_all.py --models_dir ./experiments --save_dir ./results")
        return

    print("\n" + "=" * 65)
    print(f"BAT DAU HUAN LUYEN DA LUONG (MAX_WORKERS = {MAX_WORKERS})")
    print(f"Tong so kich ban: {total_tasks}")
    print("=" * 65)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(run_cmd, name, cmd): name
            for (name, cmd) in tasks
        }
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                name, returncode = future.result()
                completed += 1
                elapsed_min = round((time.time() - start_time) / 60, 1)
                status = "OK" if returncode == 0 else f"ERROR (ma {returncode})"
                print(f"[{completed}/{total_tasks}] [{elapsed_min} min] XONG: {name} — {status}")
            except Exception as exc:
                print(f"[LOI] {task_name}: {exc}")

    elapsed_min = round((time.time() - start_time) / 60, 1)
    print("\n" + "=" * 65)
    print(f"  HOAN TAT ({elapsed_min} phut)")
    print("=" * 65)
    print("\nBuoc tiep theo:")
    print("  python evaluate_all.py --models_dir ./experiments --save_dir ./results")


if __name__ == "__main__":
    main()
