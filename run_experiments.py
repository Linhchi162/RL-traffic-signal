import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thư mục lưu kết quả
EXP_DIR = Path("experiments")

# Cấu hình huấn luyện
PPO_REWARDS = ["queue", "diff-waiting-time", "pressure", "average-speed"]
AE_DIMS = ["4d", "8d", "16d", "19d", "32d"]
SEEDS = [42, 123, 777]
TOTAL_STEPS = 200_000
PYTHON_EXEC = sys.executable

# SỐ LUỒNG CHẠY SONG SONG
# Nếu máy tính quá lag, bạn có thể chỉnh xuống 2 hoặc 1.
MAX_WORKERS = 3

def run_cmd(task_name, cmd):
    """Chạy command và block cho đến khi hoàn thành."""
    print(f"  [START] {task_name}")
    result = subprocess.run(cmd, check=False)
    return task_name, result.returncode

def main():
    EXP_DIR.mkdir(exist_ok=True)
    start_time = time.time()
    
    tasks = []

    # 1. PPO Training
    for reward in PPO_REWARDS:
        for seed in SEEDS:
            save_dir = EXP_DIR / f"ppo_{reward}_s{seed}"
            model_file = save_dir / "ppo_final_model.zip"
            if not model_file.exists():
                cmd = [
                    PYTHON_EXEC, "train_ppo.py",
                    "--save_dir", str(save_dir),
                    "--mode", "multi",
                    "--reward_type", reward,
                    "--seed", str(seed),
                    "--total_steps", str(TOTAL_STEPS),
                    "--obs_mode", "raw"
                ]
                tasks.append((f"PPO | Reward = {reward} | Seed = {seed}", cmd))

    # 2. DQN Training
    for seed in SEEDS:
        save_dir = EXP_DIR / f"dqn_s{seed}"
        model_file = save_dir / "dqn_final_model.zip"
        if not model_file.exists():
            cmd = [
                PYTHON_EXEC, "train_dqn.py",
                "--save_dir", str(save_dir),
                "--mode", "multi",
                "--seed", str(seed),
                "--total_steps", str(TOTAL_STEPS)
            ]
            tasks.append((f"DQN | Seed = {seed}", cmd))

    # 3. PPO Training with Autoencoders
    for ae_dim in AE_DIMS:
        for seed in SEEDS:
            save_dir = EXP_DIR / f"ppo_queue-AE-{ae_dim}_s{seed}"
            model_file = save_dir / "ppo_final_model.zip"
            if not model_file.exists():
                cmd = [
                    PYTHON_EXEC, "train_ppo.py",
                    "--save_dir", str(save_dir),
                    "--mode", "multi",
                    "--reward_type", "queue",
                    "--seed", str(seed),
                    "--total_steps", str(TOTAL_STEPS),
                    "--obs_mode", f"compressed_{ae_dim}"
                ]
                tasks.append((f"PPO AE | Dim = {ae_dim} | Seed = {seed}", cmd))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("Tat ca cac kịch ban deu da hoan tat! Ban co the chay evaluate_all.py")
        return

    print("\n" + "=" * 65)
    print(f"BAT DAU HUAN LUYEN DA LUONG (MAX_WORKERS = {MAX_WORKERS})")
    print(f"Tong so kich ban can chay: {total_tasks}")
    print("WARNING: May tinh se hoat dong het cong suat. Khong tat file nay!")
    print("=" * 65)

    completed = 0
    # Khởi chạy ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit toàn bộ job
        future_to_task = {
            executor.submit(run_cmd, name, cmd): name 
            for (name, cmd) in tasks
        }
        
        # Nhận kết quả khi complete
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                name, returncode = future.result()
                completed += 1
                elapsed_min = round((time.time() - start_time) / 60, 1)
                status = "OK" if returncode == 0 else f"ERROR (mã {returncode})"
                print(f"[{completed}/{total_tasks}] [{elapsed_min} min] XONG: {name} - Status: {status}")
            except Exception as exc:
                print(f"[LOI_CHUNG] {task_name} tao ra loi: {exc}")

    elapsed_min = round((time.time() - start_time) / 60, 1)
    print("\n" + "=" * 65)
    print("  HOAN TAT TOAN BO HUAN LUYEN (DA LUONG)")
    print(f"  Tong thoi gian: {elapsed_min} phut")
    print("=" * 65)
    print("\nBuoc tiep theo: Giai doan danh gia")
    print("  python evaluate_all.py --models_dir ./experiments --save_dir ./results")

if __name__ == "__main__":
    main()
