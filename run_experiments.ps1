# run_experiments.ps1 — Chay toan bo thi nghiem nhieu seeds
#
# Su dung:
#   .\run_experiments.ps1
#
# Chay 4 reward PPO x 3 seeds = 12 runs PPO (~10 gio)
#         1 DQN        x 3 seeds =  3 runs DQN (~3 gio)
# Tong: 15 runs, khoang 13 gio (chay qua dem)
#
# Output:
#   experiments/ppo_queue_s42/
#   experiments/ppo_queue_s123/
#   experiments/ppo_diff-waiting-time_s42/
#   ... (etc)
#   experiments/dqn_s42/
#   experiments/dqn_s123/
#   experiments/dqn_s777/

# Dung PowerShell strict mode
$ErrorActionPreference = "Stop"

$Python = ".venv\Scripts\python.exe"
$ExpDir = "experiments"

$PPORewards = @("queue", "diff-waiting-time", "pressure", "average-speed")
$Seeds      = @(42, 123, 777)
$TotalSteps = 200000

# -------------------------------------------------------
# Kiem tra moi truong
# -------------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Error "Khong tim thay $Python. Hay kich hoat virtual environment."
    exit 1
}
if (-not (Test-Path "sumo_nets\grid2x2.net.xml")) {
    Write-Host "[SETUP] Tao mang luoi SUMO 2x2..." -ForegroundColor Yellow
    & $Python sumo_nets\generate_net.py
}
if (-not (Test-Path "sumo_nets\grid2x2_test_high.rou.xml")) {
    Write-Host "[SETUP] Tao route files 3 muc luu luong..." -ForegroundColor Yellow
    & $Python sumo_nets\generate_test_flows.py
}

New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null

$TotalRuns = $PPORewards.Count * $Seeds.Count + $Seeds.Count
$RunIdx = 0
$StartTime = Get-Date

# -------------------------------------------------------
# Ham ghi log
# -------------------------------------------------------
function Write-Progress-Header($msg) {
    $elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
    Write-Host ""
    Write-Host "[$RunIdx/$TotalRuns] [$elapsed min] $msg" -ForegroundColor Cyan
    Write-Host ("-" * 55) -ForegroundColor DarkGray
}

# -------------------------------------------------------
# Huan luyen PPO (4 rewards x 3 seeds)
# -------------------------------------------------------
Write-Host ""
Write-Host "=" * 55 -ForegroundColor Green
Write-Host "  BOT DAU HUAN LUYEN PPO — $($PPORewards.Count) rewards x $($Seeds.Count) seeds" -ForegroundColor Green
Write-Host "=" * 55 -ForegroundColor Green

foreach ($reward in $PPORewards) {
    foreach ($seed in $Seeds) {
        $RunIdx++
        $saveDir = "$ExpDir\ppo_${reward}_s${seed}"

        if (Test-Path "$saveDir\ppo_final_model.zip") {
            Write-Host "[$RunIdx/$TotalRuns] [SKIP] $saveDir da co model." -ForegroundColor DarkGray
            continue
        }

        Write-Progress-Header "PPO | reward=$reward | seed=$seed"
        & $Python train_ppo.py --save_dir $saveDir --mode multi --reward_type $reward --seed $seed --total_steps $TotalSteps --obs_mode raw

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [WARN] Train that bai (exit $LASTEXITCODE). Tiep tuc..." -ForegroundColor Yellow
        } else {
            Write-Host "  [OK] Luu model: $saveDir\ppo_final_model.zip" -ForegroundColor Green
        }
    }
}

# -------------------------------------------------------
# Huan luyen DQN (1 config x 3 seeds)
# -------------------------------------------------------
Write-Host ""
Write-Host "=" * 55 -ForegroundColor Magenta
Write-Host "  BOT DAU HUAN LUYEN RESCO-DQN — $($Seeds.Count) seeds" -ForegroundColor Magenta
Write-Host "=" * 55 -ForegroundColor Magenta

foreach ($seed in $Seeds) {
    $RunIdx++
    $saveDir = "$ExpDir\dqn_s${seed}"

    if (Test-Path "$saveDir\dqn_final_model.zip") {
        Write-Host "[$RunIdx/$TotalRuns] [SKIP] $saveDir da co model." -ForegroundColor DarkGray
        continue
    }

    Write-Progress-Header "DQN | seed=$seed"
    & $Python train_dqn.py --save_dir $saveDir --mode multi --seed $seed --total_steps $TotalSteps

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] DQN train that bai (exit $LASTEXITCODE). Tiep tuc..." -ForegroundColor Yellow
    } else {
        Write-Host "  [OK] Luu model: $saveDir\dqn_final_model.zip" -ForegroundColor Green
    }
}

# -------------------------------------------------------
# Ket thuc
# -------------------------------------------------------
$elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Write-Host ""
Write-Host "=" * 55 -ForegroundColor Green
Write-Host "  HOAN TAT HUAN LUYEN" -ForegroundColor Green
Write-Host "  Tong thoi gian: $elapsed phut" -ForegroundColor Green
Write-Host "  Ket qua trong: $ExpDir\" -ForegroundColor Green
Write-Host '=' * 55 -ForegroundColor Green
Write-Host ''
Write-Host 'Buoc tiep theo — Danh gia toan bo:' -ForegroundColor Yellow
Write-Host '  python evaluate_all.py --models_dir ./experiments --save_dir ./results' -ForegroundColor White
