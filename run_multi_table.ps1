# run_multi_table.ps1 - Train toan bo bang ket qua (multi-intersection 2x2)
#
# DQN  x {queue, pressure, wait-clip} x {42, 123, 777}  =  9 runs
# DDQN x {queue, pressure, wait-clip} x {42, 123, 777}  =  9 runs
# PPO  x {queue, pressure, wait-clip} x {42, 123, 777}  =  9 runs  (obs=baseline, 7D)
# Tong: 27 runs
#
# Su dung:
#   .\run_multi_table.ps1          # mac dinh 3 jobs song song
#   .\run_multi_table.ps1 -Jobs 2  # 2 jobs neu it RAM

param(
    [int]$Jobs = 3
)

$ErrorActionPreference = "Stop"
$Python   = (Resolve-Path ".venv\Scripts\python.exe").Path
$ExpDir   = (Resolve-Path ".").Path + "\experiments"
$LogDir   = (Resolve-Path ".").Path + "\logs_parallel"
$WorkDir  = (Resolve-Path ".").Path

New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Seeds      = @(42, 123, 777)
$TotalSteps = 200000
$DQNRewards = @("queue", "pressure", "wait-clip")
$PPORewards = @("queue", "pressure", "wait-clip")

# ---------------------------------------------------------------------------
# Tao danh sach tat ca runs
# ---------------------------------------------------------------------------
$AllRuns = @()

foreach ($seed in $Seeds) {
    foreach ($reward in $DQNRewards) {
        # DQN
        $sd = "$ExpDir\dqn_${reward}_s${seed}"
        if (-not (Test-Path "$sd\dqn_final_model.zip")) {
            $AllRuns += [PSCustomObject]@{ Algo="dqn"; Reward=$reward; Seed=$seed; SaveDir=$sd }
        } else {
            Write-Host "[SKIP] dqn_${reward}_s${seed}" -ForegroundColor DarkGray
        }

        # DDQN
        $sd = "$ExpDir\ddqn_${reward}_s${seed}"
        if (-not (Test-Path "$sd\dqn_final_model.zip")) {
            $AllRuns += [PSCustomObject]@{ Algo="ddqn"; Reward=$reward; Seed=$seed; SaveDir=$sd }
        } else {
            Write-Host "[SKIP] ddqn_${reward}_s${seed}" -ForegroundColor DarkGray
        }
    }

    foreach ($reward in $PPORewards) {
        # PPO 7D (baseline obs)
        $sd = "$ExpDir\ppo_${reward}_baseline_s${seed}"
        if (-not (Test-Path "$sd\ppo_final_model.zip")) {
            $AllRuns += [PSCustomObject]@{ Algo="ppo"; Reward=$reward; Seed=$seed; SaveDir=$sd }
        } else {
            Write-Host "[SKIP] ppo_${reward}_baseline_s${seed}" -ForegroundColor DarkGray
        }
    }
}

if ($AllRuns.Count -eq 0) {
    Write-Host "Tat ca models da duoc train." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  Train $($AllRuns.Count) models | $Jobs jobs song song" -ForegroundColor Cyan
Write-Host "  DQN/DDQN: 3 rewards x 3 seeds | PPO 7D: 3 rewards x 3 seeds" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

$StartTime  = Get-Date
$ActiveJobs = @()
$RunIdx     = 0
$DoneCount  = 0

function Wait-OneJob {
    while ($true) {
        $finished = $ActiveJobs | Where-Object { $_.Job.State -in "Completed","Failed","Stopped" }
        if ($finished) {
            foreach ($item in $finished) {
                Receive-Job $item.Job -ErrorAction SilentlyContinue | Out-Null
                $status = if ($item.Job.State -eq "Completed") { "[OK]  " } else { "[FAIL]" }
                $color  = if ($item.Job.State -eq "Completed") { "Green" } else { "Red" }
                Write-Host "$status $($item.Label)" -ForegroundColor $color
                Remove-Job $item.Job -Force
                $script:ActiveJobs = @($script:ActiveJobs | Where-Object { $_.Job.Id -ne $item.Job.Id })
                $script:DoneCount++
            }
            return
        }
        Start-Sleep -Seconds 15
    }
}

foreach ($run in $AllRuns) {
    while ($ActiveJobs.Count -ge $Jobs) { Wait-OneJob }

    $RunIdx++
    $label   = "$($run.Algo)_$($run.Reward)_s$($run.Seed)"
    $logFile = "$LogDir\${label}.log"
    Write-Host "[$RunIdx/$($AllRuns.Count)] Start: $label" -ForegroundColor Cyan

    $algo    = $run.Algo
    $reward  = $run.Reward
    $seed    = $run.Seed
    $saveDir = $run.SaveDir

    if ($algo -eq "ppo") {
        $job = Start-Job -ScriptBlock {
            param($py, $sd, $rw, $s, $ts, $log, $wd)
            Set-Location $wd
            $env:SUMO_HOME            = "D:\Program Files\Eclipse\Sumo"
            $env:LIBSUMO_AS_TRACI     = "1"
            $env:OPENBLAS_NUM_THREADS = "1"
            $env:OMP_NUM_THREADS      = "1"
            $env:MKL_NUM_THREADS      = "1"
            & $py train_ppo.py `
                --save_dir $sd `
                --mode multi `
                --reward_type $rw `
                --obs_mode baseline `
                --seed $s `
                --total_steps $ts `
                --lr 3e-4 *> $log
        } -ArgumentList $Python, $saveDir, $reward, $seed, $TotalSteps, $logFile, $WorkDir
    } else {
        $job = Start-Job -ScriptBlock {
            param($py, $al, $sd, $rw, $s, $ts, $log, $wd)
            Set-Location $wd
            $env:SUMO_HOME            = "D:\Program Files\Eclipse\Sumo"
            $env:LIBSUMO_AS_TRACI     = "1"
            $env:OPENBLAS_NUM_THREADS = "1"
            $env:OMP_NUM_THREADS      = "1"
            $env:MKL_NUM_THREADS      = "1"
            & $py train_dqn.py `
                --algo $al `
                --save_dir $sd `
                --reward_type $rw `
                --mode multi `
                --seed $s `
                --total_steps $ts *> $log
        } -ArgumentList $Python, $algo, $saveDir, $reward, $seed, $TotalSteps, $logFile, $WorkDir
    }

    $ActiveJobs += [PSCustomObject]@{ Job=$job; Label=$label }
}

Write-Host ""
Write-Host "Tat ca jobs da duoc tao. Cho hoan tat..." -ForegroundColor Yellow
while ($ActiveJobs.Count -gt 0) { Wait-OneJob }

$elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  HOAN TAT - $DoneCount/$($AllRuns.Count) models - $elapsed phut" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "Buoc tiep theo (evaluate):"
Write-Host "  python evaluate_all.py --scope multi --skip_ae --save_dir ./results_multi"
