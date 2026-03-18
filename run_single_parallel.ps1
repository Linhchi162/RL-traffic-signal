# run_single_parallel.ps1 — Train toan bo single-intersection (PPO + DQN + DDQN)
#
# Su dung:
#   .\run_single_parallel.ps1          # mac dinh 3 jobs song song
#   .\run_single_parallel.ps1 -Jobs 4  # 4 jobs song song
#
# Models se train:
#   PPO  : queue, pressure, wait-clip  x  s42/123/777  =  9
#   DQN  : queue, pressure, wait-clip  x  s42/123/777  =  9
#   DDQN : queue, pressure, wait-clip  x  s42/123/777  =  9
#   TONG : 27 models

param(
    [int]$Jobs = 3
)

$ErrorActionPreference = "Stop"
$Python  = (Resolve-Path ".venv\Scripts\python.exe").Path
$ExpDir  = (Resolve-Path ".").Path + "\experiments"
$LogDir  = (Resolve-Path ".").Path + "\logs_parallel"

New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Seeds = @(42, 123, 777)

# ---------------------------------------------------------------------------
# Tao danh sach tat ca runs
# ---------------------------------------------------------------------------
$AllRuns = @()

# --- PPO ---
$PPORewards = @("queue", "pressure", "wait-clip")
foreach ($reward in $PPORewards) {
    foreach ($seed in $Seeds) {
        $tag     = "ppo_${reward}_single_s${seed}"
        $saveDir = "$ExpDir\$tag"
        if (Test-Path "$saveDir\ppo_final_model.zip") {
            Write-Host "[SKIP] $tag" -ForegroundColor DarkGray
        } else {
            $AllRuns += [PSCustomObject]@{
                Algo    = "ppo"
                Reward  = $reward
                Seed    = $seed
                Tag     = $tag
                SaveDir = $saveDir
            }
        }
    }
}

# --- DQN + DDQN ---
$DQNRewards = @("queue", "pressure", "wait-clip")
foreach ($algo in @("dqn", "ddqn")) {
    foreach ($reward in $DQNRewards) {
        foreach ($seed in $Seeds) {
            $tag     = "${algo}_${reward}_s${seed}"
            $saveDir = "$ExpDir\$tag"
            if (Test-Path "$saveDir\dqn_final_model.zip") {
                Write-Host "[SKIP] $tag" -ForegroundColor DarkGray
            } else {
                $AllRuns += [PSCustomObject]@{
                    Algo    = $algo
                    Reward  = $reward
                    Seed    = $seed
                    Tag     = $tag
                    SaveDir = $saveDir
                }
            }
        }
    }
}

if ($AllRuns.Count -eq 0) {
    Write-Host "Tat ca models da duoc train." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  Train $($AllRuns.Count) models | $Jobs song song" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------
$StartTime  = Get-Date
$ActiveJobs = @()
$RunIdx     = 0
$DoneCount  = 0

function Wait-OneJob {
    while ($true) {
        $finished = $ActiveJobs | Where-Object { $_.Job.State -in "Completed","Failed","Stopped" }
        if ($finished) {
            foreach ($item in $finished) {
                $out    = Receive-Job $item.Job -ErrorAction SilentlyContinue
                $ok     = $item.Job.State -eq "Completed"
                $status = if ($ok) { "[OK]  " } else { "[FAIL]" }
                $color  = if ($ok) { "Green" } else { "Red" }
                Write-Host "$status $($item.Tag)" -ForegroundColor $color
                if (-not $ok -and $out) { $out | Select-Object -Last 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray } }
                Remove-Job $item.Job -Force
                $script:ActiveJobs = @($script:ActiveJobs | Where-Object { $_.Job.Id -ne $item.Job.Id })
                $script:DoneCount++
            }
            return
        }
        Start-Sleep -Seconds 10
    }
}

$workDir = (Resolve-Path ".").Path

foreach ($run in $AllRuns) {
    while ($ActiveJobs.Count -ge $Jobs) { Wait-OneJob }

    $RunIdx++
    $logFile = "$LogDir\$($run.Tag).log"
    Write-Host "[$RunIdx/$($AllRuns.Count)] Start: $($run.Tag)" -ForegroundColor Cyan

    $algo    = $run.Algo
    $reward  = $run.Reward
    $seed    = $run.Seed
    $saveDir = $run.SaveDir
    $tag     = $run.Tag

    if ($algo -eq "ppo") {
        $job = Start-Job -ScriptBlock {
            param($py, $sd, $rw, $s, $log, $wd)
            Set-Location $wd
            & $py train_ppo.py `
                --save_dir $sd `
                --mode single `
                --reward_type $rw `
                --seed $s `
                --total_steps 200000 `
                --obs_mode raw `
                --lr 3e-4 *> $log
            $LASTEXITCODE
        } -ArgumentList $Python, $saveDir, $reward, $seed, $logFile, $workDir
    } else {
        $job = Start-Job -ScriptBlock {
            param($py, $sd, $algo, $rw, $s, $log, $wd)
            Set-Location $wd
            & $py train_dqn.py `
                --save_dir $sd `
                --algo $algo `
                --reward_type $rw `
                --mode single `
                --seed $s `
                --total_steps 200000 *> $log
            $LASTEXITCODE
        } -ArgumentList $Python, $saveDir, $algo, $reward, $seed, $logFile, $workDir
    }

    $ActiveJobs += [PSCustomObject]@{ Job = $job; Tag = $tag }
}

Write-Host ""
Write-Host "Tat ca jobs da duoc tao. Cho hoan tat..." -ForegroundColor Yellow
while ($ActiveJobs.Count -gt 0) { Wait-OneJob }

$elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  HOAN TAT — $DoneCount/$($AllRuns.Count) models — $elapsed phut" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "Log files : $LogDir"
Write-Host "Buoc tiep : python evaluate_all.py --models_dir ./experiments --save_dir ./results --scope single --skip_ae"
