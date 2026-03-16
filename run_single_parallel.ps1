# run_single_parallel.ps1 - Train PPO single intersection song song
#
# Su dung:
#   .\run_single_parallel.ps1          # mac dinh 3 jobs song song
#   .\run_single_parallel.ps1 -Jobs 4  # 4 jobs song song

param(
    [int]$Jobs = 3   # So luong runs chay cung luc
)

$ErrorActionPreference = "Stop"
$Python    = (Resolve-Path ".venv\Scripts\python.exe").Path
$ExpDir    = (Resolve-Path ".").Path + "\experiments"
$LogDir    = (Resolve-Path ".").Path + "\logs_parallel"

New-Item -ItemType Directory -Force -Path $ExpDir  | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir  | Out-Null

$PPORewards = @("queue", "diff-waiting-time", "pressure", "average-speed")
$Seeds      = @(42, 123, 777)
$TotalSteps = 200000

# Tao danh sach tat ca runs can thuc hien
$AllRuns = @()
foreach ($reward in $PPORewards) {
    foreach ($seed in $Seeds) {
        $saveDir = "$ExpDir\ppo_${reward}_single_s${seed}"
        if (-not (Test-Path "$saveDir\ppo_final_model.zip")) {
            $AllRuns += [PSCustomObject]@{
                Reward  = $reward
                Seed    = $seed
                SaveDir = $saveDir
            }
        } else {
            Write-Host "[SKIP] ppo_${reward}_single_s${seed} da co model." -ForegroundColor DarkGray
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

$StartTime   = Get-Date
$ActiveJobs  = @()
$RunIdx      = 0
$DoneCount   = 0

function Wait-OneJob {
    # Cho den khi co it nhat 1 job ket thuc, in ket qua
    while ($true) {
        $finished = $ActiveJobs | Where-Object { $_.Job.State -in "Completed","Failed","Stopped" }
        if ($finished) {
            foreach ($item in $finished) {
                $out = Receive-Job $item.Job -ErrorAction SilentlyContinue
                $status = if ($item.Job.State -eq "Completed") { "[OK]" } else { "[FAIL]" }
                $color  = if ($item.Job.State -eq "Completed") { "Green" } else { "Red" }
                Write-Host "$status ppo_$($item.Reward)_single_s$($item.Seed)" -ForegroundColor $color
                if ($out) { $out | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray } }
                Remove-Job $item.Job -Force
                $script:ActiveJobs = $script:ActiveJobs | Where-Object { $_.Job.Id -ne $item.Job.Id }
                $script:DoneCount++
            }
            return
        }
        Start-Sleep -Seconds 10
    }
}

foreach ($run in $AllRuns) {
    # Neu dang chay du Jobs jobs, cho 1 cai xong
    while ($ActiveJobs.Count -ge $Jobs) {
        Wait-OneJob
    }

    $RunIdx++
    $logFile = "$LogDir\ppo_$($run.Reward)_single_s$($run.Seed).log"

    Write-Host "[$RunIdx/$($AllRuns.Count)] Start: ppo_$($run.Reward)_single_s$($run.Seed)" -ForegroundColor Cyan

    $reward  = $run.Reward
    $seed    = $run.Seed
    $saveDir = $run.SaveDir

    $job = Start-Job -ScriptBlock {
        param($py, $sd, $rw, $s, $ts, $log)
        & $py train_ppo.py `
            --save_dir $sd `
            --mode single `
            --reward_type $rw `
            --seed $s `
            --total_steps $ts `
            --obs_mode raw `
            --lr 3e-4 *> $log
        $LASTEXITCODE
    } -ArgumentList $Python, $saveDir, $reward, $seed, $TotalSteps, $logFile `
      -WorkingDirectory (Resolve-Path ".").Path

    $ActiveJobs += [PSCustomObject]@{
        Job    = $job
        Reward = $reward
        Seed   = $seed
    }
}

# Cho tat ca jobs con lai ket thuc
Write-Host ""
Write-Host "Tat ca jobs da duoc tao. Cho hoan tat..." -ForegroundColor Yellow
while ($ActiveJobs.Count -gt 0) {
    Wait-OneJob
}

$elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  HOAN TAT - $DoneCount models - $elapsed phut" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "Log files: $LogDir"
Write-Host ""
Write-Host "Buoc tiep theo:"
Write-Host "  python evaluate_all.py --models_dir ./experiments --save_dir ./results --scope both --skip_ae"
