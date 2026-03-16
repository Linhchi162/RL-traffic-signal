# run_single_experiments.ps1 - Train PPO tren kich ban paper (single intersection)
#
# Su dung:
#   .\run_single_experiments.ps1
#
# Output:
#   experiments/ppo_queue_single_s42/
#   experiments/ppo_queue_single_s123/
#   experiments/ppo_queue_single_s777/
#   experiments/ppo_diff-waiting-time_single_s42/  ... (etc)

$ErrorActionPreference = "Stop"
$Python   = ".venv\Scripts\python.exe"
$ExpDir   = "experiments"

# Chi train 2 reward quan trong nhat de so sanh voi paper
$PPORewards = @("queue", "diff-waiting-time", "pressure", "average-speed")
$Seeds      = @(42, 123, 777)
$TotalSteps = 200000

if (-not (Test-Path $Python)) {
    Write-Error "Khong tim thay $Python"
    exit 1
}

New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null

$TotalRuns  = $PPORewards.Count * $Seeds.Count
$RunIdx     = 0
$StartTime  = Get-Date

Write-Host ""
Write-Host ("=" * 55) -ForegroundColor Green
Write-Host "  PPO SINGLE INTERSECTION - $($PPORewards.Count) rewards x $($Seeds.Count) seeds" -ForegroundColor Green
Write-Host ("=" * 55) -ForegroundColor Green

foreach ($reward in $PPORewards) {
    foreach ($seed in $Seeds) {
        $RunIdx++
        $saveDir = "$ExpDir\ppo_${reward}_single_s${seed}"

        if (Test-Path "$saveDir\ppo_final_model.zip") {
            Write-Host "[$RunIdx/$TotalRuns] [SKIP] $saveDir da co model." -ForegroundColor DarkGray
            continue
        }

        $elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
        Write-Host ""
        Write-Host "[$RunIdx/$TotalRuns] [$elapsed min] PPO | mode=single | reward=$reward | seed=$seed" -ForegroundColor Cyan
        Write-Host ("-" * 55) -ForegroundColor DarkGray

        & $Python train_ppo.py `
            --save_dir $saveDir `
            --mode single `
            --reward_type $reward `
            --seed $seed `
            --total_steps $TotalSteps `
            --obs_mode raw `
            --lr 3e-4

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [WARN] That bai (exit $LASTEXITCODE). Tiep tuc..." -ForegroundColor Yellow
        } else {
            Write-Host "  [OK] $saveDir\ppo_final_model.zip" -ForegroundColor Green
        }
    }
}

$elapsed = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Write-Host ""
Write-Host ("=" * 55) -ForegroundColor Green
Write-Host "  HOAN TAT - $elapsed phut" -ForegroundColor Green
Write-Host ("=" * 55) -ForegroundColor Green
Write-Host ""
Write-Host "Buoc tiep theo - Danh gia scope=both:" -ForegroundColor Yellow
Write-Host "  python evaluate_all.py --models_dir ./experiments --save_dir ./results --scope both --skip_ae" -ForegroundColor White
