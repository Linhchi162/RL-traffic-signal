# run_all_training.ps1
# Chạy toàn bộ 9 training jobs: 3 obs_mode × 3 seeds
# Dùng n_envs=4 để tăng tốc thu thập mẫu (~3-4x nhanh hơn single env)
#
# Cách chạy:
#   cd D:\UniProject\RL-traffic-signal
#   .\run_all_training.ps1
#
# Ước tính thời gian: ~30-60 phút / job × 9 jobs ≈ 5-9 tiếng
# Để interrupt giữa chừng: Ctrl+C (model checkpoint vẫn được lưu)

$OBS_MODES = @("raw", "compressed_16d", "baseline")
$SEEDS     = @(42, 123, 777)
$N_ENVS    = 1      # Dùng 1 trên Windows (SubprocVecEnv hay bị deadlock với spawn)
$STEPS     = 100000

$total = $OBS_MODES.Count * $SEEDS.Count
$done  = 0
$start_all = Get-Date

foreach ($obs in $OBS_MODES) {
    foreach ($seed in $SEEDS) {
        $done++
        $save_dir = "experiments/ppo_${obs}_s${seed}"
        $elapsed_all = ((Get-Date) - $start_all).ToString("hh\:mm\:ss")

        Write-Host ""
        Write-Host "=" * 62
        Write-Host "  [$done/$total]  obs=$obs  seed=$seed"
        Write-Host "  Save : $save_dir"
        Write-Host "  Thoi gian da chay: $elapsed_all"
        Write-Host "=" * 62

        $t_start = Get-Date

        # -u: unbuffered stdout/stderr (output PPO hiện realtime)
        # KHÔNG dùng Measure-Command vì nó nuốt stdout
        python -u train_ppo.py `
            --save_dir  $save_dir `
            --obs_mode  $obs `
            --seed      $seed `
            --n_envs    $N_ENVS `
            --total_steps $STEPS `
            --mode      single

        $mins = [math]::Round(((Get-Date) - $t_start).TotalMinutes, 1)
        Write-Host "  => Xong sau $mins phut"
    }
}

$total_elapsed = ((Get-Date) - $start_all).ToString("hh\:mm\:ss")
Write-Host ""
Write-Host "=" * 62
Write-Host "  HOAN TAT: $total jobs"
Write-Host "  Tong thoi gian: $total_elapsed"
Write-Host "  Ket qua luu tai: experiments/ppo_<obs>_s<seed>/"
Write-Host "=" * 62
