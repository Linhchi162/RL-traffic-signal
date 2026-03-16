import pandas as pd, numpy as np

rl = pd.read_csv('eval_multi/eval_rl_multi.csv')
fx = pd.read_csv('eval_multi/eval_fixed_multi.csv')

print('='*52)
print('  SO SANH: RL vs Fixed-time (4 nut giao, 5000s)')
print('='*52)

metrics = [
    ('total_queue',     'Hang cho TB (xe)'),
    ('vehicles_stopped','Xe dung TB'),
    ('total_wait',      'Tong cho (s)'),
    ('mean_speed',      'Toc do TB (m/s)'),
    ('step_reward',     'Reward TB/step'),
]
for col, label in metrics:
    rl_v = rl[col].mean()
    fx_v = fx[col].mean()
    diff = (rl_v - fx_v) / max(abs(fx_v), 1e-9) * 100
    sign = '+' if diff > 0 else ''
    print(f'  {label:<25} RL={rl_v:.3f}  Fixed={fx_v:.3f}  ({sign}{diff:.1f}%)')

print('='*52)
print(f'  Total reward RL    : {rl["step_reward"].sum():.3f}')
print(f'  Total reward Fixed : {fx["step_reward"].sum():.3f}')
